import shutil
import os
import open3d as o3d
import pandas
import json
import numpy as np
from tomlkit import TOMLDocument
from tqdm import tqdm
from misc_utils import get_data_path, id_set, check_id_is_in_objs, npy_to_ply, get_action_key_from_dt
from localization_utils import create_ply_intercept, create_obj_intercept, process_transform, get_obb_hull, pt_in_cvex_hull, transform_point, create_single_object_intercept, create_single_ply_intercept
from input_utils import create_net_inputs, create_net_inputs_helper
from bbox_utils import create_bb, create_seg_dict, get_sem_seg, get_scan_bboxes, get_create_ss, get_bb_bounds
from PIL import Image
from collections import Counter
from scipy.spatial import ConvexHull
from baseline_utils import create_baseline_data, get_visibility_bb_from_seq_path, get_score_dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DataManipulator:
    def __init__(self, path):
        self.valid_hex = '0123456789ABCDEF'.__contains__

        if path is not None:
            self.use_3rscan = True
            self.path = os.path.join(path, '3rscan')

            f = open(os.path.join(path, '3RScan.json'))
            self._rscan = json.load(f)
            f.close()

            f = open(os.path.join(path, 'objects.json'))
            self._objects = json.load(f)
            f.close()

            f = open(os.path.join(path, 'relationship.json'))
            self._relationships = json.load(f)
            f.close()
        else:
            self.use_3rscan = False

        self.cap_classes = [
            'moved',
            'cluttered',
            'rotated',
            'added',
            'shifted',
            'removed',
            'rearranged',
            'decorated',
            'tidied up',
            'open',
            'closed'
        ]
    
    def get_sequence_path(self, scan_name, old=False):
        dir_name = 'sequence/'
        if old:
            dir_name = 'old_sequence/'
        return os.path.join(
            self.path, scan_name, dir_name
        )

    def find_scan_reference(self, scan_name):
        scan_i = 0
        while self._rscan[scan_i]['reference'] != scan_name:
            scan_i += 1
        return self._rscan[scan_i]

    def find_scan_objects(self, scan_name):
        scan_i = 0
        while self._objects['scans'][scan_i]['scan'] != scan_name:
            scan_i += 1
        return self._objects['scans'][scan_i]

    def find_scan_relationships(self, scan_name):
        scan_i = 0
        while self._relationships['scans'][scan_i]['scan'] != scan_name:
            scan_i += 1
        return self._relationships['scans'][scan_i]
    
    def find_chg_scan(self, ref, chg):
        if type(ref) == str:
            ref = self.find_scan_reference(ref)

        for s in ref['scans']:
            if s['reference'] == chg:
                return s
        
        return -1

    def find_chg_scan_from(self, chg):
        for r in self._rscan:
            found_scan = self.find_chg_scan(r, chg)
            
            if found_scan != -1:
                return found_scan, r['reference']

    def pick_random_scans(self, only_ref = False):
        randid = np.random.randint(len(self._rscan))
        while self._rscan[randid]['type'] == 'test':
            randid = np.random.randint(len(self._rscan))
        reference_scan = self._rscan[randid]['reference']
        changed_scan = self._rscan[randid]['scans'][np.random.randint(len(self._rscan[randid]['scans']))]
        if only_ref:
            changed_scan = changed_scan['reference']
        return reference_scan, changed_scan
    
    def get_global_id_set_of(self, labels):
        ids = set()

        for a,b in self.scan_pairs():
            a_objs = self.find_scan_objects(a)['objects']
            b_objs = self.find_scan_objects(b)['objects']

            ids.update([int(x['global_id']) for x in a_objs if x['label'] in labels])
            ids.update([int(x['global_id']) for x in b_objs if x['label'] in labels])
        
        return ids
    
    def find_man_cap_folder(self, c_path, chg):
        for x in os.listdir(c_path):
            if x.endswith(chg):
                return x

    def create_located_ply(self, remove_wall_ceiling=False, caption_path=None):
        g_id_set = None
        if remove_wall_ceiling:
            g_id_set = self.get_global_id_set_of(['wall', 'ceiling'])

        for scan in tqdm(self._rscan):
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']
                    
                    output_path = None
                    if caption_path is not None:
                        output_path = os.path.join(
                            caption_path, self.find_man_cap_folder(caption_path, chg)
                        )

                    create_ply_intercept(ref, chg, rescan, g_id_set, output_path)
    
    def create_located_obj(self, remove_wall_ceiling=False, caption_path=None):
        for scan in tqdm(self._rscan):
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']

                    labels = None
                    if remove_wall_ceiling:
                        labels = ['wall', 'ceiling']
                    output_path = None
                    if caption_path is not None:
                        output_path = os.path.join(
                            caption_path, self.find_man_cap_folder(caption_path, chg)
                        )

                    create_obj_intercept(ref, chg, rescan, labels, output_path)
    
    def get_located_ply_helper(self, d, ref=True):
        prefix = self.get_prefix(ref)
        
        cloud_p = os.path.join(d, f'new_{prefix}_located_ascii.ply')
        csv_p = os.path.join(d, f'new_{prefix}_located_pyntcloud.csv')
        cloud = o3d.io.read_triangle_mesh(cloud_p, True)
        csv = pandas.read_csv(csv_p)

        return cloud, csv

    def get_located_ply(self, chg):
        ply_path = os.path.join(get_data_path(chg, self.path), 'isolated_ply/')

        ref_cloud, ref_csv = self.get_located_ply_helper(
            ply_path, True
        )

        chg_cloud, chg_csv = self.get_located_ply_helper(
            ply_path, False
        )

        return ref_cloud, chg_cloud, ref_csv, chg_csv
    
    def get_located_obj_helper(self, d, ref=True, name=None):
        prefix = self.get_prefix(ref)
        
        if name is None:
            name = f'{prefix}_located_mesh'
        
        obj_p = os.path.join(d, f'{prefix}/{name}.obj')
        tex_p = os.path.join(d, f'{prefix}/{name}_0.png')
        obj = o3d.io.read_triangle_mesh(obj_p, True)
        tex = Image.open(tex_p)
        
        return obj, tex

    def get_located_obj(self, chg):
        obj_path = os.path.join(get_data_path(chg, self.path), 'isolated_obj/')

        ref_obj, ref_tex = self.get_located_obj_helper(
            obj_path, True
        )
        chg_obj, chg_tex = self.get_located_obj_helper(
            obj_path, False
        )

        return ref_obj, chg_obj, ref_tex, chg_tex
    
    def create_sampled_inputs(self, sample_size):
        errs = []
        for scan in tqdm(self._rscan):
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']
                    located_ply = self.get_located_ply(chg)
                    located_obj = self.get_located_obj(chg)
                    try:
                        create_net_inputs(ref, chg, located_ply, located_obj, sample_size)
                    except Exception as e: 
                        print(e)
                        errs.append(tuple([ref, chg]))
        
        return errs

    def get_changes_lists(self, ref, chg):
        original_objects = self.find_scan_objects(ref)['objects']
        changed_objects = self.find_scan_objects(chg)['objects']
        ref_scan = self.find_scan_reference(ref)
        removed, moved, added = [], [], []
        changed_scan = self.find_chg_scan(ref, chg)
        
        for j in changed_scan['removed']:
            if check_id_is_in_objs(j, original_objects):
                removed.append(j)

        for j in changed_scan['rigid']:
            if check_id_is_in_objs(j['instance_reference'], original_objects) and check_id_is_in_objs(j['instance_rescan'], changed_objects):
                moved.append(j['instance_reference'])

        for j in set.difference(id_set(changed_objects), id_set(original_objects)):
            if check_id_is_in_objs(j, changed_objects):
                added.append(int(j))
        
        return removed, moved, added

    def process_ids(self, ref, chg, removed, moved, added):
        ref_objs = self.find_scan_objects(ref)['objects']
        chg_objs = self.find_scan_objects(chg)['objects']

        labels = {'removed': [], 'moved': [], 'added': [], 'still': []}

        for obj in ref_objs:
            obj_id = obj['id']
            obj_gid = obj['global_id']
            obj_label = obj['label']
            obj_tuple = {'id': obj_id, 'global_id': obj_gid, 'label': obj_label}

            if int(obj_id) in removed:
                labels['removed'].append(obj_tuple)
            elif int(obj_id) in moved:
                labels['moved'].append(obj_tuple)
            else:
                labels['still'].append(obj_tuple)
    
        for obj in chg_objs:
            obj_id = obj['id']
            obj_gid = obj['global_id']
            obj_label = obj['label']
            obj_tuple = {'id': obj_id, 'global_id': obj_gid, 'label': obj_label}

            if int(obj_id) in added:
                labels['added'].append(obj_tuple)
    
        return labels

    def create_net_outputs(self, ref, chg):
        r, m, a = self.get_changes_lists(ref, chg)
        dt = self.process_ids(ref, chg, r, m, a)
        
        base_save_path = os.path.join(get_data_path(chg, self.path), 'net_outputs/')
        os.makedirs(base_save_path, exist_ok=True)
        json_path = os.path.join(base_save_path, 'object_lists.json')
        
        with open(json_path, 'w') as fp:
            json.dump(dt, fp, indent=2)

    def create_classifier_outputs(self):
        errs = []
        for scan in tqdm(self._rscan):
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']
                    try:
                        self.create_net_outputs(ref, chg)
                    except Exception as e: 
                        print(e)
                        errs.append(tuple([ref, chg]))
        return errs

    def get_scene_list(self, dt):
        zeros, ones = [], []
        for k,v in dt.items():
            for o in v:
                if k == 'still':
                    zeros.append(int(o['global_id']))
                else:
                    ones.append(int(o['global_id']))
        zeros = set(zeros)
        ones = set(ones)
        maybe = zeros.intersection(ones)
        scene_list = [[x, 0] for x in zeros if x not in maybe]
        scene_list.extend([[x, 1] for x in ones])
        # scene_list.extend([[x, 1] for x in ones if x not in maybe])
        
        return scene_list
    
    def get_outputs_class_dict(self, ssize=75000):
        errors = []
        out_dict = {}

        for scan in tqdm(self._rscan):
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']
                    base_path = os.path.join(get_data_path(chg, self.path), f'net_inputs_sampled_{ssize}/')
                    ref_path = os.path.join(base_path, 'ref_pointcloud.npy')
                    chg_path = os.path.join(base_path, 'chg_pointcloud.npy')
                    if os.path.exists(ref_path) and os.path.exists(chg_path):
                        f = open(os.path.join(get_data_path(chg, self.path), 'net_outputs/object_lists.json'))
                        dummy = json.load(f)
                        f.close()
                        for k,v in dummy.items():
                            for o in v:
                                out_dict[o['global_id']] = o['label']

                    else:
                        errors.append(chg)
        return out_dict

    def get_classifier_outputs(self, ssize):
        errors = []
        outputs = []

        for scan in tqdm(self._rscan):
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']
                    base_path = os.path.join(get_data_path(chg, self.path), f'net_inputs_sampled_{ssize}/')
                    ref_path = os.path.join(base_path, 'ref_pointcloud.npy')
                    chg_path = os.path.join(base_path, 'chg_pointcloud.npy')
                    if os.path.exists(ref_path) and os.path.exists(chg_path):
                        f = open(os.path.join(get_data_path(chg, self.path), 'net_outputs/object_lists.json'))
                        dummy = json.load(f)
                        f.close()
                        outputs.append(self.get_scene_list(dummy))
                    else:
                        errors.append(chg)
        return outputs
    
    def get_scan_list(self):
        chg_ls = []
        for scan in tqdm(self._rscan):
            if scan['type'] != 'test':
                for rescan in scan['scans']:
                    chg_ls.append(rescan['reference'])
        return chg_ls

    def get_prefix(self, ref):
        if ref:
            return 'ref'
        return 'chg'

    def get_classifier_inputs(self, ssize):
        errors = []
        inputs_np = np.empty((903, 2, ssize, 8))
        counter = 0

        for scan in tqdm(self._rscan):
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']
                    base_path = os.path.join(get_data_path(chg, self.path), 'net_inputs_sampled/')
                    ref_path = os.path.join(base_path, 'ref_pointcloud.npy')
                    chg_path = os.path.join(base_path, 'chg_pointcloud.npy')
                    if os.path.exists(ref_path) and os.path.exists(chg_path):
                        inputs_np[counter, 0] = np.load(ref_path)
                        inputs_np[counter, 1] = np.load(chg_path)
                        counter += 1
                    else:
                        errors.append(chg)
        return inputs_np

    def get_classifier_mask(self, c_path, c_dt, ssize, is_ref):
        base_path = os.path.join(
            self.get_save_manual_dir(c_path, c_dt), 'masks'
        )

        prefix = self.get_prefix(is_ref)

        return np.load(
            os.path.join(
                base_path,
                f'mask_{prefix}_{ssize}.npy'
            )
        )

    def get_classifier_masks(self, c_path, c_dt, ssize):
        m_ref = self.get_classifier_mask(c_path, c_dt, ssize, True)
        m_chg = self.get_classifier_mask(c_path, c_dt, ssize, False)

        return m_ref, m_chg

    def get_classifier_input_from_(self, chg, ssize):
        if str.startswith(chg, 'X'):
            chg = chg[2:]
        base_path = os.path.join(get_data_path(chg, self.path), f'net_inputs_sampled_{ssize}/')
        ref_path = os.path.join(base_path, 'ref_pointcloud.npy')
        chg_path = os.path.join(base_path, 'chg_pointcloud.npy')
        ref_np = np.load(ref_path).reshape(-1,ssize,8)
        chg_np = np.load(chg_path).reshape(-1,ssize,8)
        return np.vstack([ref_np, chg_np])

    def get_single_classifier_input(self, chg, ssize, is_ref):
        if str.startswith(chg, 'X'):
            chg = chg[2:]

        base_path = os.path.join(get_data_path(chg, self.path), f'net_inputs_sampled_{ssize}/')
        if is_ref:
            path = os.path.join(base_path, 'ref_pointcloud.npy')
        else:
            path = os.path.join(base_path, 'chg_pointcloud.npy')
            
        return np.load(path).reshape(ssize,-1)

    def create_objects_list(self, ref, save_path):
        f = open(save_path, 'w')
        objs = self.find_scan_objects(ref)
        for o in objs['objects']:
            i = o['id']
            l = o['label']
            c = o['ply_color']
            f.write(f'<p style="color:{c}" /> {l} {i} </p> \n')
        f.close()    
    
    def create_bboxes(self, ref, chg, is_chg, sample_size=75000, process_bar=True, save_path=None):
        to_do = ref
        if is_chg:
            to_do = chg
        seg_dict = create_seg_dict(get_sem_seg(to_do, self.path))

        keyword = 'ref'
        transform = None
        if is_chg:
            keyword = 'chg'
            chg_scan = self.find_chg_scan(ref, chg)
            transform = process_transform(chg_scan['transform'])
        
        cloud_path = os.path.join(get_data_path(chg, self.path), f'net_inputs_sampled_{str(sample_size)}/{keyword}_pointcloud.npy')
        hull = get_obb_hull(o3d=npy_to_ply(cloud_path))
        
        if save_path == None:
            d = os.path.join(get_data_path(chg, self.path), f'{keyword}_bboxes/')
        else:
            d = save_path
        os.makedirs(d, exist_ok=True)
        pbar = seg_dict.keys()
        if process_bar:
            pbar = tqdm(seg_dict.keys())
        for k in pbar:
            segd_tuple = seg_dict[k][2]
            centroid = segd_tuple['centroid']
            if is_chg:
                transform_point(centroid, transform)
            if(pt_in_cvex_hull(hull, centroid)):
                create_bb(os.path.join(d, f'{k}_bbox'), segd_tuple, transform)

    def create_dataset_bboxes(self, sample_size=75000):
        errs = []
        for scan in tqdm(self._rscan):
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']
                    
                    try:
                        self.create_bboxes(ref, chg, False, sample_size, process_bar=False)
                        self.create_bboxes(ref, chg, True, sample_size, process_bar=False)
                    except Exception as e: 
                        print(e)
                        errs.append(tuple([ref, chg]))
        
        return errs
    
    def get_usable_labels(self):
        all_k = []
        don_t_use = ['wall', 'floor', 'window', 'door', 'item', 'object']

        for scan in self._rscan:
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']
                    all_k.extend(list(create_seg_dict(get_sem_seg(ref)).keys()))
                    all_k.extend(list(create_seg_dict(get_sem_seg(chg)).keys()))
        
        c = dict(Counter([x.split('_')[1] for x in all_k]))
        return [k for k, v in list(c.items()) if v > 100 and k not in don_t_use]

    def create_id_dict(self, scan):
        objs = self.find_scan_objects(scan)['objects']
        return {str(x['id']): str(x['global_id']) for x in objs}

    def create_dataset_bboxes_corners(self):
        bb_corn_arr = []
        glob_id_arr = []
        refs = []
        to_use = self.get_usable_labels()

        for scan in tqdm(self._rscan):
            if scan['type'] != 'test': # test rescan data doesn't have ply objects
                for rescan in scan['scans']:
                    ref = scan['reference']
                    chg = rescan['reference']

                    lab_dict = self.create_id_dict(ref)
                    scan_bb_c0 = get_scan_bboxes(scan=ref, lab_dict=lab_dict, to_use=to_use)
                    if scan_bb_c0 is not None:
                        bb_corn_arr.append(scan_bb_c0[0])
                        glob_id_arr.append(scan_bb_c0[1])
                        refs.append((True, chg))

                    lab_dict = self.create_id_dict(chg)
                    transform = process_transform(rescan['transform'])
                    scan_bb_c1 = get_scan_bboxes(scan=chg, lab_dict=lab_dict, to_use=to_use, transform=transform)
                    if scan_bb_c1 is not None:
                        bb_corn_arr.append(scan_bb_c1[0])
                        glob_id_arr.append(scan_bb_c1[1])
                        refs.append((False, chg))

        return bb_corn_arr, glob_id_arr, refs

    def create_mask_helper(self, ref, index, label, transform, c_in, ssize):
        segd = get_create_ss(ref)
        try:
            segd_tuple = segd[
                f"{index}_{label}"
            ][2]
        except:
            print(ref, index, label)
            return None
            # raise RuntimeError('RE')
        bb = get_bb_bounds(segd_tuple, transform)
        hull = ConvexHull(np.array(bb))

        mask = np.empty((ssize,), dtype=int)
        for i,x in enumerate(c_in):
            mask[i] = int(
                pt_in_cvex_hull(hull, x[:3])
            )
        return mask
    
    def get_single_npy_path(self, c_dt, ref, cap_path, ssize):
        prefix = self.get_prefix(ref)
        return os.path.join(
            cap_path, c_dt['dir'], \
                get_action_key_from_dt(c_dt), \
                    'npy', f'{ssize}', f'{prefix}_{ssize}.npy')

    def get_mask_input(self, c_dt, ssize, cap_path, ref):
        if ssize > 30000:
            c_in = self.get_single_classifier_input(
                chg=c_dt['chg'], 
                ssize=ssize, 
                is_ref=ref
            )
        else:
            c_in = np.load(
                self.get_single_npy_path(
                    c_dt=c_dt,
                    ref=ref,
                    cap_path=cap_path,
                    ssize=ssize
                )
            )
        return c_in

    def create_masks(self, c_dt, ssize, cap_path):
        mask_ref = np.zeros((ssize,), dtype=int)
        mask_chg = np.zeros((ssize,), dtype=int)

        if c_dt['action'] != 'added':
            c_in = self.get_mask_input(
                c_dt=c_dt,
                ssize=ssize,
                cap_path=cap_path,
                ref=True
            )

            mask_ref = self.create_mask_helper(
                ref=c_dt['ref'],
                index=c_dt['id'],
                label=c_dt['label'],
                transform=None,
                c_in=c_in,
                ssize=ssize
            )

        if c_dt['action'] != 'removed':
            transform = process_transform(
                self.find_chg_scan(c_dt['ref'], c_dt['chg'])['transform']
            )

            c_in = self.get_mask_input(
                c_dt=c_dt,
                ssize=ssize,
                cap_path=cap_path,
                ref=False
            )

            mask_chg = self.create_mask_helper(
                ref=c_dt['chg'],
                index=c_dt['id'],
                label=c_dt['label'],
                transform=transform,
                c_in=c_in,
                ssize=ssize
            )
        
        return mask_ref, mask_chg
    
    def get_caption_from_dir(self, manual_path, dir):
        with open(os.path.join(manual_path, dir, 'caption.txt')) as f:
            lines = f.readlines()
        return [l[:-1].split(',') if l.endswith('\n') else l.split(',') for l in lines]
    
    def get_manual_captions_meta(self, c_path):
        dirs = [d for d in os.listdir(c_path) if os.path.isdir(os.path.join(c_path, d))]
        scans = [self.get_mc_meta_helper(c_path, d) for d in dirs]
        return dirs, scans
    
    def get_mc_meta_helper(self, c_path, d):
        with open(os.path.join(c_path, d, 'meta.txt')) as f:
            return f.readline().split(',')
    
    def get_all_labels(self):
        return set([o['label'] for s in self._objects['scans'] for o in s['objects']])
    
    def get_safety_check_error_str(self, c, d):
        return f"{c} is not a valid caption. directory is {d}."

    def manual_captions_safety_check(self, c_path):
        all_labs = self.get_all_labels()

        dirs, scans = self.get_manual_captions_meta(c_path)
        for d,s in zip(dirs,scans):
            for c in self.get_caption_from_dir(c_path, d):
                assert len(c) == 3, self.get_safety_check_error_str(c, d)
                lab, action, idx = c
                ref, chg = s

                assert lab in all_labs, self.get_safety_check_error_str(c, d)
                assert action in self.cap_classes, self.get_safety_check_error_str(c, d)
                assert idx.isnumeric(), self.get_safety_check_error_str(c, d)
                assert self.check_caption_meta(lab, action, idx, ref, chg), self.get_safety_check_error_str(c, d)
    
    def check_caption_meta(self, lab, action, idx, ref, chg):
        if action != 'added' and \
            (f'{idx},{lab}' not in [f'{x["id"]},{x["label"]}' for x in self.find_scan_objects(ref)['objects']]):
            return False
        if action != 'removed' and \
            (f'{idx},{lab}' not in [f'{x["id"]},{x["label"]}' for x in self.find_scan_objects(chg)['objects']]):
            return False
        return True

    def create_captions_dict(self, c_path, filter_ssize=-1, filter_baseline=-1, ref_split=None):
        if self.use_3rscan:
            self.manual_captions_safety_check(c_path)
        dirs, scans = self.get_manual_captions_meta(c_path)

        captions = [{
            'label': c[0],
            'action':c[1],
            'id': int(c[2]),
            'ref': s[0],
            'chg': s[1],
            'dir': d
            } for d,s in zip(dirs,scans) \
            for c in self.get_caption_from_dir(c_path, d)]
        
        if filter_ssize != -1:
            captions = [
                c for c in captions if self.c_dt_npy_exists(c, c_path, filter_ssize)
            ]
        if filter_baseline != -1:
            captions = [
                c for c in captions if self.baseline_exists(c, c_path, filter_baseline)
            ]
        if ref_split is not None:
            captions = [
                c for c in captions if c['ref'] in ref_split
            ]

        return captions
    
    def get_caption_split(self, c_path, filter_ssize=-1, filter_baseline=-1, ref_split=None, split=.2, split_path=None):
        if split_path is not None:
            tr_p = os.path.join(split_path, 'train_split.txt')
            te_p = os.path.join(split_path, 'test_split.txt')
            with open(tr_p) as f:
                tr = [x.replace('\n', '') for x in f.readlines()]
            with open(te_p) as f:
                te = [x.replace('\n', '') for x in f.readlines()]
            return tr, te
        
        return self.generate_caption_split(
            c_path=c_path,
            filter_ssize=filter_ssize,
            filter_baseline=filter_baseline,
            ref_split=ref_split,
            split=split
        )


    def generate_caption_split(self, c_path, filter_ssize=-1, filter_baseline=-1, ref_split=None, split=.2):
        captions = self.create_captions_dict(
            c_path=c_path,
            filter_ssize=filter_ssize,
            filter_baseline=filter_baseline,
            ref_split=ref_split
        )

        refs = list(set([d['ref'] for d in captions]))
        tr, te = train_test_split(refs, test_size=split)
        ltr, lte = len([c for c in captions if c['ref'] in tr]), len([c for c in captions if c['ref'] in te])
        while not ((split-.05) < lte/(ltr+lte) < (split+.05)):
            tr, te = train_test_split(refs, test_size=split)
            ltr, lte = len([c for c in captions if c['ref'] in tr]), len([c for c in captions if c['ref'] in te])
        
        return tr, te
    
    def get_captions_oh_encoder(self, c_path):
        captions = self.create_captions_dict(c_path)

        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(
            X=np.array([c['action'] for c in captions]).reshape((-1,1))
        )
        return onehot_encoder

    def get_c_dt_basepath(self, c, c_path):
        return os.path.join(c_path, c['dir'], get_action_key_from_dt(c))

    def get_c_dt_npy_path(self, c, c_path, ssize):
        return os.path.join(self.get_c_dt_basepath(c, c_path), 'npy', str(ssize))
    
    def get_c_dt_mask_path(self, c, c_path):
        return os.path.join(self.get_c_dt_basepath(c, c_path), 'masks')

    def c_dt_npy_exists(self, c, c_path, filter_ssize):
        return os.path.exists(
            self.get_c_dt_npy_path(c, c_path, filter_ssize)
        )
    
    def baseline_exists(self, c, c_path, filter_baseline):
        filter_baseline = filter_baseline-1
        return os.path.exists(
            os.path.join(c_path, c['dir'], get_action_key_from_dt(c), 'baseline', 'ref', f'bb_{filter_baseline}.txt')
        ) and os.path.exists(
            os.path.join(c_path, c['dir'], get_action_key_from_dt(c), 'baseline', 'chg', f'bb_{filter_baseline}.txt')
        )

    def both_c_dt_npy_exist(self, c, c_path, filter_ssize):
        return os.path.exists(
            os.path.join(self.get_c_dt_npy_path(c, c_path, filter_ssize), f'ref_{str(filter_ssize)}.npy')
        ) and os.path.exists(
            os.path.join(self.get_c_dt_npy_path(c, c_path, filter_ssize), f'chg_{str(filter_ssize)}.npy')
        )
    
    def both_c_dt_masks_exist(self, c, c_path, filter_ssize):
        return os.path.exists(
            os.path.join(self.get_c_dt_mask_path(c, c_path), f'mask_ref_{str(filter_ssize)}.npy')
        ) and os.path.exists(
            os.path.join(self.get_c_dt_mask_path(c, c_path), f'mask_chg_{str(filter_ssize)}.npy')
        )
    
    def c_dt_obj_exist(self, c, c_path):
        return os.path.exists(
            os.path.join(self.get_c_dt_basepath(c, c_path), 'isolated_obj')
        )
    
    def c_dt_ply_exist(self, c, c_path):
        return os.path.exists(
            os.path.join(self.get_c_dt_basepath(c, c_path), 'isolated_ply')
        )

    def c_dt_baseline_exist(self, c, c_path):
        return os.path.exists(
            os.path.join(self.get_c_dt_basepath(c, c_path), 'baseline')
        )
    
    def c_dt_baseline_exist(self, c, c_path):
        return os.path.exists(
            os.path.join(self.get_c_dt_basepath(c, c_path), 'baseline')
        )
    
    def create_manual_plys(self, c_path, enhance, incremental=False):
        captions = self.create_captions_dict(c_path)
        if incremental:
            captions = [c for c in captions if not self.c_dt_ply_exist(c, c_path)]
        for c_dt in tqdm(captions):
            self.create_manual_ply(
                c_dt, c_path, enhance
            )
    
    def get_segd_key_from_dt(self, c_dt):
        return f"{c_dt['id']}_{c_dt['label']}"

    def create_manual_ply(self, c_dt, c_path, enhance):
        np_bb_ref, np_bb_chg = self.get_np_bboxes(c_dt, enhance)

        create_single_ply_intercept(
            scan=c_dt['ref'],
            save_dir=os.path.join(
                self.get_save_manual_dir(c_path, c_dt),
                'isolated_ply',
                'ref'
            ),
            file_name='new_ref',
            np_bb=np_bb_ref
        )

        create_single_ply_intercept(
            scan=c_dt['chg'],
            save_dir=os.path.join(
                self.get_save_manual_dir(c_path, c_dt),
                'isolated_ply',
                'chg'
            ),
            file_name='new_chg',
            np_bb=np_bb_chg,
            chg_scan=self.find_chg_scan(c_dt['ref'], c_dt['chg'])
        )
    
    def create_manual_objs(self, c_path, enhance, incremental=False):
        captions = self.create_captions_dict(c_path)
        if incremental:
            captions = [c for c in captions if not self.c_dt_obj_exist(c, c_path)]
        for c_dt in tqdm(captions):
            try:
                self.create_manual_obj(
                    c_dt, c_path, enhance
                )
            except Exception as e:
                print(c_dt)
                raise(e)

    def get_np_bboxes(self, c_dt, enhance):
        if c_dt['action'] != 'added':
            segd = get_create_ss(c_dt['ref'])
            segd_tuple = segd[
                self.get_segd_key_from_dt(c_dt)
            ][2]
            np_bb_ref = np.array(get_bb_bounds(segd_tuple, None, enhance))

        if c_dt['action'] != 'removed':
            segd = get_create_ss(c_dt['chg'])
            segd_tuple = segd[
                self.get_segd_key_from_dt(c_dt)
            ][2]

            chg_scan = self.find_chg_scan(c_dt['ref'], c_dt['chg'])
            transform = process_transform(chg_scan['transform'])
            np_bb_chg = np.array(get_bb_bounds(segd_tuple, transform, enhance))
        else:
            np_bb_chg = np_bb_ref
        
        if c_dt['action'] == 'added':
            np_bb_ref = np_bb_chg
        
        return np_bb_ref, np_bb_chg

    def create_manual_obj(self, c_dt, c_path, enhance):
        np_bb_ref, np_bb_chg = self.get_np_bboxes(c_dt, enhance)

        create_single_object_intercept(
            scan=c_dt['ref'],
            save_dir=os.path.join(
                self.get_save_manual_dir(c_path, c_dt),
                'isolated_obj',
                'ref'
            ),
            file_name='mesh.obj',
            np_bb=np_bb_ref
        )
        
        create_single_object_intercept(
            scan=c_dt['chg'],
            save_dir=os.path.join(
                self.get_save_manual_dir(c_path, c_dt),
                'isolated_obj',
                'chg'
            ),
            file_name='mesh.obj',
            np_bb=np_bb_chg,
            chg_scan=self.find_chg_scan(c_dt['ref'], c_dt['chg'])
        )
    
    def get_save_manual_dir(self, c_path, c_dt):
        d = os.path.join(
            c_path,
            c_dt['dir'],
            get_action_key_from_dt(c_dt)
        )
        os.makedirs(d, exist_ok=True)
        return d

    def create_manual_masks(self, ssize, c_path, incremental=False):
        captions = self.create_captions_dict(c_path, filter_ssize=ssize)
        if incremental:
            captions = [c for c in captions if not self.both_c_dt_masks_exist(c, c_path, ssize)]
        
        for c_dt in tqdm(captions):
            m_r, m_c = self.create_masks(
                c_dt=c_dt, ssize=ssize, cap_path=c_path
            )

            new_dir = os.path.join(
                self.get_save_manual_dir(c_path, c_dt), 'masks'
            )
            os.makedirs(new_dir, exist_ok=True)

            if m_r is not None:
                np.save(
                    file=os.path.join(
                        new_dir, f'mask_ref_{ssize}.npy'),
                    arr=m_r
                )

            if m_c is not None:
                np.save(
                    file=os.path.join(
                        new_dir, f'mask_chg_{ssize}.npy'),
                    arr=m_c
                )

    def scan_pairs(self):
        return [(s['reference'], c['reference']) for s in self._rscan for c in s['scans'] if s['type'] != 'test']
    
    def create_sampled_single_objects_helper(self, ssize, base_dir, generate, ref):
        prefix = self.get_prefix(ref)

        if generate:
            cloud, csv = self.get_located_ply_helper(
                os.path.join(base_dir, 'isolated_ply', prefix), ref
            )
            obj, tex = self.get_located_obj_helper(
                os.path.join(base_dir, 'isolated_obj'), ref, 'mesh'
            )
            create_net_inputs_helper(
                cloud=cloud,
                csv=csv,
                obj=obj,
                tex=tex,
                ssize=ssize,
                save_dir=os.path.join(base_dir, 'npy', f'{ssize}'),
                file_name=f'{prefix}_{ssize}.npy'
            )
        else:
            save_dir = os.path.join(base_dir, 'npy', f'{ssize}')
            os.makedirs(save_dir, exist_ok=True)
            np.save(
                file=os.path.join(save_dir, f'{prefix}_{ssize}.npy'),
                arr=np.zeros((ssize,8))
            )

    def create_sampled_single_objects(self, ssize, c_path, incremental=False):
        captions = self.create_captions_dict(c_path)
        if incremental:
            captions = [c for c in captions if not self.both_c_dt_npy_exist(c, c_path, ssize)]

        for c_dt in tqdm(captions):
            base_dir = self.get_save_manual_dir(c_path, c_dt)

            try:
                self.create_sampled_single_objects_helper(
                    ssize=ssize,
                    base_dir=base_dir,
                    generate=True,
                    ref=True
                )

                self.create_sampled_single_objects_helper(
                    ssize=ssize,
                    base_dir=base_dir,
                    generate=True,
                    ref=False
                )
            except:
                pass
            
            if not self.both_c_dt_npy_exist(c_dt, c_path, ssize):
                self.wipe_npy_error(c_path, c_dt, ssize)
    
    def wipe_npy_error(self, c_path, c_dt, ssize):
        to_wipe = os.path.join(c_path, c_dt['dir'], get_action_key_from_dt(c_dt), 'npy', str(ssize))
        if os.path.exists(to_wipe) and os.path.isdir(to_wipe):
            shutil.rmtree(
                to_wipe
            )
    
    def get_baseline_score_seq(self, scan):
        seq_path = self.get_sequence_path(scan, True)
        new_seq_path = self.get_sequence_path(scan, False)
        vis, bboxes, poses = get_visibility_bb_from_seq_path(seq_path)
        return get_score_dict(vis, bboxes, poses), seq_path, new_seq_path

    def create_baseline_metadata(self, cap_path, incremental=False):
        captions = self.create_captions_dict(cap_path)
        if incremental:
            captions = [c for c in captions if not self.c_dt_baseline_exist(c, cap_path)]
        failures = []
        for c_dt in tqdm(captions):
            score_dict_r, seq_path_r, new_seq_path_r = \
                self.get_baseline_score_seq(c_dt['ref'])
            score_dict_c, seq_path_c, new_seq_path_c = \
                self.get_baseline_score_seq(c_dt['chg'])
            chg_scan = self.find_chg_scan(c_dt['ref'], c_dt['chg'])

            out_1 = create_baseline_data(
                ref=True,
                c_dt=c_dt,
                score_dict=score_dict_r,
                seq_path_a=seq_path_r,
                seq_path_b=seq_path_c,
                new_seq_path_b=new_seq_path_c,
                cap_path=cap_path,
                chg_scan=chg_scan
            )

            out_2 = create_baseline_data(
                ref=False,
                c_dt=c_dt,
                score_dict=score_dict_c,
                seq_path_a=seq_path_c,
                seq_path_b=seq_path_r,
                new_seq_path_b=new_seq_path_r,
                cap_path=cap_path,
                chg_scan=chg_scan
            )

            if (not out_1) or (not out_2):
                failures.append(c_dt)
        return failures
