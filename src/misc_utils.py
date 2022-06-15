import numpy as np
import torch
import GPUtil
import open3d as o3d
from datetime import datetime
import os


def get_data_path(ref, path=None):
    if path != None:
        return os.path.join(path, ref)
    return f'/local/crv/danich/3rscan/{ref}/'
    

def id_set(scan_objects):
    return set([x['id'] for x in scan_objects])


def check_id_is_in_objs(obj_id, objects):
    return (str(obj_id) in [x['id'] for x in objects])


def id_to_obj(index, scan_objects):
        i = 0
        while str(scan_objects[i]['id']) != str(index):
            i += 1
        return scan_objects[i]


def get_available_device(cpu=False, gpu=-1):
    if not cpu and torch.cuda.is_available():
        if gpu != -1:
            device_num = gpu
        else:
            device_num = GPUtil.getAvailable(order = 'memory', limit = 1)[0]
        device = torch.device(f'cuda:{device_num}')
    else:
        device = torch.device('cpu')

    return device


def npy_to_ply(path, rgb_div=255, save_path = '', npy=None):
    if npy is None:
        npy = np.load(path)
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(npy[:, :3])
    pcl.colors = o3d.utility.Vector3dVector(npy[:, 3:6]/rgb_div)
    if save_path != '':
        o3d.io.write_point_cloud(save_path, pcl)
    return pcl


def npy_mask_to_ply(npy, mask, rgb_div=255, save_path = '', save_path_m = '', path=None, path_m=None):
    if npy is None:
        npy = np.load(path)

    ply = npy_to_ply(None, rgb_div, save_path, npy)

    if mask is None:
        mask = np.load(path_m)
    
    for i,m in enumerate(mask):
        npy[i, 3] = m * rgb_div
        npy[i, 4] = m * rgb_div
        npy[i, 5] = m * rgb_div
    
    ply_m = npy_to_ply(None, rgb_div, save_path_m, npy)

    return ply, ply_m


def get_run_id():
    return str(datetime.now()).replace(' ', '--')


def write_to_file(f, message, close=False):
    f.write(message)
    if close:
        f.close()
        return
    f.flush()


def set_mode(mode, models):
    assert mode in ['train', 'eval']
    for m in models:
        if m is None: continue
        if mode == 'train': m.train()
        if mode == 'eval': m.eval()


def transform_point(pt, tmat, point=True):
    ctemp = np.array(pt)
    if point:
        filler = 1.
    else:
        filler = 0.
    _ct_chg = np.array([ctemp[0], ctemp[1], ctemp[2], filler])
    _ct_align = np.matmul(tmat, _ct_chg)[:3]
    return _ct_align


def get_action_key_from_dt(c_dt):
        return f"{c_dt['action']}_{c_dt['id']}"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
