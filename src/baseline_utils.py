import os
import shutil
import numpy as np
import cv2
from localization_utils import process_transform
from misc_utils import get_action_key_from_dt


def create_baseline_data_helper(sd, seq_path_a, seq_path_b, new_seq_path_b, c_dt, scan, n_scan, cap_path, n_action, chg_scan, ref, i):
    img_path = get_img_path(seq_path_a, sd['frame'])

    base_save_path = os.path.join(
        cap_path, c_dt['dir'], get_action_key_from_dt(c_dt), 'baseline', scan
    )
    save_to_txt(base_save_path, f'img_path_{i}.txt', img_path)
    save_to_txt(base_save_path, f'bb_{i}.txt', sd['bb'])

    if c_dt['action'] == n_action:
        transform = process_transform(chg_scan['transform'])
        new_pose = transform_pose(sd['pose'], transform, not ref)
        frame_name, frame_path = get_last_frame(new_seq_path_b, seq_path_b)
        output_pose(frame_path, new_pose)

        base_save_path = os.path.join(
            cap_path, c_dt['dir'], get_action_key_from_dt(c_dt), 'baseline', n_scan
        )
        img_path = get_img_path(new_seq_path_b, frame_name)
        save_to_txt(base_save_path, f'img_path_{i}.txt', img_path)
        save_to_txt(base_save_path, f'bb_{i}.txt', sd['bb'])


def create_baseline_data(ref, c_dt, score_dict, seq_path_a, seq_path_b, new_seq_path_b, cap_path, chg_scan, num=6):
    action = 'removed'
    n_action = 'added'
    scan = 'chg'
    n_scan = 'ref'
    if ref:
        action = 'added'
        n_action = 'removed'
        scan = 'ref'
        n_scan = 'chg'
    
    if c_dt['action'] != action:
        if c_dt['id'] not in score_dict:
            return False
        sds = sorted(score_dict[c_dt['id']], key=lambda x: x['score'], reverse=True)

        max_range = min(num, len(sds))
        for i in range(max_range):
            create_baseline_data_helper(
                sd=sds[i],
                seq_path_a=seq_path_a,
                seq_path_b=seq_path_b,
                new_seq_path_b=new_seq_path_b,
                c_dt=c_dt,
                scan=scan,
                n_scan=n_scan,
                cap_path=cap_path,
                n_action=n_action,
                chg_scan=chg_scan,
                ref=ref,
                i=i
            )
    return True


def save_to_txt(base_save_path, name, to_save):
    os.makedirs(base_save_path, exist_ok=True)
    path_f = os.path.join(
        base_save_path, name
    )
    with open(path_f, 'w') as f:
        f.write(to_save)


def get_img_path(sequence_path, frame):
    return os.path.abspath(
            os.path.join(
               sequence_path, f'{frame}.rendered.color.jpg'
        )
    )


def get_bb_img(image_path, bb_s, pixel_thickness=8, color=[0,255,0]):
    img = get_img_from_path(None, None, image_path)
    return add_bb_to_img(
        img=img,
        bbox=get_bb_from_str(bb_s),
        pixel_thickness=pixel_thickness,
        color=color
    )


def get_img_from_path(sequence_path, frame, image_path=None):
    if image_path is None:
        image_path = get_img_path(sequence_path, frame)

    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_bb_to_img(img, bbox, pixel_thickness=8, color=[0,255,0]):
    arr = img.copy()
    
    color = np.array(color, dtype=np.uint8)
    ss = [get_slice(b,arr.shape[i%2],pixel_thickness) for i,b in enumerate(bbox)]
    arr[ss[0][0]:ss[0][1], bbox[1]:bbox[3]] = color
    arr[bbox[0]:bbox[2], ss[1][0]:ss[1][1]] = color

    arr[ss[2][0]:ss[2][1], bbox[1]:bbox[3]] = color
    arr[bbox[0]:bbox[2], ss[3][0]:ss[3][1]] = color
    return arr


def get_slice(idx, mx, t):
    h_t = t // 2
    if ((idx - h_t) >= 0) and ((idx + h_t) <= mx):
        return (idx-h_t),(idx+h_t)
    elif (idx - h_t) < 0:
        return 0,t
    return (mx-t),mx


def get_bb_from_str(s, w=540):
    a,b,c,d = s.split(',')
    a,b,c,d = int(a),int(b),int(c),int(d)
    box = b
    b = w - d
    d = w - box
    return a,b,c,d


def get_relevant_line(line):
    if line.endswith('\n'):
        return line[:-1]
    return line


def read_bb(b):
    bb_dict = {}
    with open(b) as f:
        lines = f.readlines()
        for line in lines:
            line = get_relevant_line(line).split(' ')
            bb_dict[int(line[0])] = f'{line[1]},{line[2]},{line[3]},{line[4]}'
    return bb_dict


def get_frame_from_visibility_path(v_path):
    return (v_path.split('/')[-1]).split('.')[0]


def get_id_truncation_occlusion(line):
    line = get_relevant_line(line).split(' ')
    return int(line[0]), float(line[3]), float(line[6])


def read_visibility(f, v_path, score_dict, bb_dict, pose):
    lines = f.readlines()
    for line in lines:
        idx, t, o = get_id_truncation_occlusion(line)
        score = t + o
        if idx in bb_dict:
            score_entry = {
                'score':score,
                'frame':get_frame_from_visibility_path(v_path),
                'bb':bb_dict[idx],
                'pose':pose
            }
            if idx not in score_dict:
                score_dict[idx] = [score_entry]
            else:
                score_dict[idx].append(score_entry)


def get_visibility_bb_from_seq_path(sequence_path):
    visibilities = [os.path.join(sequence_path, x) for x in os.listdir(sequence_path) if 'visibility.txt' in x]
    bboxes = [os.path.join(sequence_path, x) for x in os.listdir(sequence_path) if 'bb.txt' in x]
    poses = [os.path.join(sequence_path, x) for x in os.listdir(sequence_path) if 'pose.txt' in x]
    return sorted(visibilities), sorted(bboxes), sorted(poses)


def get_score_dict(visibilities, bboxes, poses):
    score_dict = {}
    for v,b,p in zip(visibilities, bboxes, poses):
        with open(v) as f_v:
            read_visibility(f_v, v, score_dict, read_bb(b), read_mat_from_pose(p))
    return score_dict


def transform_pose(pose, transform, to_ref):
    if to_ref:
        return np.matmul(transform, pose)
    return np.matmul(np.linalg.inv(transform), pose)


def read_mat_from_pose(pose_p):
    mat = np.empty((4,4))
    with open(pose_p) as f:
        for i,line in enumerate(f.readlines()):
            for k,n in enumerate(get_relevant_line(line).split(' ')):
                mat[i,k] = float(n)
    return mat


def output_pose(path, mat):
    with open(path, 'w') as f:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                f.write(f'{mat[i,j]} ')
            f.write(f'\n')


def get_last_frame(new_seq_path, old_seq_path):
    if not os.path.isdir(new_seq_path):
        os.makedirs(new_seq_path, exist_ok=True)
    files = os.listdir(new_seq_path)
    info_file = '_info.txt'

    if info_file not in files:
        shutil.copyfile(
            os.path.join(old_seq_path, info_file), os.path.join(new_seq_path, info_file)
        )
    
    new_frame_s = get_last_frame_helper(files)
    frame_name = f'frame-{new_frame_s}'
    frame_path = os.path.join(
        new_seq_path, f'{frame_name}.pose.txt'
    )
    return frame_name, frame_path


def get_last_frame_helper(files):
    frames = list(set([int((f.split('.')[0]).split('-')[1]) for f in files if 'frame' in f]))
    if len(frames) == 0:
        new_frame_num = 0
    else:
        new_frame_num = sorted(frames, reverse=True)[0]+1
    new_frame_s = f"{new_frame_num:06d}"
    return new_frame_s


def get_baselines_meta_from_c_dt(c_dt, cap_path, i):
    meta1 = ['ref', 'chg']
    baseline_meta = {m1:{} for m1 in meta1}
    img_path = 'img_path'
    for m1 in meta1:
        base_path = os.path.join(cap_path, c_dt['dir'], get_action_key_from_dt(c_dt), 'baseline', m1)

        with open(
                os.path.join(base_path, f'bb_{i}.txt')
            ) as f:
                baseline_meta[m1]['bb'] = f.readline()
        
        img_path_p = os.path.join(base_path, f'{img_path}_{i}.txt')
        if os.path.exists(img_path_p):
            with open(img_path_p) as f:
                baseline_meta[m1][img_path] = f.readline()
        else:
            img_path_p = os.path.join(base_path, f'img_{i}.jpg')
            assert os.path.exists(img_path_p), f"The baseline picture {i} is missing at {base_path}"
            baseline_meta[m1][img_path] = os.path.abspath(img_path_p)
    return baseline_meta


def get_bb_image_pair_from_c_dt(c_dt, cap_path, i):
    baseline_meta = get_baselines_meta_from_c_dt(c_dt, cap_path, i)
    ref_bb_img = get_bb_img(
        image_path=baseline_meta['ref']['img_path'],
        bb_s=baseline_meta['ref']['bb']
    )
    chg_bb_img = get_bb_img(
        image_path=baseline_meta['chg']['img_path'],
        bb_s=baseline_meta['chg']['bb']
    )
    return ref_bb_img, chg_bb_img


def get_resized_bb_img(img, a, b, c, d, meta, w, h, prefix):
    bb = get_bb_from_str(meta[prefix]['bb'])
    cropped_img = img[bb[0]-a:bb[2]+c,bb[1]-b:bb[3]+d,:]
    resized_cropped_img = cv2.resize(cropped_img, (w,h))

    new_bb = a,b,bb[2]-bb[0]+a,bb[3]-bb[1]+b
    mm = new_coordinates_after_resize_img(
        cropped_img.shape[:2], (h,w), new_bb[:2]
    )
    mM = new_coordinates_after_resize_img(
        cropped_img.shape[:2], (h,w), new_bb[2:4]
    )
    resized_new_bb = (mm[0],mm[1],mM[0],mM[1])
    return add_bb_to_img(resized_cropped_img, resized_new_bb)


def new_coordinates_after_resize_img(original_size, new_size, original_coordinate):
    original_size = np.array(original_size)
    new_size = np.array(new_size)
    original_coordinate = np.array(original_coordinate)
    xy = original_coordinate/(original_size/new_size)
    x, y = int(xy[0]), int(xy[1])
    return (x, y)


def get_same_fov_bb_image_pair_from_c_dt(c_dt, cap_path, i):
    baseline_meta = get_baselines_meta_from_c_dt(c_dt, cap_path, i)
    ref_img = get_img_from_path(
        None, None, baseline_meta['ref']['img_path']
    )
    chg_img = get_img_from_path(
        None, None, baseline_meta['chg']['img_path']
    )

    h, w, _ = ref_img.shape
    a_ref, b_ref, c_ref, d_ref = get_bb_from_str(
        baseline_meta['ref']['bb']
    )
    a_chg, b_chg, c_chg, d_chg = get_bb_from_str(
        baseline_meta['chg']['bb']
    )

    c_ref, d_ref, c_chg, d_chg = \
        h - c_ref, w - d_ref, h - c_chg, w - d_chg
    a, b, c, d = \
        min(a_ref,a_chg), min(b_ref,b_chg), min(c_ref,c_chg), min(d_ref,d_chg)

    ref_bb_img = get_resized_bb_img(
        img=ref_img,
        a=a, b=b, c=c, d=d,
        meta=baseline_meta,
        w=w, h=h,
        prefix='ref'
    )

    chg_bb_img = get_resized_bb_img(
        img=chg_img,
        a=a, b=b, c=c, d=d,
        meta=baseline_meta,
        w=w, h=h,
        prefix='chg'
    )
    
    return ref_bb_img, chg_bb_img
