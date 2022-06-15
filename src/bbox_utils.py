from misc_utils import get_data_path
from misc_utils import transform_point
import json
import os
from plyfile import PlyData, PlyElement
import numpy as np


def get_create_ss(ref, path=None):
    return create_seg_dict(get_sem_seg(ref,path))


def get_sem_seg(ref, path=None):
    p = get_data_path(ref, path)
    semseg_path = os.path.join(p, 'semseg.v2.json')
    f = open(semseg_path)
    semseg = json.load(f)
    f.close()
    return semseg


def create_seg_dict(sem_seg):
    d = {}
    for o in sorted(sem_seg['segGroups'], key=lambda x: x['id']):
        _id = o['id']
        _label = o['label']
        d[f'{_id}_{_label}'] = ((_id, _label, o['obb']))
    return d


def interpolate_and_zip(a, b, acc):
    tmp_x = np.interp([i for i in range(1,acc+1)], [1,acc], [a[0], b[0]])
    tmp_y = np.interp([i for i in range(1,acc+1)], [1,acc], [a[1], b[1]])
    tmp_z = np.interp([i for i in range(1,acc+1)], [1,acc], [a[2], b[2]])
    return zip(tmp_x, tmp_y, tmp_z)


def get_bb_arr(ct, half_alx, nax, accuracy=100):
    megarray = []

    for w in range(2):
        if w == 0:
            ww = [1,2]
        elif w == 1:
            ww = [0,2]

        bound_1 = ct + half_alx[w]*nax[:,w]
        bound_2 = ct - half_alx[w]*nax[:,w]

        for i,j,k in interpolate_and_zip(bound_1, bound_2, accuracy):
            tmp_v = [i,j,k]

            for index in ww:
                index_r = index
                if index == ww[0]:
                    index_r = ww[1]
                else:
                    index_r = ww[0]

                for flag in range(2):
                    tmp_vd = tmp_v
                    if flag == 0:
                        tmp_vd = tmp_v + half_alx[index]*nax[:,index]
                    else:
                        tmp_vd = tmp_v - half_alx[index]*nax[:,index]

                    bound_11 = tmp_vd + half_alx[index_r]*nax[:,index_r]
                    bound_22 = tmp_vd - half_alx[index_r]*nax[:,index_r]

                    for i1, j1, k1 in interpolate_and_zip(bound_11, bound_22, accuracy):
                        ppt = [i1, j1, k1]
                        megarray.append((ppt[0], ppt[1], ppt[2], 255, 255, 255))
    return megarray


def get_normalized_axes_mat(na_ax, transform=None):
    nax = np.empty((3,3))
    for i in range(3):
        na_ax_vec = [na_ax[3*i], na_ax[3*i+1], na_ax[3*i+2]]
        if transform is not None:
            na_ax_vec = transform_point(na_ax_vec, transform, point=False)
        nax[0,i] = na_ax_vec[0]
        nax[1,i] = na_ax_vec[1]
        nax[2,i] = na_ax_vec[2]
    return nax


def create_bb_ply(name, bb_arr):
    megarray_np = np.array(bb_arr, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    megarray_el = PlyElement.describe(megarray_np, 'vertex')
    PlyData([megarray_el], text=True).write(f'{name}.ply')


def create_bb(name, segd_tuple, transform=None):
    centroid, normal_axes, half_axes_length = get_bb_info(segd_tuple, transform)
    
    megarray = get_bb_arr(centroid, half_axes_length, normal_axes)
    if name != None:
        create_bb_ply(name, megarray)


def get_bb_info(segd_tuple, transform=None):
    centroid = segd_tuple['centroid']
    if transform is not None:
        centroid = transform_point(centroid, transform)
    normal_axes = get_normalized_axes_mat(segd_tuple['normalizedAxes'], transform)
    half_axes_length = [i/2. for i in segd_tuple['axesLengths']]

    return centroid, normal_axes, half_axes_length


def get_bb_bounds(segd_tuple, transform=None, enhance=0.):
    ct, nax, half_alx = get_bb_info(segd_tuple, transform)
    half_alx = np.array(half_alx) * (1.+enhance)

    corners = np.empty((8,3))

    corners[0,:] = ct + half_alx[0]*nax[:,0] + half_alx[1]*nax[:,1] + half_alx[2]*nax[:,2]
    corners[1,:] = ct + half_alx[0]*nax[:,0] + half_alx[1]*nax[:,1] - half_alx[2]*nax[:,2]
    corners[2,:] = ct + half_alx[0]*nax[:,0] - half_alx[1]*nax[:,1] + half_alx[2]*nax[:,2]
    corners[3,:] = ct + half_alx[0]*nax[:,0] - half_alx[1]*nax[:,1] - half_alx[2]*nax[:,2]
    corners[4,:] = ct - half_alx[0]*nax[:,0] + half_alx[1]*nax[:,1] + half_alx[2]*nax[:,2]
    corners[5,:] = ct - half_alx[0]*nax[:,0] + half_alx[1]*nax[:,1] - half_alx[2]*nax[:,2]
    corners[6,:] = ct - half_alx[0]*nax[:,0] - half_alx[1]*nax[:,1] + half_alx[2]*nax[:,2]
    corners[7,:] = ct - half_alx[0]*nax[:,0] - half_alx[1]*nax[:,1] - half_alx[2]*nax[:,2]

    return corners


def get_scan_bboxes(scan, lab_dict, to_use=None, transform=None, tup_num=10):
    segd_tuples = create_seg_dict(get_sem_seg(scan))

    tuples_to_use = [k for k in segd_tuples.keys() if k.split('_')[1] in to_use]
    if len(tuples_to_use) < tup_num:
        return None
    random_t_indices = np.random.choice(len(tuples_to_use), tup_num, replace=False)
    bboxes = np.empty((tup_num, 8, 3))
    labs = np.empty((tup_num,), dtype=np.int)
    for i, index in enumerate(random_t_indices):
        segd_tuple = segd_tuples[tuples_to_use[index]]
        bboxes[i] = get_bb_bounds(segd_tuple[2], transform)
        labs[i] = int(lab_dict[str(segd_tuple[0])])
        
    return bboxes, labs
