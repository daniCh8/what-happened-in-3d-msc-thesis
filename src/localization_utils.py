import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
from pyntcloud import PyntCloud
import os
from misc_utils import get_data_path
from tqdm import tqdm
from bbox_utils import get_bb_bounds, get_create_ss
from misc_utils import transform_point


def process_transform(transform_a):
    transform = np.empty((4,4))
    for i in range(4):
        transform[0,i] = transform_a[4*i]
        transform[1,i] = transform_a[4*i+1]
        transform[2,i] = transform_a[4*i+2]
        transform[3,i] = transform_a[4*i+3]
    return transform


def align_all_pts(pts, transform):
    aligned_pts = np.empty_like(pts)
    
    for i, pt in enumerate(pts):
        aligned_pts[i, :] = transform_point(pt, transform)
    
    return aligned_pts


def get_min_max_pts(array):
    minx, maxx, miny, maxy, minz, maxz = float('inf'), -float('inf'), float('inf'), -float('inf'), float('inf'), -float('inf')

    for pt in array:
        maxx = max(maxx, pt[0])
        minx = min(minx, pt[0])
        maxy = max(maxy, pt[1])
        miny = min(miny, pt[1])
        maxz = max(maxz, pt[2])
        minz = min(minz, pt[2])
    
    return minx, maxx, miny, maxy, minz, maxz


def intercept_min_max(pts1, pts2):
    x_m1, x_M1, y_m1, y_M1, z_m1, z_M1 = get_min_max_pts(pts1)
    x_m2, x_M2, y_m2, y_M2, z_m2, z_M2 = get_min_max_pts(pts2)
    x_m = max(x_m1, x_m2)
    x_M = min(x_M1, x_M2)
    y_m = max(y_m1, y_m2)
    y_M = min(y_M1, y_M2)
    z_m = max(z_m1, z_m2)
    z_M = min(z_M1, z_M2)
    return np.array([x_m, y_m, z_m]), np.array([x_M, y_M, z_M])


def pt_is_in_interception(pt, min_bounds, max_bounds):
    return np.alltrue(np.greater_equal(pt, min_bounds)) and \
            np.alltrue(np.less_equal(pt, max_bounds))


def pt_in_cvex_hull(hull, pnt):
    new_hull = ConvexHull(np.concatenate((hull.points, [pnt])))
    if np.array_equal(new_hull.vertices, hull.vertices): 
        return True
    return False


def triangle_is_in_interception(tri, keys):
    return tri[0] in keys \
            and tri[1] in keys \
            and tri[2] in keys


def get_drop_indices_all(cloud_xyz, mins, maxs, hull):
    to_drop = []
    
    for i, xyz in enumerate(cloud_xyz):
        if (not pt_is_in_interception(xyz, mins, maxs)) or (not pt_in_cvex_hull(hull, xyz)):
            to_drop.append(i)
            
    return to_drop


def get_drop_indices_hull(cloud_xyz, hull, in_hull=False):
    to_drop = []
    
    for i, xyz in enumerate(cloud_xyz):
        if pt_in_cvex_hull(hull, xyz) == in_hull:
            to_drop.append(i)
            
    return to_drop


def get_drop_indices_meshes(np_mesh, keys, monitor=False):
    to_drop = []
    keys = set(keys)
    
    if monitor:
        it = tqdm(enumerate(np_mesh))
    else:
        it = enumerate(np_mesh)
    
    for i, xyz in it:
        if not triangle_is_in_interception(xyz, keys):
            to_drop.append(i)
            
    return to_drop


def get_drop_indices_global_id(cloud, s):
    return [i for i,x in enumerate(cloud.points['globalId']) if x in s]


def get_obb_hull(cloud=None, align=False, transform=None, o3d=None):
    assert cloud != None or o3d != None, "cloud and o3d can't be both None!"
    if o3d == None:
        o3d = cloud.to_instance("open3d", mesh=True)
    obb = o3d.get_oriented_bounding_box()
    if align:
        hull = ConvexHull(align_all_pts(np.asarray(obb.get_box_points()), transform))
    else:
        hull = ConvexHull(np.asarray(obb.get_box_points()))
    
    return hull


def locate_mesh(cloud, pts, mins, maxs, hull, transform=None, g_id_set=None):
    if mins is not None and maxs is not None:
        pts_to_drop = get_drop_indices_all(pts, mins, maxs, hull)
    else:
        pts_to_drop = get_drop_indices_hull(pts, hull)
    if g_id_set is not None:
        pts_to_drop.extend(get_drop_indices_global_id(cloud, g_id_set))
        pts_to_drop = list(set(pts_to_drop))

    dropped_pts = cloud.points.copy().drop(pts_to_drop)
    if transform is not None:
        # transform points to ref space
        xyz_dummy = dropped_pts[['x', 'y', 'z']].to_numpy()
        aligned_xyz = align_all_pts(xyz_dummy, transform)
        dropped_pts['x'] = aligned_xyz[:,0]
        dropped_pts['y'] = aligned_xyz[:,1]
        dropped_pts['z'] = aligned_xyz[:,2]
    
    index_dict = dict(zip(dropped_pts.index, range(len(dropped_pts.index))))
    mesh_to_drop = get_drop_indices_meshes(cloud.mesh.to_numpy(), index_dict.keys())
    dropped_meshes = cloud.mesh.copy().drop(mesh_to_drop).applymap(lambda x: index_dict[x])
    
    return dropped_pts, dropped_meshes


def create_instances(pts, meshes):
    pynt_cloud = PyntCloud(pts.reset_index(drop=True), meshes.reset_index(drop=True))
    
    float_colors = pts.drop(pts.columns[6:], axis=1)
    float_colors['red']   = float_colors['red'].map(lambda x: x/255)
    float_colors['green'] = float_colors['green'].map(lambda x: x/255)
    float_colors['blue']  = float_colors['blue'].map(lambda x: x/255)
    o3d_cloud = PyntCloud(float_colors.reset_index(drop=True), 
                          meshes.reset_index(drop=True)).to_instance("open3d", mesh=True)
    return pynt_cloud, o3d_cloud


def create_and_save_located_ply(cloud, pts, mins, maxs, hull, save_path=None, name=None, transform=None, g_id_set=None):
    dropped_pts, dropped_meshes = locate_mesh(
        cloud=cloud,
        pts=pts,
        mins=mins,
        maxs=maxs,
        hull=hull,
        transform=transform,
        g_id_set=g_id_set
    )
    pynt_cloud, o3d_cloud = create_instances(dropped_pts, dropped_meshes)
    
    if save_path != None:
        os.makedirs(save_path, exist_ok=True)
        pynt_cloud.to_file(
            os.path.join(save_path, f'{name}_located_pyntcloud.ply')
        )
        o3d.io.write_triangle_mesh(
            os.path.join(save_path, f'{name}_located_ascii.ply'), o3d_cloud, write_ascii=True
        )
        pynt_cloud.points.to_csv(
            os.path.join(save_path, f'{name}_located_pyntcloud.csv')
        )
    
    return pynt_cloud, o3d_cloud


def get_ply_to_intercept_helper(scan, chg_scan=None):
    path = get_data_path(scan)
    cloud = PyntCloud.from_file(f'{path}labels.instances.annotated.v2.ply')
    pts = cloud.xyz

    if chg_scan is not None:
        transform = process_transform(chg_scan['transform'])
        pts_aligned = align_all_pts(pts, transform)
        return cloud, pts_aligned, transform

    return cloud, pts


def get_ply_to_intercept(ref, chg, chg_scan):
    ref_cloud, ref_pts = get_ply_to_intercept_helper(ref)
    chg_cloud, chg_pts_aligned, chg_transform = \
        get_ply_to_intercept_helper(chg, chg_scan)

    return ref_cloud, chg_cloud, ref_pts, chg_pts_aligned, chg_transform


def create_ply_intercept(ref, chg, chg_scan, g_id_set=None, output_path=None):
    ref_cloud, chg_cloud, ref_pts, chg_pts_aligned, \
        chg_transform = get_ply_to_intercept(ref, chg, chg_scan)

    mins, maxs = intercept_min_max(ref_pts, chg_pts_aligned)
    hull = get_obb_hull(chg_cloud, True, chg_transform)
    
    if output_path is None:
        output_path = f'{get_data_path(chg)}isolated_ply/'
    
    ref_pynt_cloud, ref_o3d_cloud = \
        create_and_save_located_ply(
            cloud=ref_cloud,
            pts=ref_pts,
            mins=mins,
            maxs=maxs,
            hull=hull,
            save_path=output_path,
            name='new_ref',
            g_id_set=g_id_set
        )
    
    chg_pynt_cloud, chg_o3d_cloud = \
        create_and_save_located_ply(
            cloud=chg_cloud,
            pts=chg_pts_aligned,
            mins=mins,
            maxs=maxs,
            hull=hull,
            save_path=output_path,
            name='new_chg',
            transform=chg_transform,
            g_id_set=g_id_set
        )
    
    return ref_pynt_cloud, ref_o3d_cloud, chg_pynt_cloud, chg_o3d_cloud


def create_single_ply_intercept(scan, save_dir, file_name, np_bb, chg_scan=None):
    transform = None
    if chg_scan is None:
        cloud, pts = get_ply_to_intercept_helper(scan)
    else:
        cloud, pts, transform = \
            get_ply_to_intercept_helper(scan, chg_scan)
    hull = ConvexHull(np_bb)
    
    pynt_cloud, o3d_cloud = \
        create_and_save_located_ply(
            cloud=cloud,
            pts=pts,
            mins=None,
            maxs=None,
            hull=hull,
            save_path=save_dir,
            name=file_name,
            transform=transform
    )
    
    return pynt_cloud, o3d_cloud


def create_obj_intercept_helper(pts, obj, save_dir, file_name, hull, mins=None, maxs=None, transform=None, more_hulls=None):
    os.makedirs(save_dir, exist_ok=True)
    if mins is not None and maxs is not None:
        drop_indices = get_drop_indices_all(pts, mins, maxs, hull)
    else:
        drop_indices = get_drop_indices_hull(pts, hull)
    
    if more_hulls is not None:
        for h in more_hulls:
            drop_indices.extend(get_drop_indices_hull(pts, h, True))
            drop_indices = list(set(drop_indices))

    obj.remove_vertices_by_index(drop_indices)
    if transform is not None:
        aligned_vertices = align_all_pts(np.asarray(obj.vertices), transform)
        obj.vertices = o3d.utility.Vector3dVector(aligned_vertices)
    o3d.io.write_triangle_mesh(
        filename=os.path.join(save_dir, file_name), 
        mesh=obj, 
        write_triangle_uvs=True
    )
    return obj


def get_obj_to_intercept_helper(scan, chg_scan=None):
    obj_p = os.path.join(get_data_path(scan), 'mesh.refined.v2.obj')
    obj = o3d.io.read_triangle_mesh(obj_p, True)
    pts = np.asarray(obj.vertices)

    if chg_scan is None:
        return obj, pts

    transform = process_transform(chg_scan['transform'])
    pts_aligned = align_all_pts(pts, transform)

    return obj, pts, pts_aligned, transform


def get_obj_to_intercept(ref, chg, chg_scan):
    ref_obj, ref_pts = \
        get_obj_to_intercept_helper(ref)

    chg_obj, _, chg_pts_aligned, chg_transform = \
        get_obj_to_intercept_helper(chg, chg_scan)

    return ref_obj, chg_obj, ref_pts, chg_pts_aligned, chg_transform


def create_single_object_intercept(scan, save_dir, file_name, np_bb, chg_scan=None):
    transform = None
    if chg_scan is None:
        obj, pts = get_obj_to_intercept_helper(scan)
    else:
        obj, _, pts, transform = \
            get_obj_to_intercept_helper(scan, chg_scan)
    hull = ConvexHull(np_bb)

    return create_obj_intercept_helper(
        pts=pts,
        obj=obj,
        save_dir=save_dir,
        file_name=file_name,
        hull=hull,
        transform=transform
    )


def get_bb_hulls(scan, labels, transform=None):
    hulls = []

    for _,v in get_create_ss(scan).items():
        _,lab,bb = v
        if lab in labels:
            hulls.append(
                ConvexHull(get_bb_bounds(bb, transform))
            )
    
    return hulls


def create_obj_intercept(ref, chg, chg_scan, labels=None, save_dir=None):
    ref_obj, chg_obj, ref_pts, \
        chg_pts_aligned, chg_transform = get_obj_to_intercept(ref, chg, chg_scan) 
    mins, maxs = intercept_min_max(ref_pts, chg_pts_aligned)
    hull = get_obb_hull(o3d=chg_obj, align=True, transform=chg_transform)

    ref_more_hulls = None
    chg_more_hulls = None
    if labels is not None:
        ref_more_hulls = get_bb_hulls(ref, labels)
        chg_more_hulls = get_bb_hulls(chg, labels, chg_transform)
    
    if save_dir is None:
        output_path = os.path.join(get_data_path(chg), 'isolated_obj/')
        output_path_ref = os.path.join(output_path, 'ref/')
        output_path_chg = os.path.join(output_path, 'chg/')
    else:
        output_path_ref = save_dir
        output_path_chg = save_dir

    ref_obj = create_obj_intercept_helper(
        save_dir=output_path_ref,
        file_name='ref_located_mesh.obj',
        hull=hull,
        mins=mins,
        maxs=maxs,
        pts=ref_pts,
        obj=ref_obj,
        more_hulls=ref_more_hulls
    )

    chg_obj = create_obj_intercept_helper(
        save_dir=output_path_chg,
        file_name='chg_located_mesh.obj',
        hull=hull,
        mins=mins,
        maxs=maxs,
        pts=chg_pts_aligned,
        obj=chg_obj,
        transform=chg_transform,
        more_hulls=chg_more_hulls
    )
    
    return ref_obj, chg_obj
