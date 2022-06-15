import trimesh
import open3d as o3d
from PIL import Image
import pandas
import numpy as np
import os
from misc_utils import get_data_path


def get_ccp(obj, cloud_vertices):
    scene = o3d.t.geometry.RaycastingScene()
    primitive_obj = o3d.t.geometry.TriangleMesh.from_legacy(obj)
    scene.add_triangles(primitive_obj)
    return scene.compute_closest_points(o3d.core.Tensor(cloud_vertices, dtype=o3d.core.Dtype.Float32))


def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def get_vert_helper(idx, obj_tris, obj_verts, ccp_tri_ids, ccp_pt):        
    nodes = np.empty((3,3))
    tris = obj_tris[ccp_tri_ids[idx]]
    nodes[0] = obj_verts[tris[0]]
    nodes[1] = obj_verts[tris[1]]
    nodes[2] = obj_verts[tris[2]]
    return closest_node(ccp_pt, nodes)


def get_vert_uv_index(idx, obj_tris, obj_verts, ccp_tri_ids, ccp_pt):
    """
    vert_index = 0
    for pt in obj_tris[ccp_tri_ids[idx]]:
        if np.allclose(ccp_pt, obj_verts[pt]):
            vert_index += ccp_tri_ids[idx] * 3
            return vert_index
        vert_index += 1
    return -1
    """
    closest_index = get_vert_helper(idx, obj_tris, obj_verts, ccp_tri_ids, ccp_pt)
    return closest_index + ccp_tri_ids[idx]*3


def get_vcolors(obj, ccp, tex):
    obj_tris = np.asarray(obj.triangles)
    obj_verts = np.asarray(obj.vertices)
    uv_indices = np.asarray(obj.triangle_uvs)
    ccp_tri_ids = ccp['primitive_ids'].numpy()
    ccp_pts = ccp['points'].numpy()
    
    uvs, to_remove = [], []
    for i, ccp_pt in enumerate(ccp_pts):
        vert_index = get_vert_uv_index(i, obj_tris, obj_verts, ccp_tri_ids, ccp_pt)
        if vert_index == -1:
            to_remove.append(i)
        else:
            uvs.append(uv_indices[vert_index])
    uvs_np = np.array(uvs)
    return trimesh.visual.color.uv_to_color(uvs_np, tex)[:,:3], to_remove


def get_vcolors2(obj, ccp):
    obj_tris = np.asarray(obj.triangles)
    obj_verts = np.asarray(obj.vertices)
    ccp_tri_ids = ccp['primitive_ids'].numpy()
    ccp_pts = ccp['points'].numpy()
    
    pt_indices, _ = [], []
    for i, ccp_pt in enumerate(ccp_pts):
        vert_index = get_vert_index2(i, obj_tris, obj_verts, ccp_tri_ids, ccp_pt)
        pt_indices.append(vert_index)
    return pt_indices


def get_vert_index2(idx, obj_tris, obj_verts, ccp_tri_ids, ccp_pt):
    tris = obj_tris[ccp_tri_ids[idx]]
    return tris[get_vert_helper(idx, obj_tris, obj_verts, ccp_tri_ids, ccp_pt)]


def sample_new_pointcloud(pointcloud, csv, sample_size=100000):
    new_pc = pointcloud.sample_points_uniformly(sample_size)
    ccpt = get_ccp(pointcloud, np.asarray(new_pc.points))
    points_indices = get_vcolors2(pointcloud, ccpt)
    old_pc_colors = np.asarray(pointcloud.vertex_colors)
    new_pc_colors = np.array([old_pc_colors[x] for x in points_indices])
    new_pc.colors = o3d.utility.Vector3dVector(new_pc_colors)

    old_pc_global_ids = csv['globalId'].to_numpy()
    new_pc_global_ids = np.array([old_pc_global_ids[x] for x in points_indices])

    old_pc_object_ids = csv['objectId'].to_numpy()
    new_pc_object_ids = np.array([old_pc_object_ids[x] for x in points_indices])
    return new_pc, new_pc_global_ids, new_pc_object_ids


def create_and_save_input(vertices, to_delete, global_ids, object_ids, vcolors, save_path):
    vertices = np.delete(vertices, to_delete, axis=0)
    global_ids = np.delete(global_ids, to_delete)
    object_ids = np.delete(object_ids, to_delete)
    final = np.hstack((vertices, vcolors, global_ids.reshape(-1,1), object_ids.reshape(-1,1)))
    np.save(save_path, final)
    return final


def create_net_inputs_helper(cloud, csv, obj, tex, ssize, save_dir, file_name):
    sampled_clouds, sampled_ids, sampled_oids = sample_new_pointcloud(cloud, csv, ssize)
    vertices = np.asarray(sampled_clouds.points)
    ccp = get_ccp(obj, vertices)
    vcolors, to_delete = get_vcolors(obj, ccp, tex)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    create_and_save_input(vertices, to_delete, sampled_ids, sampled_oids, vcolors, save_path)


def create_net_inputs(ref, chg, located_ply, located_obj, sample_size):
    ref_cloud, chg_cloud, ref_csv, chg_csv = located_ply
    ref_obj, chg_obj, ref_tex, chg_tex = located_obj
    base_save_path = os.path.join(get_data_path(chg), f'net_inputs_sampled_{sample_size}/')

    create_net_inputs_helper(
        cloud=ref_cloud,
        csv=ref_csv,
        obj=ref_obj,
        tex=ref_tex,
        ssize=sample_size,
        save_dir=base_save_path,
        file_name='ref_pointcloud.npy'
    )

    create_net_inputs_helper(
        cloud=chg_cloud,
        csv=chg_csv,
        obj=chg_obj,
        tex=chg_tex,
        ssize=sample_size,
        save_dir=base_save_path,
        file_name='chg_pointcloud.npy'
    )


def create_pointcloud_from_net_input(net_input, save_path_ref, save_path_chg):
    x = net_input
    if type(net_input) != type(np.empty((1,1))):
        x = net_input.numpy()

    ref_pcl = o3d.geometry.PointCloud()
    ref_pcl.points = o3d.utility.Vector3dVector(np.transpose(x[0,:3,:], (1,0)))
    ref_pcl.colors = o3d.utility.Vector3dVector(np.transpose(x[0,3:6,:]/255, (1,0)))
    o3d.io.write_point_cloud(save_path_ref, ref_pcl, write_ascii=True)

    chg_pcl = o3d.geometry.PointCloud()
    chg_pcl.points = o3d.utility.Vector3dVector(np.transpose(x[1,:3,:], (1,0)))
    chg_pcl.colors = o3d.utility.Vector3dVector(np.transpose(x[1,3:6,:]/255, (1,0)))
    o3d.io.write_point_cloud(save_path_chg, chg_pcl, write_ascii=True)
