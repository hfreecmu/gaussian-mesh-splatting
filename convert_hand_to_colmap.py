import argparse
import os
import shutil
import numpy as np
import pycolmap
import pytorch3d.transforms
import torch
import trimesh
from scipy.spatial.transform import Rotation

from scene.dataset_readers import storePly
from vine_prune.utils.io import read_np_data
from vine_prune.utils.paths import get_base_data_dir

def symlink(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)

def read_np_data(path):
    return np.load(path, allow_pickle=True).item()

def run(data_dir, scene_dir, output_dir, is_refine):
    image_dir = os.path.join(data_dir, 'undistorted')
    masks_dir = os.path.join(data_dir, 'mask_hand')
    obj_masks_dir = os.path.join(data_dir, 'mask_obj')
    human_masks_dir = os.path.join(data_dir, 'mask_human')

    col_image_dir = os.path.join(output_dir, 'images')
    col_masks_dir = os.path.join(output_dir, 'masks')
    col_obj_masks_dir = os.path.join(output_dir, 'mask_obj')
    col_human_masks_dir = os.path.join(output_dir, 'mask_human')

    symlink(os.path.abspath(image_dir), os.path.abspath(col_image_dir))
    symlink(os.path.abspath(masks_dir), os.path.abspath(col_masks_dir))
    symlink(os.path.abspath(obj_masks_dir), os.path.abspath(col_obj_masks_dir))
    symlink(os.path.abspath(human_masks_dir), os.path.abspath(col_human_masks_dir))

    sparse_dir = os.path.join(output_dir, 'sparse')
    if not os.path.exists(sparse_dir):
        os.mkdir(sparse_dir)

    sparse_dir = os.path.join(sparse_dir, '0')
    if not os.path.exists(sparse_dir):
        os.mkdir(sparse_dir)

    mesh_path = os.path.join(data_dir, 'meshes', 'hand_mesh.obj')
    col_mesh_path = os.path.join(output_dir, 'mesh.obj')
    shutil.copyfile(mesh_path, col_mesh_path)

    cam_K_path = os.path.join(data_dir, 'cam_K.txt')
    cam_K = np.loadtxt(cam_K_path)

    cam_dims_path = os.path.join(data_dir, 'cam_dims.txt')
    cam_dims = np.loadtxt(cam_dims_path).astype(int)

    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]
    height, width = cam_dims

    camera = pycolmap.Camera(
                model='PINHOLE',
                width=width,
                height=height,
                params=[fx, fy, cx, cy],
                camera_id=1,
            )

    filenames = []
    for filename in os.listdir(image_dir):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue

        filenames.append(filename)

    filenames = sorted(filenames)
    pc_images = []
    for image_id, image_name in enumerate(filenames):
        M = np.eye(4)
        M = M[0:3]

        pc_image = pycolmap.Image(
            image_id=image_id + 1,
            camera_id=camera.camera_id,
            name=image_name,
            cam_from_world=M
        )
        pc_images.append(pc_image)

    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera(camera)

    for image in pc_images:
        reconstruction.add_image(image)

    reconstruction.write(sparse_dir) 

    mesh = trimesh.load(mesh_path, process=False)
    ply_path = os.path.join(sparse_dir, 'points3D.ply')
    xyz = np.array(mesh.vertices)
    try:
        rgb = np.array(mesh.visual.to_color().vertex_colors)[:, 0:3]
    except:
        rgb = np.array(mesh.visual.vertex_colors)[:, 0:3]

    storePly(ply_path, xyz, rgb)

    # scene_splat_path = os.path.join(scene_dir, 'splat', 'scale.ply')
    # col_scene_splat_path = os.path.join(output_dir, 'scene.ply')
    # shutil.copyfile(scene_splat_path, col_scene_splat_path)

    obj_splat_path = os.path.join(data_dir, 'meshes', 'obj_splat.ply')
    col_obj_splat_path = os.path.join(output_dir, 'obj_splat.ply')
    shutil.copyfile(obj_splat_path, col_obj_splat_path)

    if not is_refine:
        # ho_path = os.path.join(data_dir, 'hand_obj', 'hold_init_ho_fit.npy')
        ho_path = os.path.join(data_dir, 'hand_obj', 'hold_init_ho.npy')
    else:
        # ho_path = os.path.join(data_dir, 'hand_obj', 'hold_refine_ho_fine.npy')
        ho_path = os.path.join(data_dir, 'hand_obj', 'hold_refine_ho_demi.npy')

    ho_data = read_np_data(ho_path)
    if not 'scale' in ho_data['right']:
        ho_data['right']['scale'] = np.array(1.0)
    
    # scene_go = np.zeros_like(ho_data['right']['global_orient']) + np.nan
    # scene_tr = np.zeros_like(ho_data['right']['transl']) + np.nan

    # scene_pose_dir = os.path.join(data_dir, 'scene_registration', 'cam_poses')
    # for filename in os.listdir(scene_pose_dir):
    #     if not filename.endswith('.npy'):
    #         continue

    #     image_ind = int(filename.split('.npy')[0])
    #     scene_pose = np.loadtxt(os.path.join(scene_pose_dir, filename))
    #     scene_go[image_ind] = Rotation.from_matrix(scene_pose[0:3, 0:3]).as_rotvec()
    #     scene_tr[image_ind] = scene_pose[0:3, 3]

    # assert np.isnan(scene_go).sum() == 0
    # ho_data['scene'] = {
    #     'global_orient': scene_go,
    #     'transl': scene_tr
    # }

    col_ho_path = os.path.join(output_dir, 'hold_init_ho_scene.npy')
    np.save(col_ho_path, ho_data)

    sdf_path = os.path.join(data_dir, 'meshes', 'sdf.pkl')
    col_sdf_path = os.path.join(output_dir, 'sdf.pkl')
    shutil.copyfile(sdf_path, col_sdf_path)

    print('Done')

def main(args):
    model_name = args.model_name
    is_refine = args.is_refine
    is_dexycb = args.is_dexycb
    is_ho3d = args.is_ho3d

    base_data_dir = get_base_data_dir(is_dexycb, is_ho3d=is_ho3d)
    data_dir = os.path.join(base_data_dir, model_name)

    scene_dir=None

    if not is_refine:
        dirname = model_name
    else:
        dirname = f'{model_name}_REFINE'

    if is_dexycb:
        dexycb_dir = os.path.join('data', 'DEXYCB')
        if not os.path.exists(dexycb_dir):
            os.mkdir(dexycb_dir)
        output_dir = os.path.join(dexycb_dir, dirname)
    elif is_ho3d:
        ho3d_dir = os.path.join('data', 'HO3D')
        if not os.path.exists(ho3d_dir):
            os.mkdir(ho3d_dir)
        output_dir = os.path.join(ho3d_dir, dirname)
    else:
        output_dir = os.path.join('data', dirname)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run(data_dir, scene_dir, output_dir, is_refine)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument('--is_refine', action='store_true')
    parser.add_argument('--is_dexycb', action='store_true')
    parser.add_argument('--is_ho3d', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)