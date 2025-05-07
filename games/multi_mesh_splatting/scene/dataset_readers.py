#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import os
import numpy as np
import trimesh
import torch

from games.multi_mesh_splatting.utils.graphics_utils import MultiMeshPointCloud
from games.mesh_splatting.utils.graphics_utils import MeshPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    getNerfppNorm,
    SceneInfo,
    storePly
)
from utils.sh_utils import SH2RGB

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    read_extrinsics_binary,
    read_intrinsics_binary
)

from scene.dataset_readers import readColmapCameras

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def readColmapMeshSceneInfo(path, images, eval, num_splats, meshes, llffhold=8, masks_dir=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    masks_dir = "masks" if masks_dir == None else masks_dir
    obj_masks_dir = "mask_obj"
    human_masks_dir = "mask_human"

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir),
                                           masks_folder=os.path.join(path, masks_dir),
                                           obj_masks_folder=os.path.join(path, obj_masks_dir),
                                           human_masks_folder=os.path.join(path, human_masks_dir)
                                           )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcds = []
    ply_paths = []
    total_pts = 0
    for i, (mesh, num) in enumerate(zip(meshes, [num_splats])):
        ply_path = os.path.join(path, f"points3d_{i}.ply")

        mesh_scene = trimesh.load(f'{path}/{mesh}.obj', force='mesh')
        vertices = mesh_scene.vertices
        faces = mesh_scene.faces
        triangles = torch.tensor(mesh_scene.triangles).float()  # equal vertices[faces]

        num_pts_each_triangle = num
        num_pts = num_pts_each_triangle * triangles.shape[0]
        total_pts += num_pts

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )

        # alpha = alpha / alpha.sum(dim=-1, keepdim=True)

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = MeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            # transform_vertices_function=transform_vertices_function,
            transform_vertices_function=None,
            triangles=triangles.cuda()
        )
        pcds.append(pcd)
        ply_paths.append(ply_path)
        storePly(ply_path, pcd.points, SH2RGB(shs) * 255)
    
    print(
        f"Generating random point cloud ({total_pts})..."
    )

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Colmap_Mesh": readColmapMeshSceneInfo
}
