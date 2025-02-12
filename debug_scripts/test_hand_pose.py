import os
import numpy as np
import torch
import cv2
import trimesh
import torchvision

from vine_prune.utils import read_json, read_np_data
from vine_prune.utils.mano import get_mano, get_faces

import sys
sys.path.append('/home/hfreeman/harry_ws/repos/gaussian-mesh-splatting')
# from scene import GaussianModel
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
from argparse import ArgumentParser
from arguments import PipelineParams
from renderer.gaussian_renderer import my_render

def run(splat_path, splat_params_path, splat_json_path,
        debug_dir, mano_pose_path,
        cam_ind=0,
        mano_pose_ind=0):
    gaussians = GaussianMeshModel(3)
    gaussians.load_ply(splat_path)
    
    splat_params = torch.load(splat_params_path)

    splat_data = read_json(splat_json_path)
    splat_cam_data = splat_data[cam_ind]
    R_inv = np.array(splat_cam_data['rotation'])
    t_inv = np.array(splat_cam_data['position'])

    M_inv = np.eye(4)
    M_inv[0:3, 0:3] = R_inv
    M_inv[0:3, 3] = t_inv
    M = np.linalg.inv(M_inv)
    R = M[0:3, 0:3]
    t = M[0:3, 3]

    fx = splat_cam_data['fx']
    fy = splat_cam_data['fy']
    cx = splat_cam_data['cx']
    cy = splat_cam_data['cy']
    width = splat_cam_data['width']
    height = splat_cam_data['height']

    R = torch.FloatTensor(R).cuda()
    t = torch.FloatTensor(t).cuda()
    intrinsics = [fx, fy, cx, cy]
    dims =[height, width]

    parser = ArgumentParser()
    pipeline_par = PipelineParams(parser)
    args, _ = parser.parse_known_args()
    pipeline = pipeline_par.extract(args)
    pipeline.convert_SHs_python = True

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    res_pgk = my_render(gaussians, pipeline, background, intrinsics, dims, R.T, t)
    image = (res_pgk['render'].cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #mask = (res_pgk['alpha'][0].cpu().numpy()*255).round().astype(np.uint8)

    im_path = os.path.join(debug_dir, 'image.png')
    #mask_path = os.path.join(debug_dir, 'mask.png')

    cv2.imwrite(im_path, image)
    #cv2.imwrite(mask_path, mask)

    torchvision.utils.save_image(res_pgk['render'], im_path.replace('.png', '_tu.png'))
    
    orig_triangles = splat_params['triangles'].cpu().numpy()

    num_pts_per_face = splat_params['_alpha'].shape[1]
    orig_gauss_pts = gaussians._xyz.detach().cpu().numpy().reshape(-1, num_pts_per_face, 3)
    
    triangles_to_compute = np.repeat(orig_triangles[:, None], 2, axis=1).reshape(-1, 3, 3)
    points_to_compute = orig_gauss_pts.reshape(-1, 3)
    barycentric_coords = trimesh.triangles.points_to_barycentric(triangles_to_compute, points_to_compute)

    mano = get_mano().cuda()
    mano_faces = get_faces()

    mano_pose_data = read_np_data(mano_pose_path)
    betas = torch.FloatTensor(mano_pose_data['right']['betas'][mano_pose_ind:mano_pose_ind+1]).cuda()
    hand_pose = torch.FloatTensor(mano_pose_data['right']['hand_pose'][mano_pose_ind:mano_pose_ind+1]).cuda()
    # betas = torch.zeros(1, 10).float().cuda()
    # hand_pose = torch.zeros(1, 45).float().cuda()

    mano_out = mano(hand_pose=hand_pose,
                    betas=betas)
    
    new_vertices = mano_out.vertices.squeeze(0).cpu().numpy()
    new_triangles = new_vertices[mano_faces]
    new_triangles_to_compute = np.repeat(new_triangles[:, None], 2, axis=1).reshape(-1, 3, 3)
    new_points = trimesh.triangles.barycentric_to_points(new_triangles_to_compute, barycentric_coords)

    gaussians._xyz[:, :] = torch.FloatTensor(new_points).cuda()

    res_pgk = my_render(gaussians, pipeline, background, intrinsics, dims, R.T, t)
    image = (res_pgk['render'].cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #mask = (res_pgk['alpha'][0].cpu().numpy()*255).round().astype(np.uint8)

    im_path = os.path.join(debug_dir, 'image_pose.png')
    #mask_path = os.path.join(debug_dir, 'mask_pose.png')

    cv2.imwrite(im_path, image)
    #cv2.imwrite(mask_path, mask)

    torchvision.utils.save_image(res_pgk['render'], im_path.replace('.png', '_tu.png'))

SPLAT_PATH = '/home/hfreeman/harry_ws/repos/gaussian-mesh-splatting/output/mano_colmap/point_cloud/iteration_7000/point_cloud.ply'
SPLAT_PARAMS_PATH = '/home/hfreeman/harry_ws/repos/gaussian-mesh-splatting/output/mano_colmap/point_cloud/iteration_7000/model_params.pt'
SPLAT_JSON_PATH = '/home/hfreeman/harry_ws/repos/gaussian-mesh-splatting/output/mano_colmap/cameras.json'
DEBUG_DIR = '/home/hfreeman/harry_ws/gopro/debug_vis/vis_gauss_hand'
MANO_POSE_PATH = '/home/hfreeman/harry_ws/gopro/datasets/simple_manip/1_prune_interact/hand_obj/bak_with_jts/hold_fit_opt.npy'

if __name__ == "__main__":
    with torch.no_grad():
        run(SPLAT_PATH, SPLAT_PARAMS_PATH, SPLAT_JSON_PATH,
            DEBUG_DIR, MANO_POSE_PATH)
