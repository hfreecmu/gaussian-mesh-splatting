import os
import numpy as np
import torch
import cv2
import trimesh
import torchvision
import pyrender

from vine_prune.utils import read_json, read_np_data
from vine_prune.utils.mano import get_mano, get_faces

import sys
sys.path.append('/home/hfreeman/harry_ws/repos/gaussian-mesh-splatting')
# from scene import GaussianModel
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
from argparse import ArgumentParser
from arguments import PipelineParams
from renderer.gaussian_renderer import my_render

def get_opengl_to_opencv_camera_trans() -> np.ndarray:
    """Returns a transformation from OpenGL to OpenCV camera frame.

    Returns:
        A 4x4 transformation matrix (flipping Y and Z axes).
    """

    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    return yz_flip

# # hardcoding this for now
# HAND_PATH = '/home/hfreeman/harry_ws/gopro/datasets/simple_manip/0_pruner_rotate/hand_obj/hold_fit_smooth_fb.npy'
# def read_np_data(path):
#     return np.load(path, allow_pickle=True).item()
# HAND_DATA = read_np_data(HAND_PATH)

def run(splat_path, splat_json_path):
    gaussians = GaussianMeshModel(3)
    gaussians.load_ply(splat_path)

    gaussians.update_alpha()
    gaussians.prepare_scaling_rot()

    ###
    mano = get_mano(flat_hand_mean=True,
                    use_pca=False)
    ###

    # global_orient_full = torch.from_numpy(HAND_DATA['right']['global_orient']).float().cuda()
    # transl_full = torch.from_numpy(HAND_DATA['right']['transl']).float().cuda()
    # hand_pose_full = torch.from_numpy(HAND_DATA['right']['hand_pose']).float().cuda()
    # betas_full = torch.from_numpy(HAND_DATA['right']['betas']).float().cuda()
    global_orient_full = gaussians.global_orient
    transl_full = gaussians.transl
    hand_pose_full = gaussians.hand_pose
    betas_full = gaussians.betas

    splat_data = read_json(splat_json_path)

    ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
    py_scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=ambient_light)

    renderer = None

    for entry in splat_data:
        image_ind = int(entry["img_name"])

        global_orient = global_orient_full[image_ind:image_ind+1]
        transl = transl_full[image_ind:image_ind+1]
        hand_pose = hand_pose_full[image_ind:image_ind+1]
        betas = betas_full[0:1]

        mano_out = mano(global_orient=global_orient,
                            hand_pose=hand_pose,
                            betas=betas,
                            transl=transl)
        
        vertices = mano_out.vertices[0]

        
        splat_cam_data = splat_data[image_ind]
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

        if renderer is None:
            renderer = pyrender.OffscreenRenderer(width, height)

        R = torch.FloatTensor(R).cuda()
        t = torch.FloatTensor(t).cuda()
        intrinsics = [fx, fy, cx, cy]
        dims =[height, width]

        parser = ArgumentParser()
        pipeline_par = PipelineParams(parser)
        args, _ = parser.parse_known_args()
        pipeline = pipeline_par.extract(args)
        #pipeline.convert_SHs_python = True

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        res_pgk = my_render(gaussians, pipeline, background, intrinsics, dims, R.T, t,
                            vertices=vertices)
        image = (res_pgk['render'].cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
        vertices = vertices.cpu().numpy()
        faces = get_faces()
        trimesh_model = trimesh.Trimesh(vertices, faces, process=False)
        mesh = pyrender.Mesh.from_trimesh(trimesh_model)
        mesh_node = py_scene.add(mesh)

        camera = pyrender.IntrinsicsCamera(fx=fx,
                                            fy=fy,
                                            cx=cx,
                                            cy=cy,
                                            znear=0.1,
                                            zfar=3000.0)
        
        M = np.eye(4)
        M_inv = np.linalg.inv(M)
        trans_c2w = M_inv @ get_opengl_to_opencv_camera_trans()

        camera_node = pyrender.Node(camera=camera, matrix=trans_c2w)
        py_scene.add_node(camera_node)

        light = pyrender.SpotLight(
                color=np.ones(3),
                intensity=2.4,
                innerConeAngle=np.pi / 16.0,
                outerConeAngle=np.pi / 6.0,
            )
        light_node = pyrender.Node(light=light, matrix=trans_c2w)
        py_scene.add_node(light_node)

        render_flags = pyrender.constants.RenderFlags.NONE
        color, depth = renderer.render(py_scene, flags=render_flags)
        mask = np.zeros_like(color)
        mask[depth > 0] = [128, 128, 128]

        py_scene.remove_node(camera_node)
        py_scene.remove_node(light_node)
        py_scene.remove_node(mesh_node)

        rend_im = res_pgk['render'].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        rend_im = cv2.cvtColor(rend_im, cv2.COLOR_RGB2BGR)

        mesh_im = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # IMAGE_DIR = '/home/hfreeman/harry_ws/gopro/datasets/simple_manip/hand_only/undistorted'
        IMAGE_DIR = '/home/hfreeman/harry_ws/gopro/datasets/simple_manip/1_prune_interact/undistorted'
        orig_im_path = os.path.join(IMAGE_DIR, entry["img_name"] + '.jpg')
        orig_im = cv2.imread(orig_im_path)
        
        comb_im = np.hstack((orig_im, rend_im, mesh_im))

        IM_DIR = '/home/hfreeman/harry_ws/gopro/debug_vis/vis_gauss_hand/'
        im_path = os.path.join(IM_DIR, entry["img_name"] + '.png')
        # torchvision.utils.save_image(res_pgk['render'], im_path)
        cv2.imwrite(im_path, comb_im)








SPLAT_PATH = 'output/comb_colmap/point_cloud/iteration_7000/point_cloud.ply'
SPLAT_JSON_PATH = '/home/hfreeman/harry_ws/repos/gaussian-mesh-splatting/output/comb_colmap/cameras.json'
with torch.no_grad():
    run(SPLAT_PATH, SPLAT_JSON_PATH)