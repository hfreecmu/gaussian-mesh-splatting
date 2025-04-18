import os
import numpy as np
import torch
import cv2
import trimesh
import torchvision
import pyrender
import pytorch3d.transforms

from vine_prune.utils.mano import get_mano, scale_mano, get_faces
from vine_prune.utils.general_utils import splat_to_image_color, read_mask
from vine_prune.utils.io import read_pickle, read_json, read_np_data

import sys
sys.path.append('/home/hfreeman/harry_ws/repos/pruner_track/submodules/gaussian-mesh-splatting')
# from scene import GaussianModel
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
from argparse import ArgumentParser
from arguments import PipelineParams
from renderer.gaussian_renderer import my_render
from scene.gaussian_model import GaussianModel
from train_comb import trans_gaussians, merge_gaussians

def get_opengl_to_opencv_camera_trans() -> np.ndarray:
    """Returns a transformation from OpenGL to OpenCV camera frame.

    Returns:
        A 4x4 transformation matrix (flipping Y and Z axes).
    """

    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    return yz_flip

def get_splat_path(splat_res_dir, target_itr=None):
    max_itr = -1
    for subdir_name in os.listdir(os.path.join(splat_res_dir, 'point_cloud')):
        itr = int(subdir_name.split('_')[-1])
        if target_itr is not None and target_itr != itr:
            continue
        if itr > max_itr:
            max_itr = itr
            splat_path = os.path.join(splat_res_dir, 'point_cloud', subdir_name, 'point_cloud.ply')
            splat_dir = os.path.join(splat_res_dir, 'point_cloud', subdir_name)

    return splat_path, splat_dir

def debug_hand_gauss(hand_gaussians,
                    view_R_hand,
                    hand_xyz, hand_rots, hand_scaling):
    
    merged_gaussians = GaussianModel(hand_gaussians.max_sh_degree)
    merged_gaussians.active_sh_degree = hand_gaussians.max_sh_degree
    
    merged_gaussians._xyz = hand_xyz
    merged_gaussians._features_dc = hand_gaussians._features_dc
    merged_gaussians._features_rest = hand_gaussians._features_rest
    merged_gaussians._opacity = hand_gaussians._opacity
    merged_gaussians._scaling = hand_scaling
    merged_gaussians._rotation = hand_rots
    
    merged_view_R = view_R_hand

    return merged_gaussians, merged_view_R, None

def run(splat_res_dir, splat_data_dir, image_dir, hand_mask_dir, output_dir, make_vid, fps,
        mano_data_path):
    splat_path, splat_dir = get_splat_path(splat_res_dir, target_itr=None)

    gaussians = GaussianMeshModel(3)
    gaussians.load_ply(splat_path)

    gaussians.update_alpha()
    gaussians.prepare_scaling_rot()

    # scene_splat_path = os.path.join(splat_data_dir, 'scene.ply')
    # #scene_splat_path = os.path.join(splat_dir, 'scene_point_cloud.ply')
    # scene_gaussians = GaussianModel(3)
    # scene_gaussians.load_ply(scene_splat_path)

    # obj_splat_path = os.path.join(splat_data_dir, 'obj_splat.ply')
    obj_splat_path = os.path.join(splat_dir, 'obj_point_cloud.ply')
    obj_gaussians = GaussianModel(3)
    obj_gaussians.load_ply(obj_splat_path)

    if mano_data_path is None:
        mano_data_path = os.path.join(splat_dir, 'hold_opt_ho.npy')
    mano_data = read_np_data(mano_data_path)

    obj_one_hot_path = os.path.join(splat_dir, 'obj_one_hot.pth')
    obj_one_hot = torch.load(obj_one_hot_path)

    hand_one_hot_path = os.path.join(splat_dir, 'hand_one_hot.pth')
    hand_one_hot = torch.load(hand_one_hot_path)

    color_path = os.path.join(splat_dir, 'color.pth')
    if os.path.exists(color_path):
        use_bg_color = True
        scene_brightness = torch.load(color_path)
        color_dict_path = os.path.join(splat_dir, 'color_dict.pkl')
        scene_bright_map = read_pickle(color_dict_path)
    else:
        use_bg_color = False

    obj_rot = torch.from_numpy(mano_data['object']['global_orient']).float().cuda()
    obj_rot = pytorch3d.transforms.axis_angle_to_quaternion(obj_rot)
    obj_trans = torch.from_numpy(mano_data['object']['transl']).float().cuda()

    # scene_rot = torch.from_numpy(mano_data['scene']['global_orient']).float().cuda()
    # scene_rot = pytorch3d.transforms.axis_angle_to_quaternion(scene_rot)
    # scene_trans = torch.from_numpy(mano_data['scene']['transl']).float().cuda()

    ###
    mano = get_mano(flat_hand_mean=True,
                    use_pca=False)
    ###

    global_orient_full = gaussians.global_orient
    transl_full = gaussians.transl
    hand_pose_full = gaussians.hand_pose
    betas_full = gaussians.betas
    hand_scale = gaussians.hand_scale

    splat_json_path = os.path.join(splat_res_dir, 'cameras.json')
    splat_data = read_json(splat_json_path)

    ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
    py_scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=ambient_light)

    renderer = None

    rotation_activation = torch.nn.functional.normalize

    vid_writer = None

    for ind, entry in enumerate(splat_data):
        image_ind = int(entry["img_name"])

        global_orient = global_orient_full[image_ind:image_ind+1]
        transl = transl_full[image_ind:image_ind+1]
        hand_pose = hand_pose_full[image_ind:image_ind+1]
        betas = betas_full[0:1]

        obj_rot_frame = pytorch3d.transforms.quaternion_to_matrix(rotation_activation(obj_rot[image_ind:image_ind+1])).squeeze(0)
        obj_trans_frame = obj_trans[image_ind:image_ind+1].squeeze(0)
        # scene_rot_frame = pytorch3d.transforms.quaternion_to_matrix(rotation_activation(scene_rot[image_ind:image_ind+1])).squeeze(0)
        # scene_trans_frame = scene_trans[image_ind:image_ind+1].squeeze(0)

        obj_gaussians_trans = trans_gaussians(obj_gaussians, obj_rot_frame, obj_trans_frame, 
                                              None, harmonic=False, should_copy=False)
        view_R_obj = (obj_rot_frame.T).repeat(obj_gaussians_trans._xyz.shape[0], 1, 1)
        # scene_gaussians_trans = trans_gaussians(scene_gaussians, scene_rot_frame, scene_trans_frame, 
        #                                       None, harmonic=False, should_copy=False)
        # view_R_scene = (scene_rot_frame.T).repeat(scene_gaussians_trans._xyz.shape[0], 1, 1)

        mano_out = mano(global_orient=global_orient,
                            hand_pose=hand_pose,
                            betas=betas,
                            )
        scale_mano(mano_out, mano, betas, hand_scale) 
        vertices = mano_out.vertices + transl[:, None]
        vertices = vertices[0]
        
        _, hand_rots, _ = gaussians.get_xyz_from_verts(vertices)
        hand_rots_r = pytorch3d.transforms.quaternion_to_matrix(hand_rots)
        view_R_hand = hand_rots_r.permute(0, 2, 1)

        hand_xyz, hand_rots, hand_scaling = gaussians.get_xyz_from_verts(vertices, activate=False)
        
        merged_gaussians, merged_view_R = merge_gaussians(gaussians, obj_gaussians_trans, None,
                                                          view_R_hand, view_R_obj, None,
                                                          hand_xyz, hand_rots, hand_scaling, include_scene=False)
        # merged_gaussians, merged_view_R, one_hot_labels = debug_hand_gauss(gaussians, view_R_hand, 
        #                                                    hand_xyz, hand_rots, hand_scaling)
        # merged_gaussians, merged_view_R = scene_gaussians_trans, view_R_scene
        # merged_gaussians, merged_view_R, one_hot_labels = obj_gaussians_trans, view_R_obj, None

        splat_cam_data = splat_data[ind]
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

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # hand_one_hot_pad = torch.stack((torch.sigmoid(hand_one_hot), torch.zeros_like(hand_one_hot)), dim=-1)
        # obj_one_hot_pad = torch.stack((torch.zeros_like(obj_one_hot), torch.sigmoid(obj_one_hot)), dim=-1)
        hand_one_hot_pad = torch.stack((hand_one_hot, torch.zeros_like(hand_one_hot)), dim=-1)
        obj_one_hot_pad = torch.stack((torch.zeros_like(obj_one_hot), obj_one_hot), dim=-1)
        one_hot_labels = torch.concat((hand_one_hot_pad, obj_one_hot_pad))

        if use_bg_color:
            frame_scene_brightness = scene_brightness[scene_bright_map[image_ind]]
        else:
            frame_scene_brightness = None

        res_pgk = my_render(merged_gaussians, pipeline, background, intrinsics, dims, R.T, t,
                            vertices=None, view_R=merged_view_R, one_hot_labels=one_hot_labels,
                            color_offset=frame_scene_brightness)
        image = splat_to_image_color(res_pgk['render'])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        label_res = res_pgk['extra']
        if one_hot_labels is not None:
            hand_label_res = label_res[0:1]
            hand_label_res = splat_to_image_color(hand_label_res)
            hand_label_res = np.concatenate([hand_label_res]*3, axis=-1)

            object_label_res = label_res[1:2]
            object_label_res = splat_to_image_color(object_label_res)
            object_label_res = np.concatenate([object_label_res]*3, axis=-1)

            scene_label_res = label_res[2:3]
            scene_label_res = splat_to_image_color(scene_label_res)
            scene_label_res = np.concatenate([scene_label_res]*3, axis=-1)
        
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

        mesh_im = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        
        orig_im_path = os.path.join(image_dir, entry["img_name"] + '.jpg')
        if not os.path.exists(orig_im_path):
            orig_im_path = os.path.join(image_dir, entry["img_name"] + '.png')
        orig_im = cv2.imread(orig_im_path)

        orig_hand_mask_path = os.path.join(hand_mask_dir, entry["img_name"] + '.png')
        orig_hand_mask = read_mask(orig_hand_mask_path)
        orig_hand_mask = np.stack([orig_hand_mask]*3, axis=-1)
        
        comb_im = np.hstack((orig_im, image, mesh_im))
        comb_hand_mask_im = np.hstack((orig_hand_mask, hand_label_res, cv2.addWeighted(image, 0.5, mesh_im, 0.5, 0) ))

        im_path = os.path.join(output_dir, entry["img_name"] + '.png')
        # torchvision.utils.save_image(res_pgk['render'], im_path)
        cv2.imwrite(im_path, comb_im)

        im_hand_path = os.path.join(output_dir, entry["img_name"] + '_hand.png')
        cv2.imwrite(im_hand_path, comb_hand_mask_im)

        if make_vid:
            if vid_writer is None:
                vid_path = os.path.join(OUTPUT_DIR, 'res.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, (image.shape[1], image.shape[0]))
            vid_writer.write(image)


        if True or ind == 0:
            obj_gaussians_trans = trans_gaussians(obj_gaussians, obj_rot_frame, obj_trans_frame, 
                                              None, harmonic=True, should_copy=False)
            # scene_gaussians_trans = trans_gaussians(scene_gaussians, scene_rot_frame, scene_trans_frame, 
            #                                   None, harmonic=True, should_copy=False)

            view_R_obj = torch.zeros_like(view_R_obj)
            view_R_obj[:] = torch.eye(3)

            # view_R_scene = torch.zeros_like(view_R_scene)
            # view_R_scene[:] = torch.eye(3)

            merged_gaussians, merged_view_R = merge_gaussians(gaussians, obj_gaussians_trans, None,
                                                          view_R_hand, view_R_obj, None,
                                                          hand_xyz, hand_rots, hand_scaling, include_scene=False)
            ply_path = os.path.join(output_dir, f'merged_{entry["img_name"]}.ply')
            merged_gaussians.save_ply(ply_path)

            hand_gaussians, _, _ = debug_hand_gauss(gaussians, view_R_hand, hand_xyz, hand_rots, hand_scaling)

            hand_ply_path = os.path.join(output_dir, f'merged_{entry["img_name"]}_hand.ply')
            hand_gaussians.save_ply(hand_ply_path)

    if vid_writer is not None:
        vid_writer.release()

EXP_NAME = '3_000663'
MANO_DATA_PATH = None
# MANO_DATA_PATH = f'/home/hfreeman/harry_ws/repos/pruner_track/datasets/DEXYCB/{EXP_NAME}/hand_obj/hold_refine_ho_fine.npy'
SPLAT_RES_DIR = f'/home/hfreeman/harry_ws/repos/pruner_track/submodules/gaussian-mesh-splatting/output/DEXYCB/{EXP_NAME}_REFINE'
# SPLAT_RES_DIR = f'output/0_pruner_rotate_single'
# SPLAT_DATA_DIR = f'/home/hfreeman/harry_ws/repos/pruner_track/submodules/gaussian-mesh-splatting/data/{EXP_NAME}'
SPLAT_DATA_DIR=None
IMAGE_DIR = f'/home/hfreeman/harry_ws/repos/pruner_track/datasets/DEXYCB/{EXP_NAME}/undistorted'
HAND_MASK_DIR = f'/home/hfreeman/harry_ws/repos/pruner_track/datasets/DEXYCB/{EXP_NAME}/mask_hand'
OUTPUT_DIR = f'/home/hfreeman/Downloads/vis_gauss_hand/{EXP_NAME}'
MAKE_VID = True
FPS=10

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
with torch.no_grad():
    run(SPLAT_RES_DIR, SPLAT_DATA_DIR, IMAGE_DIR, HAND_MASK_DIR, OUTPUT_DIR, MAKE_VID, FPS,
        MANO_DATA_PATH)