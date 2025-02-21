#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from renderer.gaussian_renderer import render, network_gui
import sys
from scene import Scene
from games import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import pyrender
import numpy as np
import trimesh
from scene.gaussian_model import GaussianModel
from vine_prune.utils.mano import get_mano, scale_mano, get_faces, get_contact_idx
from vine_prune.utils.io import read_np_data
import pytorch3d.transforms
import torch.nn as nn
from vine_prune.utils.general_utils import splat_to_image_color
import copy
from utils.rotation_main import rotate_splat_cuda, rotate_splat_cuda_angle
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import fov2focal, c2c_orig
from scipy.spatial import cKDTree
from pytorch3d.ops import knn_points
import kaolin

def compute_mano_cano_sdf(mesh_v_cano, mesh_f_cano, x_cano):
    distance = knn_points(x_cano, mesh_v_cano, K=1, return_nn=False)[0][:, :, 0]

    # inside or not
    sign = kaolin.ops.mesh.check_sign(mesh_v_cano.detach(), mesh_f_cano.detach(), x_cano.detach()).float()

    # inside: 1 -> 1 - 2 = -1, negative
    # outside: 0 -> 1 - 0 = 1, positive
    sign = 1 - 2 * sign
    signed_distance = sign * distance  # SDF of points to mesh
    return signed_distance

def get_opengl_to_opencv_camera_trans() -> np.ndarray:
    """Returns a transformation from OpenGL to OpenCV camera frame.

    Returns:
        A 4x4 transformation matrix (flipping Y and Z axes).
    """

    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    return yz_flip

def disable_grad(gaussians: GaussianModel, full=False):
    gaussians._xyz.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    #gaussians._opacity.requires_grad_(False)

    if full:
        gaussians._features_dc.requires_grad_(False)
        gaussians._features_rest.requires_grad_(False)
        gaussians._opacity.requires_grad_(False)

def trans_gaussians(splatt_gaussians, rot, trans, scale, is_rot_angle=False,
                    harmonic=True, should_copy=True):
    if should_copy:
        splatt_gaussians = copy.deepcopy(splatt_gaussians)
    else:
        splatt_gaussians_orig = splatt_gaussians
        splatt_gaussians = GaussianModel(splatt_gaussians_orig.max_sh_degree)
        splatt_gaussians._xyz = torch.clone(splatt_gaussians_orig._xyz)
        splatt_gaussians._features_dc = torch.clone(splatt_gaussians_orig._features_dc)
        splatt_gaussians._features_rest = torch.clone(splatt_gaussians_orig._features_rest)
        splatt_gaussians._opacity = torch.clone(splatt_gaussians_orig._opacity)
        splatt_gaussians._scaling = torch.clone(splatt_gaussians_orig._scaling)
        splatt_gaussians._rotation = torch.clone(splatt_gaussians_orig._rotation)
        splatt_gaussians.active_sh_degree = splatt_gaussians_orig.max_sh_degree

    if rot is not None:
        if not is_rot_angle:
            # this one is assuming matrix already
            rotate_splat_cuda(splatt_gaussians, rot, harmonic=harmonic)
        else:
            rotate_splat_cuda_angle(splatt_gaussians, rot, harmonic=harmonic)

    if scale is not None:
        curr_gauss_scale = splatt_gaussians.get_scaling
        new_gauss_scale = curr_gauss_scale * scale
        
        splatt_gaussians._scaling = splatt_gaussians.scaling_inverse_activation(new_gauss_scale)
        splatt_gaussians._xyz *= scale

    if trans is not None:
        splatt_gaussians._xyz = splatt_gaussians.get_xyz + trans

    return splatt_gaussians

def sel_gaussians(gaussians, view_R, one_hot_labels, indices):
    new_gaussians = GaussianModel(gaussians.max_sh_degree)
    new_gaussians.active_sh_degree = gaussians.max_sh_degree

    new_gaussians._xyz = gaussians._xyz[indices]
    new_gaussians._features_dc = gaussians._features_dc[indices]
    new_gaussians._features_rest = gaussians._features_rest[indices]
    new_gaussians._opacity = gaussians._opacity[indices]
    new_gaussians._scaling = gaussians._scaling[indices]
    new_gaussians._rotation = gaussians._rotation[indices]

    if view_R is not None:
        new_view_R = view_R[indices]
    else:
        new_view_R = None

    if one_hot_labels is not None:
        new_one_hot_labels = one_hot_labels[indices]
    else:
        new_one_hot_labels = None

    return new_gaussians, new_view_R, new_one_hot_labels

def merge_gaussians(hand_gaussians, obj_gaussians, scene_gaussians,
                    view_R_hand, view_R_obj, view_R_scene,
                    hand_xyz, hand_rots, hand_scaling,
                    include_scene=True):
    
    merged_gaussians = GaussianModel(hand_gaussians.max_sh_degree)
    # max or active?
    merged_gaussians.active_sh_degree = hand_gaussians.max_sh_degree
    
    if include_scene:
        merged_gaussians._xyz = torch.concat((hand_xyz, obj_gaussians._xyz, scene_gaussians._xyz), dim=0)
        merged_gaussians._features_dc = torch.concat((hand_gaussians._features_dc, obj_gaussians._features_dc, scene_gaussians._features_dc), dim=0)
        merged_gaussians._features_rest = torch.concat((hand_gaussians._features_rest, obj_gaussians._features_rest, scene_gaussians._features_rest), dim=0)
        merged_gaussians._opacity = torch.concat((hand_gaussians._opacity, obj_gaussians._opacity, scene_gaussians._opacity), dim=0)
        merged_gaussians._scaling = torch.concat((hand_scaling, obj_gaussians._scaling, scene_gaussians._scaling), dim=0)
        merged_gaussians._rotation = torch.concat((hand_rots, obj_gaussians._rotation, scene_gaussians._rotation), dim=0)
        
        merged_view_R = torch.concat((view_R_hand, view_R_obj, view_R_scene))

        # one_hot_labels_hand = torch.zeros(hand_xyz.shape[0], 3).float().cuda()
        # one_hot_labels_obj = torch.zeros(obj_gaussians._xyz.shape[0], 3).float().cuda()
        # one_hot_labels_scene = torch.zeros(scene_gaussians._xyz.shape[0], 3).float().cuda()

        # one_hot_labels_hand[:, 0] = 1.0
        # one_hot_labels_obj[:, 1] = 1.0
        # one_hot_labels_scene[:, 2] = 1.0

        # one_hot_labels = torch.concat((one_hot_labels_hand, one_hot_labels_obj, one_hot_labels_scene))
    else:
        merged_gaussians._xyz = torch.concat((hand_xyz, obj_gaussians._xyz), dim=0)
        merged_gaussians._features_dc = torch.concat((hand_gaussians._features_dc, obj_gaussians._features_dc), dim=0)
        merged_gaussians._features_rest = torch.concat((hand_gaussians._features_rest, obj_gaussians._features_rest), dim=0)
        merged_gaussians._opacity = torch.concat((hand_gaussians._opacity, obj_gaussians._opacity), dim=0)
        merged_gaussians._scaling = torch.concat((hand_scaling, obj_gaussians._scaling), dim=0)
        merged_gaussians._rotation = torch.concat((hand_rots, obj_gaussians._rotation), dim=0)
        
        merged_view_R = torch.concat((view_R_hand, view_R_obj))

        # one_hot_labels_hand = torch.zeros(hand_xyz.shape[0], 2).float().cuda()
        # one_hot_labels_obj = torch.zeros(obj_gaussians._xyz.shape[0], 2).float().cuda()

        # one_hot_labels_hand[:, 0] = 1.0
        # one_hot_labels_obj[:, 1] = 1.0

        # one_hot_labels = torch.concat((one_hot_labels_hand, one_hot_labels_obj))

    return merged_gaussians, merged_view_R#, one_hot_labels

def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz, train_scene=False, use_bg_color=False):
    
    ### pyrender stuff
    ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
    py_scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=ambient_light)
    renderer = None
    ###
    
    ###
    mano = get_mano()
    contact_idx = torch.LongTensor(get_contact_idx())
    ###
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree)

    ###
    mano_data_path = os.path.join(dataset.source_path, 'hold_init_ho_scene.npy')
    #scene_splat_path = os.path.join(dataset.source_path, 'scene.ply')
    obj_splat_path = os.path.join(dataset.source_path, 'obj_splat.ply')

    # scene_gaussians = GaussianModel(3)
    # scene_gaussians.load_ply(scene_splat_path)
    # scene_gaussians.spatial_lr_scale = 5.0

    obj_gaussians = GaussianModel(3)
    obj_gaussians.load_ply(obj_splat_path)
    # obj_gaussians.spatial_lr_scale = 5.0

    # disable_grad(scene_gaussians, full=True)
    disable_grad(obj_gaussians, full=True)

    mano_data = read_np_data(mano_data_path)

    gaussians.set_hand_data(mano_data)

    obj_rot = torch.from_numpy(mano_data['object']['global_orient']).float().cuda()
    obj_rot = pytorch3d.transforms.axis_angle_to_quaternion(obj_rot)
    obj_rot = nn.Parameter(obj_rot)
    obj_trans = torch.from_numpy(mano_data['object']['transl']).float().cuda()
    obj_trans = nn.Parameter(obj_trans)
    hand_faces = get_faces()
    hand_faces_torch = torch.LongTensor(hand_faces).cuda()

    # obj_eid = torch.randn(obj_gaussians._xyz.shape[0], 16).float().cuda()
    # obj_eid = nn.Parameter(obj_eid)

    # scene_rot = torch.from_numpy(mano_data['scene']['global_orient']).float().cuda()
    # scene_rot = pytorch3d.transforms.axis_angle_to_quaternion(scene_rot)
    # scene_trans = torch.from_numpy(mano_data['scene']['transl']).float().cuda()
    
    # if train_scene:
    #     scene_rot = nn.Parameter(scene_rot)
    #     scene_trans = nn.Parameter(scene_trans)

    scene = Scene(dataset, gaussians)

    # gauss_eid = torch.randn(gaussians._xyz.shape[0], 16).float().cuda()
    # gauss_eid = nn.Parameter(gauss_eid)
    # seg_mlp = nn.Linear(16, 3).cuda()

    obj_one_hot = inverse_sigmoid(torch.zeros(obj_gaussians._xyz.shape[0]).float().cuda() + 0.5)
    obj_one_hot = nn.Parameter(obj_one_hot)

    hand_one_hot = inverse_sigmoid(torch.zeros(gaussians._xyz.shape[0]).float().cuda() + 0.5)
    hand_one_hot = nn.Parameter(hand_one_hot)

    cano_obj_pts = torch.FloatTensor(mano_data['object']['cano_v3d']).cuda()

    num_frames = len(scene.getTrainCameras().copy())

    # obj_brightness = torch.zeros(num_frames, 3).float().cuda()
    # obj_brightness = nn.Parameter(obj_brightness)
    if use_bg_color:
        scene_brightness = torch.zeros(num_frames, 3).float().cuda()
        scene_brightness = nn.Parameter(scene_brightness)

    rotation_activation = nn.functional.normalize
    l_params = [
        {'params': [obj_rot], 'lr': 1e-4, "name": "obj_rot"},
        {'params': [obj_trans], 'lr': 1e-5, "name": "obj_trans"},
        # {'params': [obj_eid], 'lr': 1e-5, "name": "obj_eid"},
        # {'params': [gauss_eid], 'lr': 1e-5, "name": "gauss_eid"},
        # {'params': seg_mlp.parameters(), 'lr': 1e-4, "name": "seg_mlp"}
        {'params': [obj_one_hot], 'lr': 1e-5, "name": "obj_one_hot"},
        {'params': [hand_one_hot], 'lr': 1e-5, "name": "gauss_one_hot"},
    ]

    #opacity_loss = torch.nn.BCEWithLogitsLoss()

    if use_bg_color:
        l_params.append({'params': [scene_brightness], 'lr': 1e-3, "name": "scene_brightness"})

    # if train_scene:
    #     l_params.append({'params': [scene_rot], 'lr': 1e-3, "name": "scene_rot"})
    #     l_params.append({'params': [scene_trans], 'lr': 1e-3, "name": "scene_trans"})

    optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)

    #obj_gaussians.training_setup(opt)
    #scene_gaussians.training_setup(opt)
    ###

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    BATCH_SIZE = min(200, len(scene.getTrainCameras()))
    losses = []
    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        #obj_gaussians.update_learning_rate(iteration)
        #scene_gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        vertices = None
        gt_image = viewpoint_cam.original_image.cuda()
        gt_hand_mask = viewpoint_cam.hand_mask.cuda()
        gt_object_mask = viewpoint_cam.object_mask.cuda()

        net_mask = torch.clamp(gt_hand_mask + gt_object_mask, 0.0, 1.0)
        gt_image = torch.where(net_mask > 0, gt_image, torch.zeros_like(gt_image))

        image_name = viewpoint_cam.image_name
        image_ind = int(image_name)

        global_orient = gaussians.global_orient[image_ind:image_ind+1]
        transl = gaussians.transl[image_ind:image_ind+1]
        hand_pose = gaussians.hand_pose[image_ind:image_ind+1]
        betas = gaussians.betas[0:1]
        hand_scale = gaussians.hand_scale

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

        # merged_gaussians, merged_view_R, one_hot_labels = merge_gaussians(gaussians, obj_gaussians_trans, scene_gaussians_trans,
        #                                                                   view_R_hand, view_R_obj, view_R_scene,
        #                                                                   hand_xyz, hand_rots, hand_scaling)
        merged_gaussians, merged_view_R = merge_gaussians(gaussians, obj_gaussians_trans, None,
                                                                          view_R_hand, view_R_obj, None,
                                                                          hand_xyz, hand_rots, hand_scaling,
                                                                          include_scene=False)
        
        hand_one_hot_pad = torch.stack((torch.sigmoid(hand_one_hot), torch.zeros_like(hand_one_hot)), dim=-1)
        obj_one_hot_pad = torch.stack((torch.zeros_like(obj_one_hot), torch.sigmoid(obj_one_hot)), dim=-1)
        one_hot_labels = torch.concat((hand_one_hot_pad, obj_one_hot_pad))
        
        # one_hot_labels = torch.concat((gauss_eid, obj_eid))
        
        render_pkg = render(viewpoint_cam, merged_gaussians, pipe, bg, 
                            vertices=None, view_R=merged_view_R,
                            one_hot_labels=one_hot_labels)
        
        sel_hand_gaussian_model, sel_view_R_hand, sel_one_hot_labels_hand = sel_gaussians(merged_gaussians, merged_view_R, 
                                                                                          one_hot_labels[:, 0:1], torch.arange(hand_xyz.shape[0]))

        hand_render_pkg = render(viewpoint_cam, sel_hand_gaussian_model, pipe, bg, 
                            vertices=None, view_R=sel_view_R_hand,
                            one_hot_labels=sel_one_hot_labels_hand)
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # hand_colors = render_pkg['colors_precomp'][0:gaussians.get_xyz.shape[0]]
        # hand_xyz_np = hand_xyz.detach().cpu().numpy()
        # tree = cKDTree(hand_xyz_np)
        # indices = tree.query(hand_xyz_np, k=11)[0][:, 1:]
        # color_diffs = hand_colors[:, None] - hand_colors[indices]
        # color_loss = 0.1*(color_diffs**2).mean()

        if use_bg_color:
            frame_scene_brightness = scene_brightness[image_ind]
            image = image + frame_scene_brightness[:, None, None]

        # label_res = render_pkg['extra'].permute(1,2,0)
        label_res = render_pkg['extra']
        hand_label_res = label_res[0:1]
        obj_label_res = label_res[1:2]

        sel_hand_label_res = hand_render_pkg['extra']

        # label_res = seg_mlp(label_res)

        #scene_label_res = label_res[2:3]

        # ###
        # import cv2
        # # vis_im = splat_to_image_color(image)
        # # vis_im = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
        # # cv2.imshow('test', vis_im)

        # # vis_gt_im = splat_to_image_color(gt_image)
        # # vis_gt_im = cv2.cvtColor(vis_gt_im, cv2.COLOR_RGB2BGR)
        # # cv2.imshow('gt', vis_gt_im)
        # # cv2.waitKey(0)

        # vis_im = splat_to_image_color(gt_object_mask)[:, :, 0]
        # vis_im = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
        # cv2.imshow('test', vis_im)
        # cv2.waitKey(0)
        # breakpoint()
        # ###

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        #Ll1 = l2_loss(image, gt_image)
        #loss = Ll1

        # gt_seg = torch.concatenate((gt_hand_mask, gt_object_mask, torch.zeros_like(gt_hand_mask))).permute(1, 2, 0)
        # mask_loss = nn.functional.cross_entropy(label_res, gt_seg)

        hand_mask_loss = l1_loss(hand_label_res, gt_hand_mask)
        obj_mask_loss = l1_loss(obj_label_res, gt_object_mask)
        # hand_mask_loss = 0.5*l1_loss(hand_label_res, gt_hand_mask)
        # obj_mask_loss = 0.5*l1_loss(obj_label_res, gt_object_mask)
        #gt_scene_mask = 1.0 - (gt_hand_mask + gt_object_mask).clamp(0.0, 1.0)
        #scene_mask_loss = 0.5*l1_loss(scene_label_res, gt_scene_mask)

        # op_loss = 0.1*opacity_loss(gaussians._opacity, torch.ones_like(gaussians._opacity))
        #op_loss = 0.0

        ### mano mask loss
        if renderer is None:
            renderer = pyrender.OffscreenRenderer(gt_image.shape[2], gt_image.shape[1])

        trimesh_model = trimesh.Trimesh(vertices.detach().cpu().numpy(), hand_faces, process=False)
        mesh = pyrender.Mesh.from_trimesh(trimesh_model)
        mesh_node = py_scene.add(mesh)

        fx = fov2focal(viewpoint_cam.FoVx, gt_image.shape[2])
        fy = fov2focal(viewpoint_cam.FoVy, gt_image.shape[1])
        cx = c2c_orig(viewpoint_cam.cx, gt_image.shape[2])
        cy = c2c_orig(viewpoint_cam.cy, gt_image.shape[1])
        camera = pyrender.IntrinsicsCamera(fx=fx,
                                           fy=fy,
                                           cx=cx,
                                           cy=cy,
                                           znear=0.1,
                                           zfar=3000.0)

        M = np.eye(4)
        M[0:3, 0:3] = viewpoint_cam.R.T
        M[0:3, 3] = viewpoint_cam.T
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
        _, depth = renderer.render(py_scene, flags=render_flags)

        mano_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=float)
        mano_mask[depth > 0] = 1.0
        mano_mask = torch.from_numpy(mano_mask).float().cuda().unsqueeze(0)

        # should mano mask loss be back prop?
        mano_mask_loss = l1_loss(sel_hand_label_res, mano_mask)

        py_scene.remove_node(camera_node)
        py_scene.remove_node(light_node)
        py_scene.remove_node(mesh_node)
        ###

        v3d_tips = vertices[contact_idx]
        v3d_object = cano_obj_pts @ obj_rot_frame.T + obj_trans_frame

        contact_loss = knn_points(v3d_tips.unsqueeze(0), v3d_object.unsqueeze(0), K=1, return_nn=False)[0]
        contact_loss = contact_loss.mean()

        sdf = compute_mano_cano_sdf(vertices.unsqueeze(0), hand_faces_torch, v3d_object.unsqueeze(0)).squeeze(0)
        sdf_loss = torch.clamp(-sdf, min=0.00, max=0.005).mean()

        loss = loss + 0.5*hand_mask_loss + 0.5*obj_mask_loss + 0.5*mano_mask_loss + 1.0*contact_loss + 1.0*sdf_loss#+ color_loss
        losses.append(loss)

        if len(losses) == BATCH_SIZE:
            losses = torch.stack(losses).sum()
            losses.backward()

            # Optimizer step
            if iteration < opt.iterations:
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                #obj_gaussians.optimizer.step()
                #obj_gaussians.optimizer.zero_grad(set_to_none=True)

                #scene_gaussians.optimizer.step()
                #scene_gaussians.optimizer.zero_grad(set_to_none=True)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses = []
        
        #loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

                point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))

                mano_data['object']['global_orient'][:] = pytorch3d.transforms.quaternion_to_axis_angle(rotation_activation(obj_rot.detach())).cpu().numpy()
                mano_data['object']['transl'][:] = obj_trans.detach().cpu().numpy()
                #mano_data['scene']['transl'][:] = pytorch3d.transforms.quaternion_to_axis_angle(rotation_activation(scene_rot.detach())).cpu().numpy()
                #mano_data['scene']['transl'][:] = scene_trans.detach().cpu().numpy()
                mano_data['right']['global_orient'][:] = gaussians.global_orient.detach().cpu().numpy()
                mano_data['right']['betas'][:] = gaussians.betas[0].detach().cpu().numpy()
                mano_data['right']['hand_pose'][:] = gaussians.hand_pose.detach().cpu().numpy()
                mano_data['right']['transl'][:] = gaussians.transl.detach().cpu().numpy()
                mano_data['right']['scale'] = gaussians.hand_scale.detach().cpu().numpy()
                mano_path = os.path.join(point_cloud_path, 'hold_opt_ho.npy')
                np.save(mano_path, mano_data)

                obj_gaussians.save_ply(os.path.join(point_cloud_path, "obj_point_cloud.ply"))
                #scene_gaussians.save_ply(os.path.join(point_cloud_path, "scene_point_cloud.ply"))

                path_obj_one_hot = os.path.join(point_cloud_path, 'obj_one_hot.pth')
                torch.save(obj_one_hot, path_obj_one_hot)

                path_hand_one_hot = os.path.join(point_cloud_path, 'hand_one_hot.pth')
                torch.save(hand_one_hot, path_hand_one_hot)

            # # Densification
            # if (args.gs_type == "gs") or (args.gs_type == "gs_flat"):
            #     if iteration < opt.densify_until_iter:
            #         # Keep track of max radii in image-space for pruning
            #         gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
            #                                                              radii[visibility_filter])
            #         gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #         if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #             size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #             gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
            #                                         size_threshold)

            #         if iteration % opt.opacity_reset_interval == 0 or (
            #                 dataset.white_background and iteration == opt.densify_from_iter):
            #             gaussians.reset_opacity()
            # # else:
            # #     if iteration % opt.opacity_reset_interval == 0:
            # #         gaussians.reset_opacity()

            # # Optimizer step
            # if iteration < opt.iterations:
            #     gaussians.optimizer.step()
            #     gaussians.optimizer.zero_grad(set_to_none=True)

            #     #obj_gaussians.optimizer.step()
            #     #obj_gaussians.optimizer.zero_grad(set_to_none=True)

            #     #scene_gaussians.optimizer.step()
            #     #scene_gaussians.optimizer.zero_grad(set_to_none=True)

            #     optimizer.step()
            #     optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 40_000, 50_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 40_000, 50_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.num_splats = args.num_splats
    lp.meshes = args.meshes
    lp.gs_type = args.gs_type

    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.save_xyz
    )

    # All done
    print("\nTraining complete.")

# python3 train.py -s data/hand_colmap/ -m output/hand_colmap --gs_type gs_mesh --meshes 'mesh' --num_splats 5 --iterations 30000 --sh_degree 3 --resolution 2
