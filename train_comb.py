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

import numpy as np
from scene.gaussian_model import GaussianModel
from vine_prune.utils.mano import get_mano, scale_mano, get_faces, get_contact_idx
from vine_prune.utils.io import read_np_data
import pytorch3d.transforms
import torch.nn as nn
# from vine_prune.utils.general_utils import splat_to_image_color
import copy
from utils.rotation_main import rotate_splat_cuda, rotate_splat_cuda_angle
from vine_prune.utils.io import write_pickle, read_pickle
import torch.nn.functional as F

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
    # new_gaussians.active_sh_degree = gaussians.active_sh_degree

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
    # merged_gaussians.active_sh_degree = hand_gaussians.active_sh_degree
    
    if include_scene:
        merged_gaussians._xyz = torch.concat((hand_xyz, obj_gaussians._xyz, scene_gaussians._xyz), dim=0)
        merged_gaussians._features_dc = torch.concat((hand_gaussians._features_dc, obj_gaussians._features_dc, scene_gaussians._features_dc), dim=0)
        merged_gaussians._features_rest = torch.concat((hand_gaussians._features_rest, obj_gaussians._features_rest, scene_gaussians._features_rest), dim=0)
        merged_gaussians._opacity = torch.concat((hand_gaussians._opacity, obj_gaussians._opacity, scene_gaussians._opacity), dim=0)
        merged_gaussians._scaling = torch.concat((hand_scaling, obj_gaussians._scaling, scene_gaussians._scaling), dim=0)
        merged_gaussians._rotation = torch.concat((hand_rots, obj_gaussians._rotation, scene_gaussians._rotation), dim=0)
        
        merged_view_R = torch.concat((view_R_hand, view_R_obj, view_R_scene))

    else:
        merged_gaussians._xyz = torch.concat((hand_xyz, obj_gaussians._xyz), dim=0)
        merged_gaussians._features_dc = torch.concat((hand_gaussians._features_dc, obj_gaussians._features_dc), dim=0)
        merged_gaussians._features_rest = torch.concat((hand_gaussians._features_rest, obj_gaussians._features_rest), dim=0)
        merged_gaussians._opacity = torch.concat((hand_gaussians._opacity, obj_gaussians._opacity), dim=0)
        merged_gaussians._scaling = torch.concat((hand_scaling, obj_gaussians._scaling), dim=0)
        merged_gaussians._rotation = torch.concat((hand_rots, obj_gaussians._rotation), dim=0)
        
        merged_view_R = torch.concat((view_R_hand, view_R_obj))

    return merged_gaussians, merged_view_R

def sample_sdf(sdf, min_bound, spacing, pts):
    D, _, _ = sdf.shape
    coords_idx = (pts - min_bound) / spacing
    coords_normalized = coords_idx / (D - 1) * 2 - 1

    sdf_to_use = sdf.unsqueeze(0).unsqueeze(0)
    grid_coords = coords_normalized[:, [2, 1, 0]]
    grid_coords = grid_coords.view(1, 1, 1, -1, 3) 
    sampled = F.grid_sample(sdf_to_use, grid_coords, align_corners=True, mode='bilinear', padding_mode='border')
    sampled = sampled.view(-1)
    return sampled

def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz, use_bg_color=False):
    
    ###
    mano = get_mano()
    ###
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree)

    ###
    mano_data_path = os.path.join(dataset.source_path, 'hold_init_ho_scene.npy')
    obj_splat_path = os.path.join(dataset.source_path, 'obj_splat.ply')

    obj_gaussians = GaussianModel(3)
    obj_gaussians.load_ply(obj_splat_path)

    # WARNING 
    # if not disable grad, clone in trans_gaussians might need to retain grad
    # see https://discuss.pytorch.org/t/how-does-clone-interact-with-backpropagation/8247/5
    disable_grad(obj_gaussians, full=True)

    mano_data = read_np_data(mano_data_path)

    gaussians.set_hand_data(mano_data)

    obj_rot = torch.from_numpy(mano_data['object']['global_orient']).float().cuda()
    obj_rot = pytorch3d.transforms.axis_angle_to_quaternion(obj_rot)
    obj_rot = nn.Parameter(obj_rot)
    obj_trans = torch.from_numpy(mano_data['object']['transl']).float().cuda()
    obj_trans = nn.Parameter(obj_trans)

    scene = Scene(dataset, gaussians)

    obj_one_hot = torch.zeros(obj_gaussians._xyz.shape[0]).float().cuda() + 0.5
    obj_one_hot = nn.Parameter(obj_one_hot)

    hand_one_hot = torch.zeros(gaussians._xyz.shape[0]).float().cuda() + 0.5
    hand_one_hot = nn.Parameter(hand_one_hot)

    sdf_dict = read_pickle(os.path.join(dataset.source_path, 'sdf.pkl'))
    sdf_torch = torch.FloatTensor(sdf_dict['sdf']).cuda()
    min_bound_torch = torch.FloatTensor(sdf_dict['min_bound']).cuda()
    spacing_torch = torch.FloatTensor(sdf_dict['spacing']).cuda()

    num_frames = len(scene.getTrainCameras().copy())

    if use_bg_color:
        scene_brightness = torch.zeros(num_frames, 3).float().cuda()
        scene_brightness = nn.Parameter(scene_brightness)

        scene_bright_map = {}
        scene_bright_count = 0

    rotation_activation = nn.functional.normalize
    l_params = [
        {'params': [obj_rot], 'lr': 1e-3, "name": "obj_rot"},
        {'params': [obj_trans], 'lr': 1e-3, "name": "obj_trans"},
        {'params': [obj_one_hot], 'lr': 1e-4, "name": "obj_one_hot"},
        {'params': [hand_one_hot], 'lr': 1e-4, "name": "gauss_one_hot"},
    ]

    if use_bg_color:
        l_params.append({'params': [scene_brightness], 'lr': 1e-4, "name": "scene_brightness"})

    optimizer = torch.optim.Adam(l_params, lr=0, eps=1e-15)

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
    BATCH_SIZE = min(50, len(scene.getTrainCameras()))
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

        obj_gaussians_trans = trans_gaussians(obj_gaussians, obj_rot_frame, obj_trans_frame, 
                                              None, harmonic=False, should_copy=False)
        view_R_obj = (obj_rot_frame.T).repeat(obj_gaussians_trans._xyz.shape[0], 1, 1)

        
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
                                                                          hand_xyz, hand_rots, hand_scaling,
                                                                          include_scene=False)
        
        hand_one_hot_pad = torch.stack((hand_one_hot, torch.zeros_like(hand_one_hot)), dim=-1)
        obj_one_hot_pad = torch.stack((torch.zeros_like(obj_one_hot), obj_one_hot), dim=-1)
        one_hot_labels = torch.concat((hand_one_hot_pad, obj_one_hot_pad))
        
        if use_bg_color:
            if image_ind not in scene_bright_map:
                scene_bright_map[image_ind] = scene_bright_count
                scene_bright_count += 1

            frame_scene_brightness = scene_brightness[scene_bright_map[image_ind]]
        else:
            frame_scene_brightness = None
        
        render_pkg = render(viewpoint_cam, merged_gaussians, pipe, bg, 
                            vertices=None, view_R=merged_view_R,
                            one_hot_labels=one_hot_labels,
                            color_offset=frame_scene_brightness)
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        label_res = render_pkg['extra']
        hand_label_res = label_res[0:1]
        obj_label_res = label_res[1:2]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        hand_mask_loss = l1_loss(hand_label_res, gt_hand_mask)
        obj_mask_loss = l1_loss(obj_label_res, gt_object_mask)

        obj_rot_frame_inv = obj_rot_frame.T
        obj_trans_frame_inv = -obj_rot_frame_inv@obj_trans_frame
        vertices_obj_cano = vertices @ obj_rot_frame_inv.T + obj_trans_frame_inv
        sdf = sample_sdf(sdf_torch, min_bound_torch, spacing_torch, vertices_obj_cano)
        sdf_loss = torch.clamp(-sdf, min=0.00, max=0.05).sum()

        rgb_coeff = torch.linspace(0.1, 1.0, opt.iterations + 1)[iteration]
        mask_coeff = torch.linspace(1.1, 0.1, opt.iterations + 1)[iteration]
        sdf_coeff = 0.1

        loss = rgb_coeff*loss + mask_coeff*hand_mask_loss + mask_coeff*obj_mask_loss + sdf_coeff*sdf_loss

        losses.append(loss)

        if len(losses) == BATCH_SIZE:
            losses = torch.stack(losses).mean()
            losses.backward()

            # Optimizer step
            if iteration < opt.iterations:

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                gaussians.clamp_opacity()
                gaussians.clamp_scale()

            losses = []
        
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
            if (iteration in saving_iterations or iteration % 20000 == 0):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

                point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))

                mano_data['object']['global_orient'][:] = pytorch3d.transforms.quaternion_to_axis_angle(rotation_activation(obj_rot.detach())).cpu().numpy()
                mano_data['object']['transl'][:] = obj_trans.detach().cpu().numpy()
                mano_data['right']['global_orient'][:] = gaussians.global_orient.detach().cpu().numpy()
                mano_data['right']['betas'][:] = gaussians.betas[0].detach().cpu().numpy()
                mano_data['right']['hand_pose'][:] = gaussians.hand_pose.detach().cpu().numpy()
                mano_data['right']['transl'][:] = gaussians.transl.detach().cpu().numpy()
                mano_data['right']['scale'] = gaussians.hand_scale.detach().cpu().numpy()
                mano_path = os.path.join(point_cloud_path, 'hold_opt_ho.npy')
                np.save(mano_path, mano_data)

                obj_gaussians.save_ply(os.path.join(point_cloud_path, "obj_point_cloud.ply"))

                path_obj_one_hot = os.path.join(point_cloud_path, 'obj_one_hot.pth')
                torch.save(obj_one_hot, path_obj_one_hot)

                path_hand_one_hot = os.path.join(point_cloud_path, 'hand_one_hot.pth')
                torch.save(hand_one_hot, path_hand_one_hot)

                if use_bg_color:
                    path_color = os.path.join(point_cloud_path, 'color.pth')
                    torch.save(scene_brightness, path_color)

                    path_color_dict = os.path.join(point_cloud_path, 'color_dict.pkl')
                    write_pickle(path_color_dict, scene_bright_map)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 7_000, 20_000, 30_000, 40_000, 50_000, 60_000, 90_000,
                                                                           100_000, 120_000, 140_000, 160_000, 180_000,
                                                                           200_000, 300_000, 400_000, 500_000, 600_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000, 7_000, 20_000, 30_000, 40_000, 50_000, 60_000, 90_000,
                                                                           100_000, 120_000, 140_000, 160_000, 180_000,
                                                                           200_000, 300_000, 400_000, 500_000, 600_000])
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

# python3 train_comb.py -s data/ABF12/ -m output/ABF12 --gs_type gs_mesh --meshes 'mesh' --num_splats 1 --iterations 400000 --sh_degree 3 --resolution 1