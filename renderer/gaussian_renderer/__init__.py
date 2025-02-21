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

import torch
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import trimesh

from utils.graphics_utils import focal2fov
from scene.cameras import Camera
import pytorch3d.transforms

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def my_render(gaussians, pipeline, background, intrinsics, dims, R, T,
              vertices=None, view_R=None, one_hot_labels=None):
    fx, fy, cx, cy = intrinsics
    image_height, image_width = dims

    FoVx = focal2fov(fx, image_width)
    FoVy = focal2fov(fy, image_height)

    cx = (cx - image_width / 2) / image_width * 2
    cy = (cy - image_height / 2) / image_height * 2

    dummy_image = torch.ones(3, image_height, image_width).float().to('cuda')
    
    cam = Camera(colmap_id=None, R=R.detach().cpu().numpy(), T=T.detach().cpu().numpy(), 
                FoVx=FoVx, FoVy=FoVy, 
                cx=cx, cy=cy,
                image=dummy_image, 
                gt_alpha_mask=None,
                image_name=None, uid=None,
                )
        
    res_pkg = render(cam, gaussians, pipeline, background,
                     vertices=vertices, view_R=view_R, one_hot_labels=one_hot_labels)
    return res_pkg

def render(viewpoint_camera, pc : GaussianModel, pipe, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           override_color = None,
           vertices = None,
           view_R = None,
           one_hot_labels=None,
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    viewpoint_camera.camera_center = viewpoint_camera.camera_center
    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(viewpoint_camera.image_height),
    #     image_width=int(viewpoint_camera.image_width),
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=viewpoint_camera.world_view_transform,
    #     projmatrix=viewpoint_camera.full_proj_transform,
    #     sh_degree=pc.active_sh_degree,
    #     campos=viewpoint_camera.camera_center,
    #     prefiltered=False,
    #     debug=pipe.debug,
    #     antialiasing=pipe.antialiasing
    # )
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        #antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if vertices is None:
        _xyz = pc.get_xyz
        scales = pc.get_scaling
        rotations = pc.get_rotation
    else:
        _xyz, rotations, scales = pc.get_xyz_from_verts(vertices)

    means3D = _xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    #scales = None
    #rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        raise RuntimeError('not supported right now')
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        pass
        # scales = pc.get_scaling
        # rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    features = pc.get_features

    if override_color is None:
        if pipe.convert_SHs_python:
            raise RuntimeError('not supported yet')
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        elif view_R is not None:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

            dir_pp_normalized = torch.einsum('bij,bj->bi', view_R, dir_pp_normalized)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        elif vertices is not None:
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (_xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

            rot_mats = pytorch3d.transforms.quaternion_to_matrix(rotations)
            R_T = rot_mats.permute(0, 2, 1)
            # R_T = rot_mats

            dir_pp_normalized = torch.einsum('bij,bj->bi', R_T, dir_pp_normalized)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color

    # # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii, depth_image = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)

    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii,
    #         "depth": depth_image
    #         }

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3Ds_precomp = cov3D_precomp,
        extra_attrs = one_hot_labels)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, 
            "depth": rendered_depth,
            'norm': rendered_norm,
            'alpha': rendered_alpha,
            'extra': extra,
            'colors_precomp': colors_precomp}
