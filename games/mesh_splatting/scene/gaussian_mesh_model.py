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

import torch
import numpy as np

from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.sh_utils import RGB2SH
from games.mesh_splatting.utils.graphics_utils import MeshPointCloud

class GaussianMeshModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self._scale = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.vertices = None
        self.faces = None
        self.triangles = None
        self.eps_s0 = 1e-8

    @property
    def get_xyz(self):
        return self._xyz

    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):

        self.point_cloud = pcd
        self.triangles = self.point_cloud.triangles
        self.spatial_lr_scale = spatial_lr_scale
        pcd_alpha_shape = pcd.alpha.shape

        print("Number of faces: ", pcd_alpha_shape[0])
        print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

        alpha_point_cloud = pcd.alpha.float().cuda()
        scale = torch.ones((pcd.points.shape[0], 1)).float().cuda()

        print("Number of points at initialisation : ",
              alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))
        opacities = inverse_sigmoid(0.95 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))
        self.clamp_opacity_inv = opacities[0, 0]

        # self.vertices = nn.Parameter(
        #     self.point_cloud.vertices.clone().detach().requires_grad_(True).cuda().float()
        # )
        self.vertices = nn.Parameter(
            torch.as_tensor(self.point_cloud.vertices).requires_grad_(True).cuda().float()
        )

        self.faces = torch.tensor(self.point_cloud.faces).cuda()

        self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))  # check update_alpha
        self.update_alpha()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scale = nn.Parameter(scale.requires_grad_(True))
        self.prepare_scaling_rot()
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # self.clamp_scale_max = 2.0
        self.clamp_scale_max = 3.0
        self.clamp_scale_min = -100.0

    def set_hand_data(self, mano_data):
        self.global_orient = nn.Parameter(torch.from_numpy(mano_data['right']['global_orient']).float().cuda())
        self.transl = nn.Parameter(torch.from_numpy(mano_data['right']['transl']).float().cuda())
        self.hand_pose = nn.Parameter(torch.from_numpy(mano_data['right']['hand_pose']).float().cuda())
        self.betas = nn.Parameter(torch.from_numpy(mano_data['right']['betas'][0:1]).float().cuda())
        # self.hand_scale = nn.Parameter(torch.from_numpy(mano_data['right']['scale']).float().cuda())
        self.hand_scale = torch.from_numpy(mano_data['right']['scale']).float().cuda()

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        _xyz = torch.matmul(
            self.alpha,
            self.triangles
        )
        self._xyz = _xyz.reshape(
                _xyz.shape[0] * _xyz.shape[1], 3
            )
        
    def prepare_scaling_rot(self):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from
        centroid to 2nd vertex onto subspace spanned by v0 and v1.
        """
        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)
        
        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        triangles = self.triangles
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + self.eps_s0)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        # v1 = triangles[:, 0] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + self.eps_s0
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        # v2_init = triangles[:, 1] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + self.eps_s0)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = self.eps_s0 * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        scales = scales.broadcast_to((*self.alpha.shape[:2], 3))

        scales_to_use = self._scale * scales.flatten(start_dim=0, end_dim=1)
        scales_to_use[:, 0] = self.eps_s0

        self._scaling = torch.log(
            torch.nn.functional.relu(scales_to_use) + self.eps_s0
        )
        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)

    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0
        """
        self.alpha = torch.relu(self._alpha) + 1e-8
        self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)
        self.triangles = self.vertices[self.faces]
        self._calc_xyz()

    def get_xyz_from_verts(self, vertices, activate=True):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0
        """
        alpha = torch.relu(self._alpha) + 1e-8
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)
        triangles = vertices[self.faces]

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(
                xyz.shape[0] * xyz.shape[1], 3
            )
        
        ###
        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)
        
        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u
        
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + self.eps_s0)
        means = torch.mean(triangles, dim=1)
        #TODO WARNING I am not sure if this is the correct logic and I found a bug or other
        # also same for above
        v1 = triangles[:, 1] - means
        # v1 = triangles[:, 0] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + self.eps_s0
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        # v2_init = triangles[:, 1] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + self.eps_s0)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = self.eps_s0 * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        scales = scales.broadcast_to((*self.alpha.shape[:2], 3))
        
        scales_to_use = self._scale * scales.flatten(start_dim=0, end_dim=1)
        scales_to_use[:, 0] = self.eps_s0

        scaling = torch.log(
            torch.nn.functional.relu(scales_to_use) + self.eps_s0
        )

        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation.transpose(-2, -1)
        rotation = rot_to_quat_batch(rotation)
        ###

        if activate:
            scaling = self.scaling_activation(scaling)
            rotation = self.rotation_activation(rotation)

        return xyz, rotation, scaling

    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l_params = [
            #{'params': [self.vertices], 'lr': training_args.vertices_lr, "name": "vertices"},
            {'params': [self._alpha], 'lr': training_args.alpha_lr, "name": "alpha"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scale], 'lr': training_args.scaling_lr, "name": "scaling"},

            {'params': [self.global_orient], 'lr': 1e-3, "name": "global_orient"},
            {'params': [self.transl], 'lr': 1e-3, "name": "transl"},
            # {'params': [self.hand_pose], 'lr': 1e-4, "name": "hand_pose"},
            {'params': [self.hand_pose], 'lr': 5e-4, "name": "hand_pose"},
            {'params': [self.betas], 'lr': 1e-3, "name": "betas"},
            # {'params': [self.hand_scale], 'lr': 1e-3, "name": "hand_scale"}
        ]

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration) -> None:
        """ Learning rate scheduling per step """
        pass

    def save_ply(self, path):
        self.update_alpha()
        self.prepare_scaling_rot()
        self._save_ply(path)

        attrs = self.__dict__
        additional_attrs = [
            '_alpha', 
            '_scale',
            #'point_cloud',
            'triangles',
            'vertices',
            'faces',
            'global_orient',
            'betas',
            'hand_pose',
            'transl',
            'hand_scale',
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            save_dict[attr_name] = attrs[attr_name]

        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)

    def load_ply(self, path):
        self._load_ply(path)
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        params = torch.load(path_model)
        alpha = params['_alpha']
        scale = params['_scale']
        if 'vertices' in params:
            self.vertices = params['vertices']
        if 'triangles' in params:
            self.triangles = params['triangles']
        if 'faces' in params:
            self.faces = params['faces']
        # point_cloud = params['point_cloud']
        self._alpha = nn.Parameter(alpha)
        self._scale = nn.Parameter(scale)

        self.global_orient = nn.Parameter(params['global_orient'])
        self.transl = nn.Parameter(params['transl'])
        self.hand_pose = nn.Parameter(params['hand_pose'])
        self.betas = nn.Parameter(params['betas'])
        # self.hand_scale = nn.Parameter(params['hand_scale'])
        self.hand_scale = params['hand_scale']

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.max(self.get_opacity, torch.ones_like(self.get_opacity)*0.05))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def clamp_opacity(self):
        opacities_new = torch.clamp(self._opacity, min=self.clamp_opacity_inv)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def clamp_scale(self):
        scales_new = torch.clamp(self._scale, min=self.clamp_scale_min, max=self.clamp_scale_max)
        optimizable_tensors = self.replace_tensor_to_optimizer(scales_new, "scaling")
        self._scale = optimizable_tensors["scaling"]
