import pyrender
import numpy as np
import json
import trimesh
import torch
import matplotlib.pyplot as plt
import cv2

from vine_prune.utils import read_json
from vine_prune.utils.mano import get_mano, get_faces

def get_opengl_to_opencv_camera_trans() -> np.ndarray:
    """Returns a transformation from OpenGL to OpenCV camera frame.

    Returns:
        A 4x4 transformation matrix (flipping Y and Z axes).
    """

    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    return yz_flip

torch.set_grad_enabled(False)

SPLAT_JSON_PATH = '/home/hfreeman/harry_ws/repos/gaussian-mesh-splatting/output/mano_colmap/cameras.json'
CAM_IND = 0
splat_data = read_json(SPLAT_JSON_PATH)
splat_cam_data = splat_data[CAM_IND]
R_inv = np.array(splat_cam_data['rotation'])
t_inv = np.array(splat_cam_data['position'])
M_inv = np.eye(4)
M_inv[0:3, 0:3] = R_inv
M_inv[0:3, 3] = t_inv
M = M_inv #np.linalg.inv(M_inv)
trans_c2w = M @ get_opengl_to_opencv_camera_trans()
# trans_c2w = M

# im_size = (640, 480)
# fx = 610.3570557 
# fy = 610.5270996
# cx = 323.1975098
# cy = 245.29776
im_size = (420, 420)
fx = 606.6992797851562 
fy = 606.69921875
cx = 209.75
cy = 209.75
renderer = pyrender.OffscreenRenderer(im_size[0], im_size[1])

ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=ambient_light)

mano = get_mano(flat_hand_mean=False,
                use_pca=True)

betas = torch.zeros(1, 10).float().cuda()
beta_noise = torch.randn_like(betas)
betas += 1.5*beta_noise


# hand_pose = torch.zeros(1, 45).float().cuda()
hand_pose = torch.zeros(1, 6).float().cuda()
hand_pose_noise = torch.randn_like(hand_pose)
hand_pose += 1.5*hand_pose_noise

global_orient = torch.zeros(1, 3).float().cuda()
transl = torch.zeros(1, 3).float().cuda()
mano_out = mano(global_orient=global_orient,
                hand_pose=hand_pose,
                betas=betas,
                transl=transl)

vertices = mano_out.vertices.squeeze(0).cpu().numpy()
faces = get_faces()
trimesh_model = trimesh.Trimesh(vertices, faces, process=False)
mesh = pyrender.Mesh.from_trimesh(trimesh_model)
mesh_node = scene.add(mesh)

camera = pyrender.IntrinsicsCamera(fx=fx,
                                   fy=fy,
                                   cx=cx,
                                   cy=cy,
                                   znear=0.1,
                                   zfar=3000.0)

camera_node = pyrender.Node(camera=camera, matrix=trans_c2w)
scene.add_node(camera_node)

light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=2.4,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
light_node = pyrender.Node(light=light, matrix=trans_c2w)
scene.add_node(light_node)

render_flags = pyrender.constants.RenderFlags.NONE
color, _ = renderer.render(scene, flags=render_flags)

scene.remove_node(camera_node)
scene.remove_node(light_node)
scene.remove_node(mesh_node)

plt.imshow(color)
plt.show()
# vis_im = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
# cv2.imshow('test', vis_im)
# cv2.waitKey(0)
breakpoint()


