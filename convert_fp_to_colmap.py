import os
import json
import numpy as np
import pycolmap
import shutil
import trimesh

from scene.dataset_readers import storePly

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def symlink(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)

def run(input_dir, mesh_path, output_dir):
    image_dir = os.path.join(input_dir, 'rgb')
    masks_dir = os.path.join(input_dir, 'mask')

    col_image_dir = os.path.join(output_dir, 'images')
    col_masks_dir = os.path.join(output_dir, 'masks')

    symlink(os.path.abspath(image_dir), os.path.abspath(col_image_dir))
    symlink(os.path.abspath(masks_dir), os.path.abspath(col_masks_dir))

    sparse_dir = os.path.join(output_dir, 'sparse')
    if not os.path.exists(sparse_dir):
        os.mkdir(sparse_dir)

    sparse_dir = os.path.join(sparse_dir, '0')
    if not os.path.exists(sparse_dir):
        os.mkdir(sparse_dir)

    col_mesh_path = os.path.join(sparse_dir, 'mesh.obj')
    shutil.copyfile(mesh_path, col_mesh_path)

    metadata_path = os.path.join(input_dir, 'metadata.json')
    metadata = read_json(metadata_path)

    camera = None
    pc_images = []
    for image_id, entry in enumerate(metadata):
        camera_data = entry["cameras"]
        
        model_view_matrix = np.array(camera_data["ModelViewMatrix"])
        cam_from_world = model_view_matrix[0:3]
        cam_from_world[0:3, 3] /= 1000.0

        image_name = os.path.basename(entry["rgb_image_path"])

        if camera is None:
            fx = camera_data["fx"]
            fy = camera_data["fy"]
            cx = camera_data["cx"]
            cy = camera_data["cy"]

            width = camera_data["ImageSizeX"]
            height = camera_data["ImageSizeY"]

            camera = pycolmap.Camera(
                    model='PINHOLE',
                    width=width,
                    height=height,
                    params=[fx, fy, cx, cy],
                    camera_id=1,
            )

        pc_image = pycolmap.Image(
            image_id=image_id + 1,
            camera_id=camera.camera_id,
            name=image_name,
            cam_from_world=cam_from_world
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

    print('Done')

INPUT_DIR = 'data/mano'
MESH_PATH = 'data/mano/flat_hand.obj'
OUTPUT_DIR = 'data/mano_colmap'

if __name__ == "__main__":
    run(INPUT_DIR, MESH_PATH, OUTPUT_DIR)