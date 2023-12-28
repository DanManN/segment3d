import os
import time

import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels  # pip install mesh-to-sdf; https://github.com/marian42/mesh_to_sdf

def mesh_to_tsdf(mesh_path, vol_dim=101, save_path=None):
    mesh = trimesh.load(mesh_path)
    voxels = np.swapaxes(mesh_to_voxels(mesh, vol_dim - 2, pad=True), 0, 2)
    scale = 2 / np.max(mesh.bounding_box.extents)
    voxel_size = 2 / (vol_dim - 1) / scale
    vol_origin = mesh.bounding_box.centroid - voxel_size * (vol_dim - 1) / 2

    tsdf_vol = voxels.reshape(-1)
    weight_vol = np.ones_like(tsdf_vol) * 100
    color_vol = np.ones_like(tsdf_vol)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(mesh_path), "tsdf.npz")

    np.savez_compressed(save_path,
        vol_dim=np.array([vol_dim, vol_dim, vol_dim], dtype=np.int32),
        vol_origin=np.ascontiguousarray(vol_origin, dtype=np.float32),
        voxel_size=voxel_size,
        trunc_margin=0.015,
        tsdf_vol=tsdf_vol,
        weight_vol=weight_vol,
        color_vol=color_vol
    )        
    print(f"tsdf volume has been saved to: {save_path}")
    return save_path


if __name__ == '__main__':
    print("Start generating TSDF from mesh model. This may take a few minutes...")
    models = "/home/lsy/dataset/YCB_Video_Dataset/models_16k/"   # CHANGE THIS PATH
    objects = ['001_pringles']  # CHANGE THIS PATH
    for obj in objects:
        print(f"start processing {obj}")
        tic = time.time()
        mesh_path = os.path.join(models, obj, "google_16k", "textured.obj")  # CHANGE THIS PATH
        obj_tsdf_path = mesh_to_tsdf(mesh_path, vol_dim=101)
        print(f"Convert mesh to TSDF in {time.time() - tic}s.")
