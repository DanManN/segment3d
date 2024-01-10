import time

import numpy as np
import cv2
import open3d as o3d
from numba import njit, prange
import cupy as cp
from matplotlib import pyplot as plt

from pf_pose_estimation.tsdf_lib import TSDFVolume
from pf_pose_estimation.cuda_kernels import source_module


class ParticleFilter:

    def __init__(self, obj_tsdf_volume: TSDFVolume, num_particles: int = 2048):
        # object model
        self.obj_tsdf_volume = obj_tsdf_volume
        self.obj_surface = obj_tsdf_volume.get_surface_cloud_marching_cubes(voxel_size=0.005)
        self.obj_offset = np.asarray(self.obj_surface.points).mean(0)

        # initialize particle filter
        self.num_particles = num_particles
        self.particles = np.tile(np.eye(4), (self.num_particles, 1, 1)).astype(np.float32)  # (N, 4, 4)
        self.particles = self.jitter(self.particles, 180, 180, 180, 0.05, 0.05, 0.05, init_offset=self.obj_offset)
        self.particle_weights_gpu = cp.zeros(self.num_particles, dtype=np.float32)

        # load cuda kernels
        self._cuda_batch_inlier_metric = source_module.get_function('batchInlierMetric')

    @staticmethod
    @njit(fastmath=True)
    def random_sample_transformations(N, ai, aj, ak, i, j, k):
        T = np.empty((N, 4, 4), np.float32)
        for idx in prange(N):
            ai_rand = np.random.uniform(-ak, ak)  # exchange ai, ak
            aj_rand = np.random.uniform(-aj, aj)
            ak_rand = np.random.uniform(-ai, ai)  # exchange ai, ak
            x_rand = np.random.uniform(-i, i)
            y_rand = np.random.uniform(-j, j)
            z_rand = np.random.uniform(-k, k)

            si, sj, sk = np.sin(ai_rand), np.sin(aj_rand), np.sin(ak_rand)
            ci, cj, ck = np.cos(ai_rand), np.cos(aj_rand), np.cos(ak_rand)
            cc, cs = ci*ck, ci*sk
            sc, ss = si*ck, si*sk

            T[idx, 0, 0] = cj*ck
            T[idx, 0, 1] = sj*sc-cs
            T[idx, 0, 2] = sj*cc+ss
            T[idx, 1, 0] = cj*sk
            T[idx, 1, 1] = sj*ss+cc
            T[idx, 1, 2] = sj*cs-sc
            T[idx, 2, 0] = -sj
            T[idx, 2, 1] = cj*si
            T[idx, 2, 2] = cj*ci
            T[idx, 3, :3] = 0
            T[idx, 3, 3] = 1
            T[idx, 0, 3] = x_rand
            T[idx, 1, 3] = y_rand
            T[idx, 2, 3] = z_rand
        return T

    @staticmethod
    def jitter(particles, ai, aj, ak, i, j, k, init_offset=None):
        """
        Randomly sample N transformation matrices, by randomly rotating 'rzyx' plus translation
        reference: https://github.com/davheld/tf/blob/master/src/tf/transformations.py
        ai, aj, ak (degrees) along x-axis, y-axis, z-axis
        i, j, k (m)
        """
        particles = particles.copy()

        if init_offset is not None:
            particles[:, :3, 3] -= init_offset

        ai = ai * np.pi / 180
        aj = aj * np.pi / 180
        ak = ak * np.pi / 180

        T = ParticleFilter.random_sample_transformations(particles.shape[0], ai, aj, ak, i, j, k)
        particles = T @ particles

        if init_offset is not None:
            particles[:, :3, 3] += init_offset

        return particles

    @staticmethod
    @njit(parallel=True)
    def get_roi_from_mask(mask: np.ndarray):
        H, W = mask.shape
        start_row = H - 1
        start_col = W - 1
        end_row = 0
        end_col = 0
        for i in prange(H):
            for j in prange(W):
                if mask[i, j]:
                    start_row = min(start_row, i)
                    start_col = min(start_col, j)
                    end_row = max(end_row, i)
                    end_col = max(end_col, j)
        return np.array([start_row, start_col, end_row, end_col], dtype=np.int32)

    @staticmethod
    def create_pcd(depth_im: np.ndarray, cam_intr: np.ndarray, color_im: np.ndarray = None,
                   cam_extr: np.ndarray = np.eye(4)):
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.intrinsic_matrix = cam_intr
        depth_im_o3d = o3d.geometry.Image(depth_im)
        if color_im is not None:
            color_im_o3d = o3d.geometry.Image(color_im)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im_o3d, depth_im_o3d,
                depth_scale=1, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic=cam_extr)
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_im_o3d, intrinsic_o3d, extrinsic=cam_extr,
                                                                  depth_scale=1)
        return pcd

    def estimate(self, color_im: np.ndarray, depth_im: np.ndarray, cam_intr: np.ndarray, mask: np.ndarray = None,
                 num_iters: int = 50, visualize: bool = True):

        H, W = depth_im.shape

        if mask is None:
            color_im_bgr = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
            roi = cv2.selectROI("select_roi", color_im_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("select_roi")
            start_col, start_row, roi_w, roi_h = roi
            mask = np.zeros((H, W), dtype=bool)
            mask[start_row:start_row+roi_h+1, start_col:start_col+roi_w+1] = True

        # find the center of current observation as the initial position
        masked_depth_im = depth_im.copy()
        masked_depth_im[~mask] = 0
        obs_pcd = ParticleFilter.create_pcd(masked_depth_im, cam_intr)
        obs_offset = np.asarray(obs_pcd.points).mean(0)

        # get region of interest
        start_row, start_col, end_row, end_col = ParticleFilter.get_roi_from_mask(mask)
        roi_h = end_row - start_row + 1
        roi_w = end_col - start_col + 1

        # cropped depth image
        cropped_depth_im = masked_depth_im[start_row:end_row+1, start_col:end_col+1]

        tic = time.time()
        for idx in range(num_iters):
            # Particle diffusion
            top_thresh = int(0.1 * self.num_particles)  # top 10% of the particles will be kept without diffusion
            top_particles = self.particles[:top_thresh].copy()
            if idx < 0.5 * num_iters:
                self.particles = self.jitter(self.particles, 10, 10, 10, 0.04, 0.04, 0.04, init_offset=self.obj_offset)
            elif idx < 0.3 * num_iters:
                self.particles = self.jitter(self.particles, 2, 2, 2, 0.02, 0.02, 0.02, init_offset=self.obj_offset)
            elif idx < 0.2 * num_iters:
                self.particles = self.jitter(self.particles, 2, 2, 2, 0.01, 0.01, 0.01, init_offset=self.obj_offset)
            else:
                self.particles = self.jitter(self.particles, 1, 1, 1, 0.01, 0.01, 0.01, init_offset=self.obj_offset)
            self.particles[:top_thresh] = top_particles

            # rendering
            shifted_particles = self.particles.copy()
            shifted_particles[:, :3, 3] += obs_offset
            batch_depth_gpu, _ = self.obj_tsdf_volume.batch_ray_casting(roi_w, roi_h, cam_intr,
                np.linalg.inv(shifted_particles), shifted_particles, start_row, start_col,
                self.num_particles, to_host=False)

            # compute weights
            self.compute_weights_inlier_metric(batch_depth_gpu, cropped_depth_im, self.particle_weights_gpu,
                                               inlier_thresh=0.01)

            weights = self.particle_weights_gpu.get()
            sorted_indices = np.argsort(weights)[::-1]  # descending order

            # get maximum likely estimate
            best_weight = weights[sorted_indices[0]]
            best_particle = shifted_particles[sorted_indices[0]].copy()

            # resample particles
            weights_sum = np.sum(weights)
            if np.allclose(weights_sum, 0):
                p = np.ones_like(weights) / len(weights)
            else:
                p = weights / weights_sum

            resampled_indices = np.random.choice(self.num_particles, size=self.num_particles, replace=True, p=p)
            resampled_indices[:top_thresh] = sorted_indices[:top_thresh]
            self.particles = self.particles[resampled_indices]

            if visualize:
                self.visualize_particles(color_im, batch_depth_gpu, sorted_indices, start_row, start_col, top_k=5,
                                         pause=False, text="iteration:" + str(idx).zfill(4), text_color=(0, 0, 255),
                                         window_name="visualization")
        toc = time.time()
        print(f"Perform {num_iters} iterations in {toc - tic:.03f}s")

        if visualize:
            self.visualize_particles(color_im, batch_depth_gpu, sorted_indices, start_row, start_col, top_k=10,
                                     pause=True, text="iteration:" + str(idx).zfill(4), text_color=(0, 0, 255),
                                     window_name='visualization')
            cv2.destroyWindow("visualization")
        return best_particle, best_weight

    def visualize_particles(self, color_im, batch_depth_gpu, sorted_indices, start_row, start_col,
                            top_k=1, pause=False, text=None, text_color=(0, 0, 0), window_name="visualization"):
        color_im = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
        color_im = cv2.putText(color_im, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                               1, text_color, 1, cv2.LINE_AA)
        rendered_im = color_im.copy()

        batch_depth_cpu = batch_depth_gpu.get()[sorted_indices[:top_k]].astype(bool)
        for i in range(1, top_k):
            batch_depth_cpu[0] |= batch_depth_cpu[i]

        h, w = batch_depth_cpu[0].shape
        rendered_depth = np.zeros((h, w, 3), dtype=np.uint8)
        rendered_depth[batch_depth_cpu[0].astype(bool)] = [0, 0, 255]
        rendered_im[start_row:start_row+h, start_col:start_col+w, :] = rendered_depth

        alpha = 0.5
        blended_im = cv2.addWeighted(color_im, alpha, rendered_im, (1 - alpha), 0.0)

        cv2.imshow(window_name, blended_im)
        if pause:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)

    def compute_weights_inlier_metric(self, batch_depth_gpu, depth_im, particle_weights_gpu, inlier_thresh=0.005):
        H, W = depth_im.shape
        self._cuda_batch_inlier_metric(
            (int(np.ceil(self.num_particles / 1024)), 1, 1),
            (1024, 1, 1),
            (
                cp.int32(H),
                cp.int32(W),
                cp.int32(self.num_particles),
                batch_depth_gpu,
                cp.array(depth_im, dtype=cp.float32),
                particle_weights_gpu,
                cp.float32(inlier_thresh),
            )
        )
