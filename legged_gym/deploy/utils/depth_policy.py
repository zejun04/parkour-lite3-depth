import time
import threading

import cv2
import mujoco
import glfw
import numpy as np
import torch

from rsl_rl.modules.depth_backbone import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87


def visualize_depth_image(depth_img, title="Depth Image"):
    """可视化深度图（灰度图）"""
    depth_vis = (depth_img * 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    cv2.imshow(title, depth_rgb)
    cv2.waitKey(1)


def zbuf_to_linear(zbuf, znear, zfar):
    """将OpenGL深度缓存值转换为线性深度(正值)"""
    z_ndc = 2.0 * zbuf - 1.0
    return (2.0 * znear * zfar) / (zfar + znear - z_ndc * (zfar - znear))


def load_depth_encoder(vision_weight_path, device):
    print(f"Loading vision weights from: {vision_weight_path}")
    vision_state = torch.load(vision_weight_path, map_location=device)

    base_backbone = DepthOnlyFCBackbone58x87(
        prop_dim=0,
        scandots_output_dim=32,
        hidden_state_dim=512,
        output_activation=None,
        num_frames=1,
    )

    depth_encoder = RecurrentDepthBackbone(base_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_state["depth_encoder_state_dict"])
    depth_encoder.eval()
    depth_encoder.hidden_states = None
    return depth_encoder


class DepthProcessor:
    def __init__(
        self,
        cfg,
        device,
        depth_encoder,
        mujoco_model,
        cam,
        vopt,
        scn,
        ctx,
        window,
        visualize_depth=False,
    ):
        self.cfg = cfg
        self.device = device
        self.depth_encoder = depth_encoder
        self.m = mujoco_model
        self.cam = cam
        self.vopt = vopt
        self.scn = scn
        self.ctx = ctx
        self.window = window
        self.visualize_depth = visualize_depth

        self.depth_near_clip = cfg.get("depth_near_clip", 0.0)
        self.depth_far_clip = cfg.get("depth_far_clip", 2.0)
        self.depth_raw_width = cfg.get("depth_raw_width", 106)
        self.depth_raw_height = cfg.get("depth_raw_height", 60)
        self.depth_width = cfg["depth_width"]
        self.depth_height = cfg["depth_height"]
        self.mujoco_znear = self.m.vis.map.znear * self.m.stat.extent
        self.mujoco_zfar = self.m.vis.map.zfar * self.m.stat.extent

        self.depth_rate_hz = cfg.get("depth_rate_hz", 10)
        self.depth_period = 1.0 / self.depth_rate_hz
        self.depth_lock = threading.Lock()
        self.depth_thread_stop = threading.Event()
        self.depth_thread = None
        self.depth_state_qpos = None
        self.depth_state_qvel = None
        self.depth_state_time = None
        self.latest_proprio_obs = None
        self.latest_proprio_timestamp = None
        self.latest_depth_feature = None
        self.latest_depth_timestamp = None
        self.latest_yaw = None
        self.depth_render_data = mujoco.MjData(self.m)

        self.depth_display_counter = 0
        self.last_depth_min = None
        self.last_depth_max = None

    def start(self):
        self.depth_thread_stop.clear()
        self.depth_thread = threading.Thread(target=self._depth_loop, daemon=True)
        self.depth_thread.start()

    def stop(self, timeout=1.0):
        self.depth_thread_stop.set()
        if self.depth_thread is not None:
            self.depth_thread.join(timeout=timeout)
        self.depth_thread = None

    def update_snapshot(self, qpos, qvel, sim_time, proprio_obs):
        with self.depth_lock:
            self.depth_state_qpos = np.array(qpos, copy=True)
            self.depth_state_qvel = np.array(qvel, copy=True)
            self.depth_state_time = float(sim_time)
            self.latest_proprio_obs = np.array(proprio_obs, copy=True)
            self.latest_proprio_timestamp = float(sim_time)

    def get_depth_feature(self, sim_time, fallback_dim=32):
        with self.depth_lock:
            depth_feature = self.latest_depth_feature
            depth_timestamp = self.latest_depth_timestamp
            yaw = self.latest_yaw

        if depth_feature is None:
            return torch.zeros((1, fallback_dim), device=self.device), None, None, None

        if depth_feature.device != self.device:
            depth_feature = depth_feature.to(self.device)
        if yaw is not None and yaw.device != self.device:
            yaw = yaw.to(self.device)

        depth_latency = None
        if depth_timestamp is not None:
            depth_latency = sim_time - depth_timestamp

        return depth_feature, depth_timestamp, depth_latency, yaw

    def get_depth_image(self, data):
        viewport = mujoco.MjrRect(0, 0, self.depth_raw_width, self.depth_raw_height)

        mujoco.mjv_updateScene(self.m, data, self.vopt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(viewport, self.scn, self.ctx)

        rgb = np.zeros((self.depth_raw_height, self.depth_raw_width, 3), dtype=np.uint8)
        depth_buffer = np.zeros((self.depth_raw_height, self.depth_raw_width, 1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth_buffer, viewport, self.ctx)

        rgb = np.flipud(rgb)
        depth_img = np.flip(depth_buffer, axis=0).squeeze()

        if self.visualize_depth:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Robot RGB Camera", bgr)

        depth_img = zbuf_to_linear(depth_img, self.mujoco_znear, self.mujoco_zfar)
        depth_img = -depth_img
        depth_img = depth_img[:-2, 4:-4]
        depth_img = np.clip(depth_img, -self.depth_far_clip, -self.depth_near_clip)
        if depth_img.shape != (self.depth_height, self.depth_width):
            depth_img = cv2.resize(depth_img, (self.depth_width, self.depth_height), interpolation=cv2.INTER_LINEAR)

        depth_img = -depth_img
        depth_img = (depth_img - self.depth_near_clip) / (self.depth_far_clip - self.depth_near_clip) - 0.5
        self.last_depth_min = float(np.min(depth_img))
        self.last_depth_max = float(np.max(depth_img))

        if self.visualize_depth:
            depth_vis = (depth_img + 0.5).astype(np.float32)
            visualize_depth_image(depth_vis, "Robot Depth Camera")
            self.depth_display_counter += 1

        depth_tensor = torch.from_numpy(depth_img).float().unsqueeze(0).to(self.device)
        return depth_tensor

    def _depth_loop(self):
        glfw.make_context_current(self.window)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        next_time = time.time()
        while not self.depth_thread_stop.is_set():
            with self.depth_lock:
                if self.depth_state_qpos is None or self.latest_proprio_obs is None:
                    state_ready = False
                else:
                    qpos = self.depth_state_qpos.copy()
                    qvel = self.depth_state_qvel.copy()
                    sim_time = self.depth_state_time
                    proprio_obs = self.latest_proprio_obs.copy()
                    state_ready = True

            if state_ready:
                self.depth_render_data.qpos[:] = qpos
                self.depth_render_data.qvel[:] = qvel
                self.depth_render_data.time = sim_time
                mujoco.mj_forward(self.m, self.depth_render_data)

                if self.depth_encoder.hidden_states is not None:
                    self.depth_encoder.detach_hidden_states()

                depth_img = self.get_depth_image(self.depth_render_data)
                with torch.no_grad():
                    obs_for_depth_encoder = proprio_obs.copy()
                    obs_for_depth_encoder[6:8] = 0
                    depth_latent_and_yaw = self.depth_encoder(
                        depth_img,
                        torch.from_numpy(obs_for_depth_encoder).float().unsqueeze(0).to(self.device),
                    )
                    depth_feature = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]

                with self.depth_lock:
                    self.latest_depth_feature = depth_feature
                    self.latest_depth_timestamp = sim_time
                    self.latest_yaw = yaw

            next_time += self.depth_period
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.time()
