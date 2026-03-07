import os
# 必须放在所有 import 之前
os.environ['MUJOCO_GL'] = 'glfw'

import time
import threading
import termios
import tty
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
import sys
import glfw
from pathlib import Path

# 添加 rsl_rl 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "rsl_rl"))
from utils.base_policy import load_base_policy
from utils.depth_policy import load_depth_encoder, DepthProcessor

from legged_gym import LEGGED_GYM_ROOT_DIR

def get_gravity_orientation(quaternion):
    """将四元数转换为重力投影向量"""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def quat_rotate_inverse(quat, vec):
    """将世界坐标向量旋转到机体坐标系 (quat: [qw,qx,qy,qz])"""
    qw, qx, qy, qz = quat
    qvec = np.array([-qx, -qy, -qz])
    t = 2.0 * np.cross(qvec, vec)
    return vec + qw * t + np.cross(qvec, t)

class Lite3VisionDeploy:
    def __init__(self, config_path, debug_obs=False):
        # 1. 安全加载配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        with open(config_path, 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_class = self.cfg.get("env_class", 17)
        self.reorder_dofs = self.cfg.get("reorder_dofs", False)
        self.clip_obs = self.cfg.get("clip_observations", 100.0)
        self.clip_actions = self.cfg.get("clip_actions", 1.2)
        self.lin_vel_scale = self.cfg.get("lin_vel_scale", 2.0)
        self.ang_vel_scale = self.cfg.get("ang_vel_scale", 0.25)
        self.dof_pos_scale = self.cfg.get("dof_pos_scale", 1.0)
        self.dof_vel_scale = self.cfg.get("dof_vel_scale", 0.05)
        self.height_samples_path = self.cfg.get("height_samples_path")
        self.height_samples = None
        self.border_size = self.cfg.get("border_size", 5.0)
        self.horizontal_scale = self.cfg.get("horizontal_scale", 0.1)
        self.vertical_scale = self.cfg.get("vertical_scale", 0.005)

        # 2. 加载 Base Model (JIT 模型)
        base_jit_path = self.cfg["base_jit_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        self.base_policy = load_base_policy(base_jit_path, self.device)

        # 3. 加载 Vision Model (深度编码器)
        vision_weight_path = self.cfg["vision_weight_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        self.depth_encoder = load_depth_encoder(vision_weight_path, self.device)

        # 4. 加载 MuJoCo的urdf.xml模型
        xml_path = self.cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.cfg["simulation_dt"]


        # 6. 视觉上下文（参考深度-点云图.py的方式）
        # 使用glfw创建离屏渲染窗口
        import glfw
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(self.cfg["depth_width"], self.cfg["depth_height"], "Offscreen", None, None)
        glfw.make_context_current(window)

        self.ctx = mujoco.MjrContext(self.m, mujoco.mjtFontScale.mjFONTSCALE_150)
        # 创建帧缓冲对象并启用离屏渲染
        self.framebuffer = mujoco.MjrRect(0, 0, self.cfg["depth_width"], self.cfg["depth_height"])
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        self.window = window
        glfw.make_context_current(None)

        # 设置相机属性
        camera_name = "depth_camera"
        camera_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.cam = mujoco.MjvCamera()
        if camera_id != -1:
            print(f"Using camera: {camera_name}, id={camera_id}")
            # 使用 FIXED 模式，因为 depth_camera 固定在机器人身上
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id
        else:
            print(f"Camera '{camera_name}' not found, using default camera (id=2)")
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = 2
            print(f"Camera tracking body: TORSO (id={tracking_body_id})")

        self.scn = mujoco.MjvScene(self.m, maxgeom=10000)
        self.vopt = mujoco.MjvOption()

        self.action_scale = self.cfg.get("action_scale", 0.25)

        # 7. 控制与状态初始化
        self.num_actions = self.cfg["num_actions"]
        self.default_angles_config = np.array(self.cfg["default_angles"])  # 配置文件顺序
        
        self.default_angles_policy = self.default_angles_config
        self.target_dof_pos = np.zeros(12)
        self.action = np.zeros(self.num_actions)
        
        # 7. 观测维度配置
        self.n_proprio = self.cfg["num_obs"]  # 53
        self.n_scan = 132  # laser scan
        self.n_priv_explicit = 9  # 3 + 3 + 3
        self.n_priv_latent = 29  # 4 + 1 + 12 + 12
        self.history_len = 10

        self.last_contacts = np.zeros(4, dtype=bool)

        self.mass_params = np.array(self.cfg.get("mass_params", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.friction_coeff = float(self.cfg.get("friction_coeff", 1.0))
        self.motor_strength_0 = np.full(self.num_actions, self.cfg.get("motor_strength_0", 1.0), dtype=np.float32)
        self.motor_strength_1 = np.full(self.num_actions, self.cfg.get("motor_strength_1", 1.0), dtype=np.float32)

        self.height_points = self._init_height_points()
        if self.height_samples_path:
            height_path = self.height_samples_path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            if os.path.exists(height_path):
                self.height_samples = np.load(height_path)
            else:
                print(f"[WARN] height_samples_path not found: {height_path}")
        
        # 总观测维度
        self.total_obs_dim = (self.n_proprio + self.n_scan + 
                              self.n_priv_explicit + self.n_priv_latent + 
                              self.history_len * self.n_proprio)
        
        # 历史缓冲区
        self.history_buffer = np.zeros((self.history_len, self.n_proprio))
        self.action_history_len = int(self.cfg.get("action_buf_len", 8))
        self.action_history = np.zeros((self.action_history_len, self.num_actions))

        # 深度图可视化控制
        self.visualize_depth = self.cfg.get("visualize_depth", False)
        self.debug_obs = debug_obs or self.cfg.get("debug_obs", False)

        # 深度异步处理器
        self.depth_processor = DepthProcessor(
            cfg=self.cfg,
            device=self.device,
            depth_encoder=self.depth_encoder,
            mujoco_model=self.m,
            cam=self.cam,
            vopt=self.vopt,
            scn=self.scn,
            ctx=self.ctx,
            window=self.window,
            visualize_depth=self.visualize_depth,
        )
        
        # 初始化yaw相关变量
        self.target_yaw = 0.0
        self.yaw = 0.0
        self.next_target_yaw = 0.0
        self.delta_yaw = 0.0
        self.next_delta_yaw = 0.0

        print("\n✓ Model initialization completed successfully!")
        print(f"  - Base policy input dim: {self.total_obs_dim}")
        print(f"  - Depth encoder input: [batch, 58, 87]")
        print(f"  - Action dim: {self.num_actions}")
        print(f"  - Visualize depth: {self.visualize_depth}") 

    def _init_height_points(self):
        measured_points_x = self.cfg.get(
            "measured_points_x",
            [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2],
        )
        measured_points_y = self.cfg.get(
            "measured_points_y",
            [-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
        )
        grid_x, grid_y = np.meshgrid(np.array(measured_points_x), np.array(measured_points_y), indexing="xy")
        points = np.zeros((grid_x.size, 3), dtype=np.float32)
        points[:, 0] = grid_x.flatten()
        points[:, 1] = grid_y.flatten()
        return points

    def _reindex(self, vec):
        if not self.reorder_dofs:
            return vec
        return vec[[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def _reindex_feet(self, vec):
        if not self.reorder_dofs:
            return vec
        return vec[[1, 0, 3, 2]]

    def _get_heights(self, base_pos, quat):
        if self.height_samples is None:
            return np.zeros(self.n_scan, dtype=np.float32)

        qw, qx, qy, qz = quat
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated_xy = (rot @ self.height_points[:, :2].T).T
        points = rotated_xy + base_pos[:2]
        points += self.border_size
        points = (points / self.horizontal_scale).astype(np.int64)
        px = np.clip(points[:, 0], 0, self.height_samples.shape[0] - 2)
        py = np.clip(points[:, 1], 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = np.minimum(heights1, heights2)
        heights = np.minimum(heights, heights3)
        return (heights * self.vertical_scale).astype(np.float32)
        
    def run(self):
        # 初始化机器人状态
        mujoco.mj_resetData(self.m, self.d)
        
        # 设置初始位置和姿态
        initial_height = self.cfg.get("initial_height", 0.35)  # 默认 0.35m
        self.d.qpos[0] = 0.0  # x
        self.d.qpos[1] = 0.0  # y
        self.d.qpos[2] = initial_height  # z (base height)
        self.d.qpos[3] = 1.0  # qw
        self.d.qpos[4] = 0.0  # qx
        self.d.qpos[5] = 0.0  # qy
        self.d.qpos[6] = 0.0  # qz
        
        # 设置初始关节角度
        for i, angle in enumerate(self.default_angles_config):
            self.d.qpos[7 + i] = angle
        
        # 清空速度
        self.d.qvel[:] = 0.0

        # 前置步进让机器人稳定（使用 PD 控制保持默认姿态）
        print("Stabilizing robot...")
        for i in range(50000):
            # PD Control
            kp = np.array(self.cfg["kps"])
            kd = np.array(self.cfg["kds"])

            q = self.d.qpos[7:]
            dq = self.d.qvel[6:]


            tau = kp * (self.default_angles_config - q) - kd * dq
            tau = np.clip(tau, -30, 30)
            self.d.ctrl[:] = tau

            mujoco.mj_step(self.m, self.d)
        print("Robot stabilized.")
        
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            self.depth_processor.start()
            try:
                start_time = time.time()
                counter = 0
                self.policy_enabled = False
                self._listen_keyboard = True
                self._stdin_state = None

                def start_terminal_listener():
                    if not sys.stdin.isatty():
                        print("[WARN] stdin is not a TTY; terminal key control disabled")
                        return

                    stdin_fd = sys.stdin.fileno()
                    self._stdin_state = termios.tcgetattr(stdin_fd)
                    tty.setcbreak(stdin_fd)

                    def _listen():
                        while self._listen_keyboard:
                            ch = sys.stdin.read(1)
                            if not ch:
                                continue
                            if ch.lower() == "p":
                                self.policy_enabled = not self.policy_enabled
                                state = "ON" if self.policy_enabled else "OFF"
                                print(f"[Policy] Toggled {state}")

                    thread = threading.Thread(target=_listen, daemon=True)
                    thread.start()

                start_terminal_listener()
                
                viewer.sync()
                time.sleep(0.1)

                while viewer.is_running() and (time.time() - start_time) < self.cfg["simulation_duration"]:
                    step_start = time.time()

                    if counter % self.cfg["control_decimation"] == 0:
                        # 每秒打印一次状态
                        if counter % 500 == 0:
                            print(f"\nStep {counter}: Base height = {self.d.qpos[2]:.3f}m, "
                                  f"Base vel = [{self.d.qvel[0]:.3f}, {self.d.qvel[1]:.3f}]")

                        # --- 传感器数据获取 ---
                        qj = self.d.qpos[7:]
                        dqj = self.d.qvel[6:]
                        quat = self.d.qpos[3:7]  # MuJoCo四元数: [qw,qx,qy,qz]
                        omega_world = self.d.qvel[3:6]
                        omega = quat_rotate_inverse(quat, omega_world)
                        base_lin_vel = quat_rotate_inverse(quat, self.d.qvel[:3])
                        gravity = get_gravity_orientation(quat)

                        # 提取四元数分量
                        qw, qx, qy, qz = quat

                        # 计算yaw (偏航角)
                        self.yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
                        self.target_yaw = 0.0
                        self.next_target_yaw = 0.0
                        self.delta_yaw = self.target_yaw - self.yaw
                        self.next_delta_yaw = self.next_target_yaw - self.yaw
                        # --- 指令 ---
                        cmd_x = 0.3
                        
                        # --- 构造当前帧观测 (53维) ---
                        # 1. 角速度（需要缩放0.25）
                        obs_omega = omega * self.ang_vel_scale

                        # 2. IMU (roll, pitch) - 需要从四元数计算
                        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
                        pitch = np.arcsin(2*(qw*qy - qz*qx))
                        obs_imu = np.array([roll, pitch])

                        # 3. Delta yaw (需要真实值，不能全零)
                        obs_delta_yaw_masked = np.zeros(1)  # 屏蔽版本
                        obs_delta_yaw = np.array([self.delta_yaw])  # 真实值
                        obs_delta_next_yaw = np.array([self.next_delta_yaw])  # 下一目标

                        # 4. 命令
                        obs_cmd_masked = np.zeros(2)  # 屏蔽版本 [vx, vy]
                        obs_cmd_unmasked = np.array([cmd_x])  # 真实vx命令

                        # 
                        obs_env1 = np.array([0.0])  # env_class != 17
                        obs_env2 = np.array([0.0])  # env_class == 17

                        # 6. 关节信息 (使用策略顺序)
                        qj_obs = (qj - self.default_angles_policy) * self.dof_pos_scale
                        dqj_obs = dqj * self.dof_vel_scale
                        obs_action = self._reindex(self.action_history[-1])

                        # 7. 接触状态 
                        # 从MuJoCo获取4个足端接触状态 (1表示接触，0表示未接触)
                        obs_contact = np.zeros(4)
                        foot_geom_names = ["FL_FOOT_collision", "FR_FOOT_collision", "HL_FOOT_collision", "HR_FOOT_collision"]
                        for i, geom_name in enumerate(foot_geom_names):
                            foot_geom_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                            if foot_geom_id == -1:
                                obs_contact[i] = -0.5
                                continue
                            obs_contact[i] = 0.5 if np.any(self.d.contact.geom1 == foot_geom_id) or np.any(self.d.contact.geom2 == foot_geom_id) else -0.5
                        contact_bool = obs_contact > 0
                        contact_filt = np.logical_or(contact_bool, self.last_contacts)
                        self.last_contacts = contact_bool
                        obs_contact = self._reindex_feet(contact_filt.astype(np.float32) - 0.5)
                        # print("obs_contact:", obs_contact)

                        current_obs = np.concatenate([
                            obs_omega,                    # 3
                            obs_imu,                       # 2
                            obs_delta_yaw_masked,          # 1
                            obs_delta_yaw,                 # 1
                            obs_delta_next_yaw,            # 1
                            obs_cmd_masked,                # 2
                            obs_cmd_unmasked,              # 1
                            obs_env1,                      # 1
                            obs_env2,                      # 1
                            self._reindex(qj_obs),          # 12
                            self._reindex(dqj_obs),         # 12
                            obs_action,                      # 12
                            obs_contact                      # 4
                        ]) # 总计: 53维

                        sim_time = float(self.d.time)
                        self.depth_processor.update_snapshot(self.d.qpos, self.d.qvel, sim_time, current_obs)

                        # --- 更新历史缓冲区 ---
                        self.history_buffer = np.roll(self.history_buffer, -1, axis=0)
                        hist_obs = current_obs.copy()
                        hist_obs[6:8] = 0  # 训练中历史观测对yaw做mask
                        self.history_buffer[-1] = hist_obs

                        
                        
                        # --- 构建完整观测 ---
                        # 格式: [proprio, scan, priv_explicit, priv_latent, history]
                        obs = np.zeros(self.total_obs_dim)
                        base_pos = self.d.qpos[:3]
                        heights = np.clip(base_pos[2] - 0.3 - self._get_heights(base_pos, quat), -1.0, 1.0)
                        priv_explicit = np.concatenate(
                            [base_lin_vel * self.lin_vel_scale, np.zeros(3), np.zeros(3)]
                        )
                        priv_latent = np.concatenate(
                            [
                                self.mass_params,
                                np.array([self.friction_coeff], dtype=np.float32),
                                self.motor_strength_0 - 1.0,
                                self.motor_strength_1 - 1.0,
                            ]
                        )

                        # 填充本体感知
                        obs[:self.n_proprio] = current_obs
                        obs[self.n_proprio:self.n_proprio + self.n_scan] = heights
                        priv_start = self.n_proprio + self.n_scan
                        obs[priv_start:priv_start + self.n_priv_explicit] = priv_explicit
                        latent_start = priv_start + self.n_priv_explicit
                        obs[latent_start:latent_start + self.n_priv_latent] = priv_latent
                        
                        # 填充历史 (flat)
                        hist_flat = self.history_buffer.flatten()
                        hist_start = (self.n_proprio + self.n_scan + 
                                     self.n_priv_explicit + self.n_priv_latent)
                        obs[hist_start:hist_start + hist_flat.shape[0]] = hist_flat
                        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
                        
                        # 转换为 tensor
                        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

                        depth_feature, depth_timestamp, depth_latency, yaw = self.depth_processor.get_depth_feature(sim_time)

                        # 关键：将 depth_encoder 输出的 yaw 乘以 1.5 后更新到 obs 的 6:8 位置
                        # 这与 play.py 中的处理方式一致
                        if yaw is not None:
                            obs_tensor[:, 6:8] = 1.5 * yaw

                        if self.debug_obs and counter % 500 == 0:
                            joint_err = np.linalg.norm(self.d.qpos[7:] - self.default_angles_config)
                            print(f"  obs[:10] = {np.round(current_obs[:10], 4)}")
                            if depth_latency is not None:
                                print(f"  depth_latency = {depth_latency:.3f}s")
                            print(f"  joint_err_norm = {joint_err:.4f}")

                        if self.policy_enabled:
                            # 使用 base policy 生成动作
                            action_tensor = self.base_policy.act(obs_tensor, depth_feature)
                            self.action = action_tensor.cpu().numpy().squeeze()
                            clip_actions = self.cfg.get("clip_actions", 1.2) / self.action_scale
                            self.action = np.clip(self.action, -clip_actions, clip_actions)
                            self.action_history = np.roll(self.action_history, -1, axis=0)
                            self.action_history[-1] = self.action

                            # 应用动作
                            # 计算目标位置
                            self.target_dof_pos = self.action * self.action_scale + self.default_angles_config
                            if counter % 5000 == 0:
                                print(f"\naction: {self.target_dof_pos}")
                        else:
                            self.action = np.zeros(self.num_actions, dtype=np.float32)
                            self.action_history = np.roll(self.action_history, -1, axis=0)
                            self.action_history[-1] = self.action
                            self.target_dof_pos = self.default_angles_config

                    # PD Control
                    kp = np.array(self.cfg["kps"])
                    kd = np.array(self.cfg["kds"])

                    q = self.d.qpos[7:]
                    dq = self.d.qvel[6:]
                    tau = kp* (self.target_dof_pos - q) - kd * dq
                    tau = np.clip(tau, -25, 25)
                    self.d.ctrl[:] = tau
                    
                    mujoco.mj_step(self.m, self.d)
                    counter += 1
                    viewer.sync()
                    
                    time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
            finally:
                self._listen_keyboard = False
                if self._stdin_state is not None:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._stdin_state)
                self.depth_processor.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="lite3_depth.yaml")
    parser.add_argument("--test", action="store_true", help="Test model loading without running simulation")
    parser.add_argument("--debug_obs", action="store_true", help="Print key observation/depth stats")
    args = parser.parse_args()
    
    # 自动处理路径，防止找不到文件
    if os.path.exists(args.config):
        cfg_path = args.config
    else:
        cfg_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/configs/", args.config)
    
    deployer = Lite3VisionDeploy(cfg_path, debug_obs=args.debug_obs)
    deployer.run()

    # 清理glfw资源
    import glfw
    glfw.terminate()
