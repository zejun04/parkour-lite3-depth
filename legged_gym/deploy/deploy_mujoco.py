import os
# 必须放在所有 import 之前
os.environ['MUJOCO_GL'] = 'glfw'

import time
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
        
        # 总观测维度
        self.total_obs_dim = (self.n_proprio + self.n_scan + 
                              self.n_priv_explicit + self.n_priv_latent + 
                              self.history_len * self.n_proprio)
        
        # 历史缓冲区
        self.history_buffer = np.zeros((self.history_len, self.n_proprio))

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
        for i in range(500):
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
                        gravity = get_gravity_orientation(quat)

                        # 提取四元数分量
                        qw, qx, qy, qz = quat

                        # 计算yaw (偏航角)
                        self.yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
                        # --- 指令 ---
                        cmd_x = 0.2
                        
                        # --- 构造当前帧观测 (53维) ---
                        # 1. 角速度（需要缩放0.25）
                        obs_omega = omega * 0.25

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

                        # 5. 环境类型 (one-hot编码)
                        # 17: parkour  18:hand_stand
                        obs_env1 = np.array([0.0])  # env_class != 17
                        obs_env2 = np.array([0.0])  # env_class == 17

                        # 6. 关节信息 (使用策略顺序)
                        qj_obs = (qj - self.default_angles_policy) * 1.0
                        dqj_obs = dqj * 0.05
                        obs_action = self.action

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
                            qj_obs,                        # 12
                            dqj_obs,                       # 12
                            obs_action,                     # 12
                            obs_contact                     # 4
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
                        
                        # 填充本体感知
                        obs[:self.n_proprio] = current_obs
                        
                        # 填充历史 (flat)
                        hist_flat = self.history_buffer.flatten()
                        hist_start = (self.n_proprio + self.n_scan + 
                                     self.n_priv_explicit + self.n_priv_latent)
                        obs[hist_start:hist_start + hist_flat.shape[0]] = hist_flat
                        
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

                        # 使用 base policy 生成动作
                        action_tensor = self.base_policy.act(obs_tensor, depth_feature)
                        self.action = action_tensor.cpu().numpy().squeeze()
                        clip_actions = self.cfg.get("clip_actions", 1.2) / self.action_scale
                        self.action = np.clip(self.action, -clip_actions, clip_actions)

                        # 应用动作
                        # 计算目标位置
                        self.target_dof_pos = self.action * self.action_scale + self.default_angles_config
                        if counter %500 == 0:
                            print(f"\naction: {self.target_dof_pos}")

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