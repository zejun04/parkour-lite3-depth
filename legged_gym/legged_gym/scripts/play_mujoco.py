#!/usr/bin/env python3
"""
MuJoCo环境测试脚本 - 用于测试IsaacGym训练的distill策略
支持在MuJoCo环境中加载地形并运行策略
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import tempfile
import mujoco
import mujoco.viewer
import re
import math

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 必须在导入torch之前导入isaacgym
import isaacgym

# 现在可以安全地导入torch
import torch

# 避免循环导入，只在需要时导入
# from legged_gym.utils import get_args
# from legged_gym.utils import task_registry
# from legged_gym.envs import *


class MuJoCoPolicyRunner:
    """在MuJoCo环境中运行策略"""
    
    def __init__(self, args):
        self.args = args
        
        # 路径设置
        self.urdf_path = Path(f"{args.urdf_root}/urdf/Lite3.urdf")
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        
        # 训练日志路径
        self.log_root = Path(args.log_root) / args.exptid
        if not self.log_root.exists():
            raise FileNotFoundError(f"Log directory not found: {self.log_root}")
        
        # 创建MuJoCo模型
        self.model = self._create_mujoco_model()
        self.data = mujoco.MjData(self.model)
        
        # 加载策略
        self.policy = self._load_policy()
        
        # 观测空间和动作空间配置 (从IsaacGym训练配置中读取)
        self.setup_observation_space()
        
        # 历史缓冲区
        self.history_len = 10
        self.proprio_dim = 45  # 从配置中读取: 3+2+3+4+33 = 45
        
        # 历史观测缓冲区
        self.history_buffer = np.zeros((self.history_len, self.proprio_dim))
        
    def setup_observation_space(self):
        """设置观测空间维度 (基于Lite3配置)"""
        # n_proprio = 3(lin_vel) + 2(ang_vel_xy) + 3(quat) + 4(base_rpy) + 33(dof_pos/vel/feet) = 45
        # n_scan = 132
        # n_priv_latent = 4 + 1 + 12 + 12 = 29
        # n_priv = 9 (3+3+3)
        # history_len * n_proprio = 10 * 45 = 450
        
        self.n_proprio = 45
        self.n_scan = 132
        self.n_priv = 9
        self.n_priv_latent = 29
        self.num_observations = self.n_proprio + self.n_scan + self.history_len * self.n_proprio + self.n_priv_latent + self.n_priv
        self.num_actions = 12
        
    def _create_mujoco_model(self):
        """创建包含执行器的MuJoCo模型"""
        # 切到URDF目录以加载相对路径
        os.chdir(self.urdf_path.parent)
        
        # 从URDF生成MJCF
        base_model = mujoco.MjModel.from_xml_path(self.urdf_path.name)
        tmp_path = Path(tempfile.mkstemp(suffix=".xml")[1])
        mujoco.mj_saveLastXML(str(tmp_path), base_model)
        xml_text = tmp_path.read_text()
        
        # 添加执行器
        actuated_joints = [
            "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
            "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
            "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
            "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
        ]
        
        act_lines = ["  <actuator>"]
        for j in actuated_joints:
            act_lines.append(f"    <position joint=\"{j}\" kp=\"50\" kv=\"5\" ctrlrange=\"-2 2\"/>")
        act_lines.append("  </actuator>")
        act_block = "\n".join(act_lines)
        
        if "<actuator" not in xml_text:
            xml_text = xml_text.replace("</mujoco>", f"{act_block}\n</mujoco>")
        
        # 加载地形 (可选)
        if self.args.terrain_type != "flat":
            xml_text = self._add_terrain_to_xml(xml_text)
        
        model = mujoco.MjModel.from_xml_string(xml_text)
        
        # 设置仿真参数
        model.opt.timestep = 0.005  # 与IsaacGym一致
        model.opt.gravity[:] = [0, 0, -9.81]
        
        tmp_path.unlink()
        return model
    
    def _add_terrain_to_xml(self, xml_text):
        """添加地形到MuJoCo XML"""
        terrain_xml = """
  <worldbody>
    <geom name="terrain" type="plane" pos="0 0 0" size="100 100 0.1" material="geom_mat"/>
  </worldbody>
  <asset>
    <material name="geom_mat" reflectance="0.5" specular="0.5" shininess="0.5"/>
  </asset>
"""
        
        # 支持不同地形类型
        if self.args.terrain_type == "rough":
            # 添加随机凸起
            terrain_xml = """
  <worldbody>
    <geom name="terrain" type="plane" pos="0 0 0" size="100 100 0.1" material="geom_mat"/>
    <!-- 添加一些随机障碍物 -->
    <geom name="obstacle1" type="box" pos="2 0 0.05" size="0.5 2 0.05" material="geom_mat"/>
    <geom name="obstacle2" type="box" pos="4 1 0.1" size="0.5 2 0.1" material="geom_mat"/>
    <geom name="obstacle3" type="box" pos="6 -1 0.15" size="0.5 2 0.15" material="geom_mat"/>
  </worldbody>
  <asset>
    <material name="geom_mat" reflectance="0.5" specular="0.5" shininess="0.5"/>
  </asset>
"""
        elif self.args.terrain_type == "stairs":
            # 添加阶梯
            terrain_xml = """
  <worldbody>
    <geom name="base" type="plane" pos="0 0 0" size="2 2 0.1" material="geom_mat"/>
    <geom name="step1" type="box" pos="2.5 0 0.05" size="0.5 2 0.05" material="geom_mat"/>
    <geom name="step2" type="box" pos="3.5 0 0.1" size="0.5 2 0.1" material="geom_mat"/>
    <geom name="step3" type="box" pos="4.5 0 0.15" size="0.5 2 0.15" material="geom_mat"/>
    <geom name="step4" type="box" pos="5.5 0 0.2" size="0.5 2 0.2" material="geom_mat"/>
    <geom name="step5" type="box" pos="6.5 0 0.25" size="0.5 2 0.25" material="geom_mat"/>
  </worldbody>
  <asset>
    <material name="geom_mat" reflectance="0.5" specular="0.5" shininess="0.5"/>
  </asset>
"""
        
        if "<worldbody>" in xml_text:
            # 在worldbody中替换或插入地形
            if "<geom name=" in xml_text.split("<worldbody>")[1].split("</worldbody>")[0]:
                # 已有geom，删除旧的
                import re
                xml_text = re.sub(r'<worldbody>.*?<geom.*?type="plane".*?/>.*?</worldbody>', 
                                  f'<worldbody>\n{terrain_xml}', 
                                  xml_text, 
                                  flags=re.DOTALL)
            else:
                # 在worldbody后插入地形
                xml_text = xml_text.replace("<worldbody>", f"<worldbody>\n{terrain_xml}")
        else:
            xml_text = xml_text.replace("</mujoco>", f"{terrain_xml}\n</mujoco>")
        
        return xml_text
    
    def _load_policy(self):
        """加载训练好的策略"""
        # 延迟导入以避免循环依赖
        from legged_gym.utils import task_registry, get_load_path
        
        # 获取模型路径
        model_path, checkpoint = get_load_path(
            self.log_root, 
            checkpoint=self.args.checkpoint
        )
        full_path = os.path.join(self.log_root, model_path)
        
        print(f"Loading policy from: {full_path}")
        print(f"Checkpoint: {checkpoint}")
        
        # 使用task_registry加载模型 (与play.py相同的方式)
        # 创建临时环境来获取配置
        env_cfg, train_cfg = task_registry.get_cfgs(name=self.args.task)
        
        # 创建环境 (小规模, 只用于加载模型)
        env_cfg.env.num_envs = 1
        env, _ = task_registry.make_env(name=self.args.task, args=None, env_cfg=env_cfg)
        
        # 创建算法runner并加载策略
        train_cfg.runner.resume = True
        ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(
            log_root=str(self.log_root),
            env=env,
            name=self.args.task,
            args=None,
            train_cfg=train_cfg,
            return_log_dir=True
        )
        
        # 获取推理策略
        policy = ppo_runner.get_inference_policy(device='cpu')
        
        print(f"Policy loaded successfully")
        print(f"Observation dim: {self.num_observations}")
        print(f"Action dim: {self.num_actions}")
        
        return policy
    
    def get_observation(self):
        """从MuJoCo状态构建观测"""
        # 从MuJoCo数据中提取状态
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        # 线速度 [vx, vy, vz]
        lin_vel = self.data.qvel[:3]
        
        # 角速度 [wx, wy, wz] - 只取xy用于简化
        ang_vel = self.data.qvel[3:5]  # 只取xy
        
        # 姿态 (四元数) [x, y, z, w]
        quat = qpos[3:7]
        
        # 基础姿态 (roll, pitch, yaw)
        import math
        q0, q1, q2, q3 = quat[3], quat[0], quat[1], quat[2]  # w, x, y, z
        roll = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
        pitch = math.asin(2*(q0*q2 - q3*q1))
        yaw = math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))
        base_rpy = np.array([roll, pitch, yaw])
        
        # 关节位置 (12 DOFs)
        joint_pos = qpos[7:19]
        
        # 关节速度 (12 DOFs)
        joint_vel = qvel[6:18]
        
        # 脚接触 (简化: 基于z轴速度或高度)
        feet_contact = np.zeros(4)  # FL, FR, HL, HR
        
        # 基础高度
        base_height = qpos[2]
        
        # 命令 (简化: 固定或随机)
        commands = np.array([0.5, 0.0, 0.0])  # vx, vy, vyaw
        
        # 构建proprioception观测 (45维)
        # 3(lin_vel) + 2(ang_vel_xy) + 3(quat) + 4(base_rpy) + 12(dof_pos) + 12(dof_vel) + 4(feet) + 3(commands) + 2(proj_gravity)
        # 简化版本:
        proprio = np.concatenate([
            lin_vel,
            ang_vel,
            quat,
            base_rpy,
            joint_pos,
            joint_vel,
            feet_contact,
            commands,
            np.array([0.0, 0.0]),  # proj_gravity (简化)
        ])
        
        # 扫描点云 (零填充, 实际需要从传感器获取)
        scan = np.zeros(self.n_scan)
        
        # 私有信息 (零填充 - 在推理时不应该使用)
        priv_latent = np.zeros(self.n_priv_latent)
        priv = np.zeros(self.n_priv)
        
        # 组合观测
        obs = np.concatenate([
            proprio[:self.n_proprio],  # 确保维度正确
            scan,
            self.history_buffer.flatten(),
            priv_latent,
            priv
        ])
        
        return obs
    
    def update_history(self, obs):
        """更新历史缓冲区"""
        proprio = obs[:self.n_proprio]
        self.history_buffer = np.roll(self.history_buffer, -1, axis=0)
        self.history_buffer[-1] = proprio
    
    def reset(self):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始姿态
        default_angles = {
            'FL_HipY_joint': -0.8,
            'FL_Knee_joint': 1.6,
            'FR_HipY_joint': -0.8,
            'FR_Knee_joint': 1.6,
            'HL_HipY_joint': -0.8,
            'HL_Knee_joint': 1.6,
            'HR_HipY_joint': -0.8,
            'HR_Knee_joint': 1.6,
        }
        
        for joint_name, angle in default_angles.items():
            if joint_name in self.model.joint_names:
                jid = self.model.joint_names.index(joint_name)
                qpos_adr = self.model.jnt_qposadr[jid]
                self.data.qpos[qpos_adr] = angle
        
        # 初始位置和姿态
        self.data.qpos[0] = 0.0  # x
        self.data.qpos[1] = 0.0  # y
        self.data.qpos[2] = 0.35  # z (base height)
        
        # 初始四元数 (no rotation)
        self.data.qpos[3] = 0.0  # qx
        self.data.qpos[4] = 0.0  # qy
        self.data.qpos[5] = 0.0  # qz
        self.data.qpos[6] = 1.0  # qw
        
        # 清空速度
        self.data.qvel[:] = 0.0
        
        # 清空历史缓冲区
        self.history_buffer[:] = 0
        
        # 前置步进让机器人稳定
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
    
    def step(self):
        """执行一步仿真"""
        # 获取观测
        obs = self.get_observation()
        
        # 策略推理
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            # 使用hist_encoding=True以使用历史编码
            action = self.policy(obs_tensor, hist_encoding=True)
            if isinstance(action, tuple):
                action = action[0]  # 有时返回 (mean, latent)
            action = action.squeeze(0).numpy()
        
        # 更新历史
        self.update_history(obs)
        
        # 应用动作
        action_scale = 0.25
        
        # 获取关节索引
        joint_names = ["FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
                       "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
                       "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
                       "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"]
        
        joint_indices = []
        for name in joint_names:
            if name in self.model.joint_names:
                jid = self.model.joint_names.index(name)
                qpos_adr = self.model.jnt_qposadr[jid]
                joint_indices.append(qpos_adr)
        
        default_angles = np.array([
            0.0, -0.8, 1.6,  # FL
            0.0, -0.8, 1.6,  # FR
            0.0, -0.8, 1.6,  # HL
            0.0, -0.8, 1.6,  # HR
        ])
        
        # 应用关节角度
        for i, idx in enumerate(joint_indices):
            self.data.qpos[idx] = default_angles[i] + action_scale * action[i]
            self.data.ctrl[i] = self.data.qpos[idx]  # 设置控制目标
        
        # 物理步进 (多次小步以提高稳定性)
        for _ in range(4):  # decimation = 4
            mujoco.mj_step(self.model, self.data)
        
        return obs, action
    
    def run(self, num_steps=1000):
        """运行测试"""
        self.reset()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print(f"Running MuJoCo simulation for {num_steps} steps...")
            print("Press ESC to quit")
            
            step = 0
            while viewer.is_running() and step < num_steps:
                obs, action = self.step()
                viewer.sync()
                
                if step % 100 == 0:
                    print(f"Step {step}: Base height = {self.data.qpos[2]:.3f}, "
                          f"Base vel = [{self.data.qvel[0]:.3f}, {self.data.qvel[1]:.3f}]")
                
                step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='lite3', help='Robot task name')
    parser.add_argument('--exptid', type=str, default='parkour_new/lite3-distil', help='Experiment ID')
    parser.add_argument('--log_root', type=str, default='../../logs/', help='Log directory')
    parser.add_argument('--urdf_root', type=str, default='../resources/robots/lite3', help='URDF directory')
    parser.add_argument('--checkpoint', type=int, default=-1, help='Checkpoint number (-1 for latest)')
    parser.add_argument('--terrain_type', type=str, default='flat', choices=['flat', 'rough', 'stairs'], help='Terrain type')
    parser.add_argument('--num_steps', type=int, default=2000, help='Number of simulation steps')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = Path(__file__).resolve().parent
    if not os.path.isabs(args.log_root):
        args.log_root = os.path.abspath(os.path.join(script_dir, args.log_root))
    
    if not os.path.isabs(args.urdf_root):
        args.urdf_root = os.path.abspath(os.path.join(script_dir, args.urdf_root))
    
    print(f"URDF path: {args.urdf_root}")
    print(f"Log path: {args.log_root}")
    
    runner = MuJoCoPolicyRunner(args)
    runner.run(num_steps=args.num_steps)


if __name__ == '__main__':
    main()
