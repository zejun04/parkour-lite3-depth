# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch
from scipy.spatial.transform import Rotation as R

def euler_from_quaternion(quat):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    if len(quat.shape) == 1:
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    else:
        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]
        w = quat[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def test_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 10)

    # 固定基座
    env_cfg.asset.fix_base_link = True

    # 设置更高的初始高度
    # env_cfg.init_state.pos[2] = 1.0

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    print("task is ", args.task)
    print()

    # 调试信息：打印DOF名称和默认角度
    print("DOF names:", env.dof_names)
    print("Default DOF pos:", env.default_dof_pos)
    print("Current DOF pos:", env.dof_pos[0])
    print("Root state (pos, quat):", env.root_states[0, :3], env.root_states[0, 3:7])
    print()

    

    # 打印每个关节的名称、限制、默认值和当前值
    for i, name in enumerate(env.dof_names):
        print(f"{i}: {name:20s} | limits: [{env.dof_pos_limits[i, 0]:.3f}, {env.dof_pos_limits[i, 1]:.3f}] | default: {env.default_dof_pos[0, i]:.3f} | current: {env.dof_pos[0, i]:.3f}")
    print()

    for i in range(int(10*env.max_episode_length)):
        # 设置 action 为 0，使关节保持在默认位置
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

        # 计算并打印机器人姿态的欧拉角（横滚角和俯仰角）
        root_quat = env.root_states[0, 3:7]
        roll, pitch, yaw = euler_from_quaternion(root_quat.unsqueeze(0))
        print(f"Roll: {roll.item():.3f}, Pitch: {pitch.item():.3f}, Yaw: {yaw.item():.3f}")
        

        obs, _, rew, done, info = env.step(actions)
    print("Done")

if __name__ == '__main__':
    args = get_args()
    test_env(args)
