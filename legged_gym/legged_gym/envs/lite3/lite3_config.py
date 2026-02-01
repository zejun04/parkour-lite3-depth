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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Lite3RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        reorder_dofs = False
        num_actions = 12

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # 正常朝向
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HipX_joint': 0.0,   # [rad]
            'FL_HipY_joint': -0.8,   # [rad] - URDF中范围[-2.67, 0.314]
            'FL_Knee_joint': 1.6,   # [rad] - URDF中范围[0.524, 2.792]

            'FR_HipX_joint': 0.0,   # [rad]
            'FR_HipY_joint': -0.8,   # [rad]
            'FR_Knee_joint': 1.6,   # [rad]

            'HL_HipX_joint': 0.0,   # [rad]
            'HL_HipY_joint': -0.8,   # [rad]
            'HL_Knee_joint': 1.6,   # [rad]

            'HR_HipX_joint': 0.0,   # [rad]
            'HR_HipY_joint': -0.8,   # [rad]
            'HR_Knee_joint': 1.6,   # [rad]
        }

    class init_state_slope( LeggedRobotCfg.init_state ):
        pos = [0.5, 0.0, 0.35] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # 正常朝向
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HipX_joint': 0.0,   # [rad]
            'FL_HipY_joint': -0.8,   # [rad]
            'FL_Knee_joint': 1.6,   # [rad]

            'FR_HipX_joint': 0.0,   # [rad]
            'FR_HipY_joint': -0.8,   # [rad]
            'FR_Knee_joint': 1.6,   # [rad]

            'HL_HipX_joint': 0.0,   # [rad]
            'HL_HipY_joint': -0.8,   # [rad]
            'HL_Knee_joint': 1.6,   # [rad]

            'HR_HipX_joint': 0.0,   # [rad]
            'HR_HipY_joint': -0.8,   # [rad]
            'HR_Knee_joint': 1.6,   # [rad]
        }
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3/urdf/Lite3.urdf'
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35
        # class scales( LeggedRobotCfg.rewards.scales ):
            # torques = -0.0002
            # dof_pos_limits = -10.0

class Lite3RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_lite3'

  
