#!/usr/bin/env python3
"""
MuJoCo XML 可视化脚本 - 用于可视化 MJCF/XML 格式的机器人模型
直接加载 Lite3.xml 文件并在 MuJoCo viewer 中显示
"""

import os
import sys
from pathlib import Path
import mujoco
import mujoco.viewer

# XML 文件路径 (使用绝对路径)
XML_PATH = Path("/home/shenlan/RL_gym/extreme-parkour/legged_gym/resources/robots/lite3_mjcf/mjcf/Lite3.xml")

if not XML_PATH.exists():
    print(f"Error: XML file not found: {XML_PATH}")
    sys.exit(1)

print(f"Loading MuJoCo model from: {XML_PATH}")

# 切到 XML 所在目录，便于加载相对 mesh 路径
os.chdir(XML_PATH.parent)

# 直接加载 MuJoCo XML 文件
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

print(f"Model loaded successfully")
print(f"  - Number of bodies: {model.nbody}")
print(f"  - Number of joints: {model.njnt}")
print(f"  - Number of actuators: {model.nu}")
print(f"  - Number of geoms: {model.ngeom}")
print(f"  - Number of meshes: {model.nmesh}")

# 打印关节名称
print("\nJoint names:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name:
        qpos_adr = model.jnt_qposadr[i]
        range_low = model.jnt_range[i, 0] if model.jnt_limited[i] else -3.14159
        range_high = model.jnt_range[i, 1] if model.jnt_limited[i] else 3.14159
        joint_type = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        print(f"  {i:2d}. {name:20s} ({joint_type:6s}) (range: [{range_low:7.3f}, {range_high:7.3f}])")

# 打印执行器名称
print("\nActuator names:")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name:
        print(f"  {i:2d}. {name}")
    else:
        print(f"  {i:2d}. (unnamed)")

# 设置仿真参数
model.opt.timestep = 0.002  # 500Hz 控制频率
model.opt.gravity[:] = [0, 0, -9.81]

# 添加阻尼以保持稳定
model.dof_damping[:] = 0.5

# 可选：关闭碰撞（如果只想看模型结构，可以取消注释以下几行）
# for i in range(model.ngeom):
#     model.geom_contype[i] = 0
#     model.geom_conaffinity[i] = 0

# 启动 MuJoCo viewer
print("\n" + "="*60)
print("MuJoCo Viewer Started")
print("="*60)
print("Controls:")
print("  - ESC: Quit")
print("  - Space: Show/hide control panel")
print("  - In control panel, go to 'Actuators' to manually control joints")
print("="*60 + "\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

print("\nViewer closed.")
