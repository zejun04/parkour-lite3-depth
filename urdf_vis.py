import os
from pathlib import Path
import tempfile
import mujoco
import mujoco.viewer

# URDF 路径
URDF_PATH = Path("/home/shenlan/RL_gym/extreme-parkour/legged_gym/resources/robots/lite3/urdf/Lite3.urdf")
# 需要加执行器以在 MuJoCo viewer 里出现 joint 滑块
ACTUATED_JOINTS = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]

# 切到 URDF 所在目录，便于加载相对 mesh 路径
os.chdir(URDF_PATH.parent)

# 先从 URDF 生成 MJCF，随后插入 actuators
base_model = mujoco.MjModel.from_xml_path(str(URDF_PATH.name))
tmp_path = Path(tempfile.mkstemp(suffix=".xml")[1])
mujoco.mj_saveLastXML(str(tmp_path), base_model)
xml_text = tmp_path.read_text()

# 构造 actuator 块（position 控制器，kp 可按需调整）
act_lines = ["  <actuator>"]
for j in ACTUATED_JOINTS:
    act_lines.append(f"    <position joint=\"{j}\" kp=\"50\" ctrlrange=\"-2 2\"/>")
act_lines.append("  </actuator>")
act_block = "\n".join(act_lines)

if "<actuator" not in xml_text:
    xml_text = xml_text.replace("</mujoco>", f"{act_block}\n</mujoco>")
else:
    # 若已有 actuator 块，可直接插入在末尾
    xml_text = xml_text.replace("</mujoco>", f"{act_block}\n</mujoco>")

# 重载包含执行器的模型
model = mujoco.MjModel.from_xml_string(xml_text)
data = mujoco.MjData(model)

# 静态展示为主：关闭重力、略加阻尼；如需重力可改为 -9.81
model.opt.gravity[:] = [0, 0, 0]
model.dof_damping[:] = 0.5

# 关闭碰撞以避免接触抖动，可按需注释掉
for i in range(model.ngeom):
    model.geom_contype[i] = 0
    model.geom_conaffinity[i] = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("MuJoCo viewer started. Press ESC to quit.")
    print("Use滑块控制: 按空格显示控制面板 (Panel)，在 Actuators 里拖动各关节")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

# 清理临时文件
try:
    tmp_path.unlink()
except OSError:
    pass
