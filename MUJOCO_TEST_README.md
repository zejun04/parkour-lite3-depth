# MuJoCo环境测试脚本使用说明

## 概述

`play_mujoco.py` 是一个在MuJoCo环境中测试IsaacGym训练策略的脚本，支持测试distill后的lite3策略。

## 文件位置

```
/home/shenlan/RL_gym/extreme-parkour/legged_gym/legged_gym/scripts/play_mujoco.py
```

## 功能特性

- 在MuJoCo环境中加载IsaacGym训练的策略
- 支持完整的策略类型（包含历史编码）
- 支持不同地形类型（flat, rough, stairs）
- 实时可视化仿真结果
- 自动加载训练配置和模型

## 使用方法

### 基本用法

```bash
cd /home/shenlan/RL_gym/extreme-parkour/legged_gym/legged_gym/scripts
python play_mujoco.py
```

### 使用shell脚本

```bash
cd /home/shenlan/RL_gym/extreme-parkour/legged_gym/legged_gym/scripts
./play_mujoco.sh
```

### 指定参数

```bash
python play_mujoco.py \
    --task lite3 \
    --exptid parkour_new/lite3-distil \
    --checkpoint 6000 \
    --terrain_type rough \
    --num_steps 2000
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 机器人任务名称 | `lite3` |
| `--exptid` | 实验ID（相对logs/的路径） | `parkour_new/lite3-distil` |
| `--log_root` | 日志根目录 | `../../logs/` |
| `--urdf_root` | URDF文件目录 | `./resources/robots/lite3` |
| `--checkpoint` | checkpoint编号 (-1表示最新) | `-1` |
| `--terrain_type` | 地形类型 (flat/rough/stairs) | `flat` |
| `--num_steps` | 仿真步数 | `2000` |

## 地形类型

### `flat` (平面地形)
标准的平面地形，适合基础测试。

### `rough` (粗糙地形)
包含随机障碍物：
- 三个不同高度的障碍块
- 障碍高度：0.05m, 0.1m, 0.15m

### `stairs` (阶梯地形)
包含连续的阶梯：
- 5个高度递增的台阶
- 台阶高度从0.05m递增到0.25m

## 策略加载位置

默认从以下位置加载策略：
```
/home/shenlan/RL_gym/extreme-parkour/legged_gym/logs/parkour_new/lite3-distil/
```

可用的checkpoint包括：
- `model_0.pt` 到 `model_6000.pt` (间隔100或200)

## 依赖要求

- Python 3.8+
- PyTorch
- MuJoCo
- NumPy
- legged_gym项目依赖

## 示例运行命令

### 测试最新checkpoint（平面地形）
```bash
python play_mujoco.py
```

### 测试指定checkpoint（粗糙地形）
```bash
python play_mujoco.py --checkpoint 6000 --terrain_type rough
```

### 测试阶梯地形
```bash
python play_mujoco.py --terrain_type stairs
```

### 长时间运行测试
```bash
python play_mujoco.py --num_steps 10000
```

## 注意事项

1. **IsaacGym依赖**：策略加载需要创建一个临时的IsaacGym环境，请确保IsaacGym已正确安装。

2. **URDF路径**：确保URDF文件存在于指定路径，并且mesh文件的相对路径正确。

3. **观测匹配**：MuJoCo中的观测构建需要与IsaacGym训练时的观测定义一致。当前版本进行了简化，某些观测（如扫描点云）使用了零填充。

4. **历史缓冲区**：策略使用历史编码（hist_encoding=True），确保历史缓冲区正确初始化。

5. **控制频率**：MuJoCo仿真步进设置为与IsaacGym一致（dt=0.005s），并使用decimation=4。

## 故障排除

### 问题：URDF文件未找到
```
FileNotFoundError: URDF not found: /path/to/Lite3.urdf
```
解决：检查URDF路径是否正确，使用`--urdf_root`参数指定正确的URDF目录。

### 问题：日志目录未找到
```
FileNotFoundError: Log directory not found: /path/to/logs
```
解决：检查`--exptid`和`--log_root`参数是否正确，确保训练日志存在。

### 问题：checkpoint未找到
```
Error loading checkpoint
```
解决：使用`--checkpoint`参数指定存在的checkpoint编号，或使用`-1`加载最新的。

### 问题：观测维度不匹配
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```
解决：检查`setup_observation_space`中的维度配置是否与训练配置一致。

## 与原有urdf_vis.py的区别

| 特性 | urdf_vis.py | play_mujoco.py |
|------|-------------|----------------|
| 用途 | URDF可视化展示 | 策略测试 |
| 控制方式 | 手动控制 | 自动策略控制 |
| 地形支持 | 无 | flat/rough/stairs |
| 历史编码 | 无 | 有 |
| 策略加载 | 无 | 完整训练策略 |
| 实时推理 | 无 | 有 |

## 未来改进方向

1. 添加完整的扫描点云传感器支持
2. 从IsaacGym导出真实地形数据到MuJoCo
3. 添加更多的评估指标（成功率、行走距离等）
4. 支持多机器人并行测试
5. 添加视频录制功能
