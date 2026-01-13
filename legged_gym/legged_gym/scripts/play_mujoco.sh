#!/bin/bash

# MuJoCo环境测试脚本 - 测试IsaacGym训练的distill策略
# 用法: ./play_mujoco.sh [选项]

source ~/anaconda3/bin/activate parkour

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../.."

# 默认参数
TASK="lite3"
EXPTID="parkour_new/lite3-distil"
CHECKPOINT="-1"
TERRAIN="flat"
NUM_STEPS="2000"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --exptid)
            EXPTID="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --terrain)
            TERRAIN="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--task TASK] [--exptid EXPTID] [--checkpoint CHECKPOINT] [--terrain flat|rough|stairs] [--num_steps STEPS]"
            exit 1
            ;;
    esac
done

# 进入脚本目录
cd "$(dirname "$0")"

# 运行Python脚本
python play_mujoco.py \
    --task "$TASK" \
    --exptid "$EXPTID" \
    --checkpoint "$CHECKPOINT" \
    --terrain_type "$TERRAIN" \
    --num_steps "$NUM_STEPS"
