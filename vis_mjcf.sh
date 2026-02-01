#!/bin/bash
# MuJoCo XML 可视化启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting MuJoCo visualization for Lite3.xml..."
echo "Press Ctrl+C or ESC to exit"
echo ""

python3 vis_mjcf.py
