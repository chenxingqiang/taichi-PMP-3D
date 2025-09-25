#!/bin/bash

# 设置使用GPU 1
export CUDA_VISIBLE_DEVICES=1

# 设置Taichi使用GPU 1
export TI_DEVICE_MEMORY_FRACTION=0.8
export TI_DEVICE_MEMORY_GB=20

echo "配置使用GPU 1进行仿真..."
echo "GPU 1 内存使用情况:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | grep "^1,"

echo ""
echo "开始运行两相MPM仿真..."
python complete_simulation.py
