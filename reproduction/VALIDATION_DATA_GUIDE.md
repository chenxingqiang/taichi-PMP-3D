# 验证数据获取指南

## 概述

本指南说明如何从实际仿真中提取验证所需的数据，并生成符合VALID_FIG.md要求的验证图表。

## 数据获取方法

### 方法1：完整验证工作流程（推荐）

运行完整的验证工作流程，包括仿真、数据提取和图表生成：

```bash
cd /Users/xingqiangchen/taichi-MPM-3D/reproduction
python run_validation_workflow.py
```

选择选项1运行完整验证，或选项2运行快速验证（2秒仿真）。

### 方法2：分步执行

#### 步骤1：运行仿真并提取数据

```bash
python data_extractor.py
```

这将：
- 运行完整的仿真
- 在关键时间点提取数据
- 保存时间序列数据
- 保存关键时刻的流动轮廓和速度场

#### 步骤2：生成验证图表

```bash
python real_validation_plots.py
```

使用提取的真实数据生成验证图表。

### 方法3：从现有仿真结果提取

如果你已经有仿真结果，可以：

```python
from data_extractor import SimulationDataExtractor
from real_validation_plots import generate_real_validation_plots

# 从现有仿真提取数据
extractor = SimulationDataExtractor()
validation_data = extractor.extract_from_simulation(your_simulation)
data_dir = extractor.save_validation_data(validation_data)

# 生成验证图表
plot_dir = generate_real_validation_plots(data_dir)
```

## 提取的数据类型

### 1. 时间序列数据 (`time_series_data.csv`)

包含以下列：
- `time`: 时间 (s)
- `flow_front`: 流动前沿位置 (m)
- `flow_height`: 流动高度 (m)
- `mean_velocity`: 平均速度 (m/s)
- `max_velocity`: 最大速度 (m/s)
- `total_impact`: 总冲击力 (N)
- `max_impact`: 最大冲击力 (N)

### 2. 关键时刻流动轮廓 (`flow_profile_*.csv`)

在关键时间点（t=0.0s, 0.2s, 0.4s, 2.0s）的流动轮廓：
- `x`: 归一化渠道长度
- `y`: 归一化高程

### 3. 速度场数据 (`velocity_field_*.npz`)

关键时刻的速度场：
- `x`: x坐标网格
- `y`: y坐标网格
- `u`: x方向速度分量
- `v`: y方向速度分量

## 生成的验证图表

### 1. 流动形态时序对比图
- **文件**: `1_real_flow_morphology_comparison.png`
- **内容**: 4个时间点的流动轮廓对比
- **数据来源**: 真实仿真结果 vs 理论预测

### 2. 冲击力时程曲线
- **文件**: `2_real_impact_force_time_series.png`
- **内容**: 冲击力随时间的变化
- **数据来源**: 真实仿真结果 vs 理论预测

### 3. 速度场分析
- **文件**: `3_real_velocity_field.png`
- **内容**: 关键时刻的速度场云图和流线
- **数据来源**: 真实仿真结果

### 4. 流动统计
- **文件**: `4_flow_statistics.png`
- **内容**: 流动前沿、高度、速度的时程变化
- **数据来源**: 真实仿真结果

## 配置参数

### 关键时间点设置

在 `data_extractor.py` 中修改关键时间点：

```python
# 关键时间点用于验证
self.key_times = [0.0, 0.2, 0.4, 2.0]  # 从VALID_FIG.md
```

### 数据提取间隔

在 `physics_config.yaml` 中设置：

```yaml
numerics:
  statistics_interval: 0.01    # 统计输出间隔 (s)
  vtk_output_interval: 0.1     # VTK输出间隔 (s)
```

## 数据质量检查

### 检查数据完整性

```python
import pandas as pd
import numpy as np

# 加载时间序列数据
df = pd.read_csv('time_series_data.csv')

# 检查数据质量
print(f"数据点数: {len(df)}")
print(f"时间范围: {df['time'].min():.2f} - {df['time'].max():.2f} s")
print(f"最大冲击力: {df['max_impact'].max():.2f} N")
print(f"最大速度: {df['max_velocity'].max():.2f} m/s")

# 检查是否有NaN值
print(f"NaN值数量: {df.isnull().sum().sum()}")
```

### 检查关键时刻数据

```python
# 检查关键时刻的流动轮廓
key_times = ['t_0.0s', 't_0.2s', 't_0.4s', 't_2.0s']
for key_time in key_times:
    profile_file = f'flow_profile_{key_time}.csv'
    if os.path.exists(profile_file):
        profile = pd.read_csv(profile_file)
        print(f"{key_time}: {len(profile)} 个轮廓点")
    else:
        print(f"{key_time}: 数据缺失")
```

## 故障排除

### 常见问题

1. **PCG求解器不收敛**
   - 减小时间步长
   - 放宽求解器容差
   - 检查数值参数设置

2. **GPU内存不足**
   - 使用GPU 1（第二张卡）
   - 减小网格尺寸
   - 减少粒子数量

3. **数据提取失败**
   - 检查仿真是否正常完成
   - 验证输出目录权限
   - 检查文件路径设置

### 调试模式

启用详细输出：

```python
# 在data_extractor.py中设置
DEBUG = True

if DEBUG:
    print(f"提取时间点 {current_time:.3f}s 的数据...")
    print(f"粒子数量: {len(positions)}")
    print(f"PCG迭代次数: {pcg_iterations}")
```

## 性能优化

### 快速验证

对于快速测试，使用2秒仿真：

```bash
python run_validation_workflow.py
# 选择选项2：快速验证
```

### 大规模仿真

对于完整验证，建议：
- 使用GPU加速
- 设置合适的数据提取间隔
- 监控内存使用情况

## 结果验证

### 与论文对比

生成的验证图表应该与Ng et al. (2023)中的结果进行对比：

1. **流动形态**: 检查涌浪角度和流动前沿
2. **冲击力**: 对比峰值力和时程特征
3. **速度场**: 验证流线模式和速度分布
4. **统计指标**: 对比关键参数值

### 质量评估

- 数据连续性检查
- 物理合理性验证
- 数值稳定性确认
- 收敛性分析

## 总结

通过以上方法，你可以：

1. ✅ 从实际仿真中提取验证数据
2. ✅ 生成符合VALID_FIG.md要求的图表
3. ✅ 进行数据质量检查
4. ✅ 与论文结果进行对比验证

你的两相MPM模型现在完全具备生成真实验证数据的能力！
