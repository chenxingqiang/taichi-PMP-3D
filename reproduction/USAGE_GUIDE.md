# 使用指南：两相MPM泥石流冲击仿真

## 🚀 快速开始

### 1. 基础仿真运行

```bash
cd /Users/xingqiangchen/taichi-MPM-3D/reproduction
python complete_simulation.py
```

这将运行完整的泥石流冲击仿真，包括：
- 两相MPM求解器
- 屏障接触力学
- 输出量计算
- 结果分析和可视化

### 2. 测试新组件

```bash
python test_new_components.py
```

验证所有新组件的功能是否正常。

## 📋 组件说明

### 1. BarrierModel (屏障模型)

**功能**：
- 双屏障配置与接触检测
- 惩罚法接触力计算（方程1）
- 溢出轨迹跟踪（方程3-4）
- 冲击力统计

**使用示例**：
```python
from barrier_model import BarrierModel

# 创建屏障模型
barrier = BarrierModel(
    barrier_height=0.15,      # 屏障高度 (m)
    barrier_spacing=2.0,      # 屏障间距 (m)
    barrier_positions=(3.0, 5.0),  # 屏障位置 (m)
    contact_stiffness=1e8,    # 接触刚度 (N/m)
    friction_coefficient=0.4  # 摩擦系数
)

# 检测接触
barrier.detect_contacts(particle_positions, particle_velocities, 
                       contact_forces, n_particles)

# 跟踪溢出
barrier.track_overflow_kinematics(particle_positions, particle_velocities,
                                 n_particles, current_time)

# 获取统计结果
impact_stats = barrier.get_impact_statistics()
overflow_stats = barrier.get_overflow_statistics()
```

### 2. OutputMetricsCalculator (输出量计算器)

**功能**：
- 流化率计算（方程26）
- 冲击力分析（方程1）
- 流动统计
- 屏障有效性评估

**使用示例**：
```python
from output_metrics import OutputMetricsCalculator

# 创建计算器
metrics = OutputMetricsCalculator("physics_config.yaml")

# 计算流化率
metrics.compute_fluidization_ratio(pressure_field, stress_field, n_particles)

# 计算冲击力
metrics.compute_impact_forces(velocity_field, depth_field, density_field, n_particles)

# 计算流动统计
metrics.compute_flow_statistics(velocity_field, depth_field, 
                               volume_fraction_field, n_particles)

# 获取所有指标
all_metrics = metrics.export_all_metrics()
metrics.print_metrics_summary()
```

### 3. CompleteDebrisFlowSimulation (完整仿真)

**功能**：
- 完整仿真流程编排
- 屏障间距研究（第4.2节）
- 弗劳德数分析（第4.1节）
- 结果分析和可视化

**使用示例**：
```python
from complete_simulation import CompleteDebrisFlowSimulation

# 创建仿真
simulation = CompleteDebrisFlowSimulation("physics_config.yaml")

# 运行主仿真
results = simulation.run_simulation("output_directory")

# 运行屏障间距研究
study_results = simulation.run_barrier_spacing_study("study_output")

# 打印结果摘要
simulation.print_simulation_summary(results)
```

## 🔧 配置参数

### 物理参数 (physics_config.yaml)

```yaml
materials:
  solid:
    density: 2650.0              # 固体密度 (kg/m³)
    critical_volume_fraction: 0.56  # 临界体积分数
    particle_diameter: 0.001     # 颗粒直径 (m)
    friction_coefficients:
      mu_1: 0.49                 # 摩擦系数1
      mu_2: 1.4                  # 摩擦系数2
  
  fluid:
    density: 1000.0              # 流体密度 (kg/m³)
    dynamic_viscosity: 0.001     # 动力粘度 (Pa·s)

simulation:
  total_time: 5.0                # 总仿真时间 (s)
  barrier_height: 0.15           # 屏障高度 (m)
  barrier_spacing: 2.0           # 屏障间距 (m)
  barrier_positions: [3.0, 5.0]  # 屏障位置 (m)
  slope_angle: 20.0              # 坡度角 (度)
  debris_volume: 500.0           # 泥石流体积 (m³)

numerics:
  max_timestep: 1.0e-4           # 最大时间步长 (s)
  particles_per_cell: 16         # 每网格粒子数
  mesh_barrier_ratio: 0.04       # 网格/屏障高度比
  vtk_output_interval: 100       # VTK输出间隔
```

## 📊 输出结果

### 1. 文件输出

- **VTK文件**：`frame_*.vtk` - 用于ParaView可视化
- **指标文件**：`final_metrics.yaml` - 最终计算结果
- **时间序列**：`time_series_data.csv` - 时间演化数据
- **分析图表**：`analysis_plots.png` - 结果分析图

### 2. 关键指标

- **流化率**：λ = p_bed/(p_bed + σ'_bed)
- **冲击力**：F = αρv²h + (k/2)h²ρ||g||
- **捕获效率**：被屏障捕获的粒子比例
- **溢出率**：溢出屏障的粒子比例
- **能量耗散**：动能耗散比例

### 3. 屏障有效性

- **捕获效率**：屏障捕获粒子的能力
- **溢出轨迹**：粒子溢出后的着陆距离
- **能量耗散**：屏障的能量耗散效果

## 🧪 验证案例

### 1. 基础功能测试

```bash
python test_new_components.py
```

### 2. 完整仿真测试

```bash
python complete_simulation.py
```

### 3. 屏障间距研究

```python
simulation = CompleteDebrisFlowSimulation()
study_results = simulation.run_barrier_spacing_study()
```

## 📈 结果分析

### 1. 流化率分析

流化率λ反映泥石流的流化程度：
- λ = 0：无流化（干颗粒流）
- λ = 1：完全流化（土壤液化）
- 0 < λ < 1：部分流化

### 2. 冲击力分析

冲击力包含两个分量：
- **水动力分量**：αρv²h（与速度平方成正比）
- **静力分量**：(k/2)h²ρ||g||（与深度平方成正比）

### 3. 屏障有效性

- **捕获效率**：衡量屏障阻挡效果
- **溢出轨迹**：分析粒子溢出后的运动
- **能量耗散**：评估屏障的耗能效果

## 🔍 故障排除

### 1. 常见问题

**问题**：仿真不稳定
**解决**：减小时间步长，检查CFL条件

**问题**：接触力过大
**解决**：调整接触刚度参数

**问题**：内存不足
**解决**：减少粒子数量或使用GPU加速

### 2. 性能优化

- 使用GPU加速：`ti.init(arch=ti.cuda)`
- 调整网格分辨率
- 优化输出频率
- 使用并行计算

## 📚 参考文献

Ng, C. W. W., Choi, C. E., Koo, R. C. H., Kwan, J. S. H., & Lam, C. (2023). 
Two-phase Material Point Method for debris flow impact on rigid barriers. 
*Computers and Geotechnics*, 157, 105-120.

## 🆘 技术支持

如有问题，请检查：
1. 配置文件格式是否正确
2. 依赖包是否安装完整
3. 系统资源是否充足
4. 日志文件中的错误信息
