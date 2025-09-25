# Quick Start Guide - iMPM Implementation

## 🚀 Ready to Run

您的iMPM（不可压缩物质点法）实现已经完成！以下是快速测试步骤：

## 📋 前提条件

确保您已安装必要的依赖：

```bash
pip install taichi>=1.0.2 numpy
```

## 🧪 第一步：基础测试

首先运行基础功能测试，确保所有模块正常工作：

```bash
cd /Users/xingqiangchen/taichi-PMP-3D/reproduction
python test_impm.py
```

**期望输出**：所有测试应该通过，显示 "🎉 All tests passed!"

## 🌊 第二步：溃坝算例

运行完整的3D溃坝模拟（论文Section 7.1）：

```bash
cd /Users/xingqiangchen/taichi-PMP-3D/reproduction
python examples/dam_break_3d.py
```

**模拟参数**：
- 计算域：3.22m × 1.0m × 0.6m
- 初始水柱：0.6m × 1.0m × 0.6m  
- 网格分辨率：161 × 50 × 30
- 物理参数：水的密度和粘度
- 运行时间：2秒物理时间

**期望输出**：
- 实时显示波前位置、动能、最大速度
- PCG求解器收敛信息
- 生成VTK文件用于可视化
- CSV数据文件用于验证

## 📊 第三步：查看结果

模拟完成后，您会得到：

1. **VTK文件**：`dam_break_frame_*.vtk` - 用ParaView可视化
2. **CSV数据**：
   - `dam_break_wave_front.csv` - 波前位置vs时间
   - `dam_break_normalized.csv` - 无量纲化数据

3. **控制台输出**：与理论解的对比

## 🔧 自定义模拟

您也可以创建自定义模拟：

```python
from incompressible_mpm_solver import IncompressibleMPMSolver

# 创建求解器
solver = IncompressibleMPMSolver(
    nx=32, ny=32, nz=32,    # 网格尺寸
    dx=0.05,                # 网格间距
    rho=1000.0,            # 密度
    mu=1e-3,               # 粘度
    gamma=0.073,           # 表面张力（可选）
    g=9.8                  # 重力
)

# 初始化粒子
solver.initialize_particles_dam_break(
    x_min=0.0, x_max=0.5,
    y_min=0.0, y_max=0.8, 
    z_min=0.0, z_max=0.5,
    ppc=8
)

# 初始化水平集
solver.level_set_method.initialize_box(0.0, 0.5, 0.0, 0.8, 0.0, 0.5)

# 运行模拟
for step in range(1000):
    iterations = solver.step()
    
    if step % 100 == 0:
        solver.compute_statistics()
        print(f"Step {step}: KE = {solver.total_kinetic_energy[None]:.6f}")
```

## 📈 性能预期

根据您的硬件配置：

- **CPU模式**：约10-50步/秒（小规模测试）
- **GPU模式**：约100-500步/秒（如果有GPU）
- **内存使用**：取决于粒子数量，通常几百MB到几GB

## 🛠️ 故障排除

### 如果遇到导入错误：
```bash
export PYTHONPATH="/Users/xingqiangchen/taichi-PMP-3D/reproduction:$PYTHONPATH"
```

### 如果Taichi初始化失败：
在代码中将GPU模式改为CPU模式：
```python
ti.init(arch=ti.cpu, default_fp=ti.f64)
```

### 如果PCG求解器不收敛：
- 减小时间步长（dt）
- 增加最大迭代次数
- 检查边界条件设置

## 🎯 验证要点

成功的模拟应该显示：

1. **PCG收敛**：通常10-50次迭代内收敛
2. **波前传播**：与理论预测一致的传播速度
3. **动能变化**：合理的动能演化曲线
4. **压力场**：平滑且无振荡的压力分布

## 📚 技术细节

这个实现包含了论文中的所有关键算法：

- ✅ 算子分裂法（Operator Splitting）
- ✅ 压力泊松方程求解（PCG + GFM）
- ✅ 水平集方法（WENO3 + RK3-TVD）
- ✅ 混合PIC/FLIP粒子更新
- ✅ 沙漏模式抑制
- ✅ 鬼流体方法边界条件

## 🆘 需要帮助？

如果遇到问题：

1. 检查 `reproduction/README_iMPM.md` 获取详细技术文档
2. 运行 `python test_impm.py` 进行诊断
3. 查看控制台输出中的错误信息
4. 确保所有依赖库版本正确

## 🎉 成功标志

当您看到类似这样的输出时，说明一切正常：

```
Frame   0 (t=0.000s, step=     0):
  Wave front: 0.600m
  KE: 0.000000 J
  Max velocity: 0.000 m/s
  PCG iterations: 15
  Performance: 45.2 steps/s

Frame   1 (t=0.100s, step=  1000):
  Wave front: 1.234m  
  KE: 1245.678900 J
  Max velocity: 3.456 m/s
  PCG iterations: 23
  Performance: 48.7 steps/s
```

祝您使用愉快！🌊