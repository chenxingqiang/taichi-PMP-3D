# Taichi MPM 3D: Two-Phase Debris Flow Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Taichi](https://img.shields.io/badge/Taichi-1.0.2+-green.svg)](https://www.taichi-lang.org/)

A high-performance implementation of the **Two-Phase Material Point Method (MPM)** for simulating saturated granular flows and debris flow impact on barriers, based on the Taichi programming language.

## 🎯 Features

### Core Capabilities

- **Two-Phase Coupling**: Solid (granular) and fluid (pore water) phases with inter-phase drag
- **Drucker-Prager μ(I) Rheology**: Rate-dependent constitutive model for granular materials
- **Incompressible Flow**: Pressure Poisson equation with PCG solver
- **Ghost Fluid Method (GFM)**: Accurate free surface boundary conditions
- **Mixed PIC/FLIP**: Stable and low-dissipation velocity transfer
- **Full 3D Support**: GPU-accelerated simulations via Taichi

### Validated Test Cases

| Simulation | Status | Description |
|------------|--------|-------------|
| Dam Break | ✅ | Single-phase free surface flow (Martin & Moyce 1952) |
| Saturated Column Collapse | ✅ | Two-phase granular flow (Ceccato et al. 2020) |
| Barrier Impact | 🔄 | Debris flow impact on rigid barriers |

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/chenxingqiang/Taichi-PMP-3D.git
cd Taichi-PMP-3D

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- Taichi >= 1.0.2
- NumPy >= 1.20
- Matplotlib >= 3.5
- PyYAML >= 6.0

## 🚀 Quick Start

### Single-Phase Dam Break

```python
from src.incompressible_mpm_solver import IncompressibleMPMSolver

# Create solver
solver = IncompressibleMPMSolver(
    nx=64, ny=32, nz=32,
    dx=0.02,
    rho=1000.0,
    mu=1e-3,
    g=9.8
)

# Initialize dam break
solver.initialize_particles_dam_break(
    x_min=0.0, x_max=0.6,
    y_min=0.0, y_max=1.0,
    z_min=0.0, z_max=0.6,
    ppc=8
)

# Run simulation
for step in range(1000):
    solver.step()
```

### Two-Phase Debris Flow

```python
from src.two_phase_mpm_solver import TwoPhaseMPMSolver

# Create two-phase solver
solver = TwoPhaseMPMSolver(
    nx=100, ny=25, nz=20,
    dx=0.02,
    rho_s=2650.0,           # Solid density (kg/m³)
    E_s=1e7,                # Young's modulus (Pa)
    friction_angle=26.0,    # Friction angle (degrees)
    rho_f=1000.0,           # Fluid density (kg/m³)
    mu_f=0.01,              # Fluid viscosity (Pa·s)
    d_s=0.001,              # Particle diameter (m)
    phi_s0=0.55             # Initial solid volume fraction
)

# Initialize particles
solver.init_particles(
    x_min=0.04, x_max=0.44,
    y_min=0.04, y_max=0.34,
    z_min=0.04, z_max=0.44,
    ppc=4
)

# Run simulation with pressure coupling
for step in range(3000):
    solver.step()  # Includes pressure solve
    
    # Export results
    data = solver.export_particles()
```

## 📁 Project Structure

```
taichi-mpm-3d/
├── README.md
├── LICENSE
├── requirements.txt
├── Dockerfile
├── docs/                           # Documentation
│   ├── main.pdf                    # Main paper
│   └── model.pdf                   # Model description
├── taichi_mpm/                     # Main package (lazy imports)
│   └── __init__.py
├── src/                            # Source code
│   ├── incompressible_mpm_solver.py    # Single-phase iMPM
│   ├── two_phase_mpm_solver.py         # Two-phase MPM
│   ├── pcg_solver.py                   # PCG pressure solver
│   ├── level_set_method.py             # Level set tracking
│   ├── barrier_model.py                # Barrier contact
│   ├── run_dam_break_validation.py     # Dam break script
│   ├── run_ceccato_collapse.py         # Column collapse script
│   ├── physics_config.yaml             # Parameters
│   ├── 2phase/                         # Two-phase kernels
│   │   ├── collapse.py
│   │   └── drucker_prager_rheology.py
│   ├── tests/                          # Test suite
│   └── simulation_output/              # Results
├── examples/                       # Example scripts
└── configs/                        # Configuration files
```

## 🔬 Mathematical Framework

### Governing Equations

**Solid Phase:**
```
ρ̄_s (Dv_s/Dt) = ρ̄_s g + ∇·σ' - f_d - φ∇p_f
```

**Fluid Phase:**
```
ρ̄_f (Dv_f/Dt) = ρ̄_f g + ∇·T_f + f_d - (1-φ)∇p_f
```

**Incompressibility:**
```
∇·v_f = 0  →  ∇²p = (ρ/Δt)∇·v*
```

### Key Components

1. **Pressure Poisson Solver** (PCG with MIC/SSOR preconditioner)
   - Ghost Fluid Method for free surface: θ-based coefficient modification
   - Bridson's algorithm for solid boundaries
   - Convergence: 15-30 iterations with MIC

2. **Constitutive Model** (Drucker-Prager with μ(I))
   ```
   μ_p = μ₁ + (μ₂ - μ₁)/(1 + b/I_m) + 5/2 · (φI_v)/(aI_m)
   ```

3. **Inter-Phase Drag** (Di Felice model)
   ```
   f_d = 18φ(1-φ)η_f/d² · F̂(φ,Re) · (v_s - v_f)
   ```

## 📊 Validation Results

### Dam Break (Martin & Moyce 1952)

| Metric | Experimental | Simulation | Error |
|--------|-------------|------------|-------|
| Wave front at t*=1.0 | 1.6 L₀ | 1.55 L₀ | 3.1% |
| Wave front at t*=2.0 | 2.8 L₀ | 2.72 L₀ | 2.9% |

### Saturated Column Collapse (Ceccato et al. 2020)

- Initial column: H₀ = 0.5m, L₀ = 0.25m (aspect ratio = 2)
- Final height: 0.79 H₀
- Final runout: 1.9 L₀

## ⚙️ Configuration

Physical parameters are defined in `physics_config.yaml`:

```yaml
solid:
  density: 2650.0         # kg/m³
  youngs_modulus: 1.0e7   # Pa
  poisson_ratio: 0.3
  friction_angle: 26.0    # degrees
  mu_1: 0.49              # static friction
  mu_2: 1.4               # dynamic friction

fluid:
  density: 1000.0         # kg/m³
  viscosity: 0.001        # Pa·s

coupling:
  particle_diameter: 0.001  # m
  initial_porosity: 0.45
```

## 🧪 Running Tests

```bash
cd src
pytest tests/ -v
```

### Validation Scripts

```bash
cd src

# Dam break validation
python run_dam_break_validation.py

# Ceccato column collapse
python run_ceccato_collapse.py

# Full simulation with plots
python run_simulation_and_plot.py
```

## 📈 Performance

- **Time Step**: Adaptive based on CFL condition (shear wave speed)
- **PCG Convergence**: 15-30 iterations with MIC preconditioner
- **GPU Acceleration**: ~10x speedup on NVIDIA GPUs via Taichi

### Recommended Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| `dx` | 0.01-0.02 m | Grid spacing |
| `dt` | 1e-4 - 5e-4 s | Time step |
| `ppc` | 4-8 | Particles per cell |
| `flip_ratio` | 0.95-0.99 | PIC/FLIP blending |

## 📖 References

1. **Primary Paper**: "Two-phase debris flow impact on flexible barriers" (2024)
2. **iMPM Method**: "Incompressible material point method for free surface flow"
3. **μ(I) Rheology**: Jop et al. (2006) "A constitutive law for dense granular flows"
4. **Ceccato Validation**: Ceccato et al. (2020) "Two-phase MPM for saturated soils"

## 📝 Citation

```bibtex
@software{taichi_mpm_3d_2024,
  title={Taichi MPM 3D: Two-Phase Debris Flow Simulation},
  author={Chen, Xingqiang},
  year={2024},
  url={https://github.com/chenxingqiang/Taichi-PMP-3D}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 附录：论文复现指南

<details>
<summary>点击展开详细复现流程</summary>

### 一、模型构建与仿真设置

#### 1. 仿真几何与初始条件

- 倾斜渠道坡度：θ = 20°
- 泥石流体积：500 m³（假设渠道宽度10 m，二维平面应变模型）
- 初始长深比：L_D / H_D ≈ 8
- 双刚性屏障高度：H_B = 2 × h_flow

#### 2. 网格与材料点生成

- 背景网格尺寸与屏障高度比：0.04
- 每网格单元初始生成16个材料点
- 采用二维平面应变假设

#### 3. 边界条件

- 流体相：自由滑移边界（法向速度为零）
- 固体相：Coulomb摩擦定律，基底摩擦系数 μ_bed = 0.4

### 二、参数设置（Table 1）

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 固体密度 | ρ_s | 2650 | kg/m³ |
| 流体密度 | ρ_f | 1000 | kg/m³ |
| 流体动力粘度 | η_f | 0.001 | Pa·s |
| 固体颗粒直径 | d | 1 | mm |
| 杨氏模量 | E | 10 | MPa |
| 泊松比 | ν | 0.3 | - |
| 静态摩擦系数 | μ₁ | 0.49 | - |
| 极限摩擦系数 | μ₂ | 1.4 | - |
| 临界固体体积分数 | φ_m | 0.56 | - |
| dilatancy参数 | K₄ | 4.7 | - |
| μ(I)参数 | a | 1.23 | - |
| μ(I)参数 | b | 0.31 | - |
| 基底摩擦系数 | μ_bed | 0.4 | - |

### 三、仿真流程

1. **自由流动仿真**：获取 h_flow 和 v_flow
2. **单屏障冲击**：记录冲击力、溢出速度
3. **双屏障仿真**：分析不同间距下的流体化比率变化

</details>

---

**Last Updated**: December 2024  
**Status**: Core implementation complete, validation ongoing
