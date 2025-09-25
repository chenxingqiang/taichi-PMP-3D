# Two-phase MPM Debris Flow Impact Against Dual Rigid Barriers

## Reproduction Requirements for Ng et al. (2023)

**Original Paper**: "Two-phase MPM modelling of debris flow impact against dual rigid barriers"  
**Authors**: Charles Wang Wai Ng, Zhenyang Jia, Sunil Poudyal, Aastha Bhatta, Haiming Liu  
**Journal**: Géotechnique (2023)  
**DOI**: https://doi.org/10.1680/jgeot.22.00199  

### Project Objectives

1. **Primary Goal**: Reproduce the two-phase Material Point Method (MPM) solver with:
   - Incompressible fluid phase using Newtonian rheology
   - Solid phase with shear rate-dependent non-associated Drucker-Prager plasticity
   - Inter-phase drag coupling and fluidization effects
   - Dual rigid barrier interaction with overflow and landing mechanics

2. **Performance Targets**:
   - CUDA backend achieving >500 MLUPS on A100 GPU
   - YICA backend support for cross-platform compatibility
   - Memory efficiency supporting 500m³ debris volume simulations
   - Error convergence ≤5% vs experimental validation cases

3. **Validation Benchmarks**:
   - Sand-water mixture impact force reproduction (Fig. 5)
   - Flow kinematics validation (Fig. 4)  
   - Parametric study results (Figs. 10-12)
   - Froude number effects (Fr = 2, 4, 6)

### Physics & Mathematical Framework

#### Governing Equations
1. **Momentum Conservation** (Eqs. 5-6):
   ```
   ρ̄ₛ(Dvₛ/Dt) = ρ̄ₛg + ∇·σ' - f_d - φ∇pf
   ρ̄f(Dvf/Dt) = ρ̄fg + ∇·Tf + f_d - (1-φ)∇pf
   ```

2. **Incompressibility Constraint** (Eq. 9):
   ```
   ∇·vf = 0
   ```

3. **Solid Phase Constitutive Model**:
   - **Yield Surface fshear** (Eq. 17): `√J₂ - μₚp'`
   - **Compaction Surface fcompaction** (Eq. 15) with shear rate dependency
   - **Friction Coefficient** (Eq. 18): `μₚ = μ₁ + (μ₂-μ₁)/(1+b/Im) + (5φIᵥ)/(2aIm)`

4. **Inter-phase Drag Force** (Eq. 22):
   ```
   f_d = 18φ(1-φ)ηf/d² F̂(vₛ - vf)
   ```

#### Key Physical Parameters
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| **Solid Phase** |
| ρₛ | 2650 | kg/m³ | Sand density |
| μ₁ | 0.49 | - | Static friction coefficient |
| μ₂ | 1.4 | - | Limiting friction coefficient |
| φₘ | 0.56 | - | Critical solid volume fraction |
| d | 1.0 | mm | Particle diameter |
| **Fluid Phase** |
| ρf | 1000 | kg/m³ | Water density |
| ηf | 1×10⁻³ | Pa·s | Dynamic viscosity |
| **Simulation Domain** |
| Volume | 500 | m³ | Debris flow volume |
| Slope angle θ | 20° | - | Channel inclination |
| Froude numbers | 2, 4, 6 | - | Flow regime characterization |

### Computational Architecture

#### Backend Support
- **Primary**: CUDA backend for high-performance GPU computing
- **Secondary**: YICA backend for cross-platform compatibility  
- **CPU Fallback**: Required for testing and debugging

#### Data Structures
1. **Material Points**: Position, velocity, mass, volume, deformation gradient
2. **Background Grid**: Semi-staggered MAC grid for pressure-velocity coupling
3. **Barrier Geometry**: Rigid body representation with contact detection
4. **Level Set**: Interface tracking for overflow mechanics

#### Performance Requirements
- **Memory**: Support ≥100,000 material points per phase
- **Time Step**: Adaptive CFL-limited time stepping
- **Convergence**: Pressure Poisson solver tolerance 1×10⁻⁶
- **Output**: VTK export every 100 time steps

### Validation & Acceptance Criteria

#### Level 1: Unit Tests (>95% pass rate)
- [ ] P2G/G2P transfer conservation
- [ ] Drucker-Prager yield surface accuracy  
- [ ] Incompressible fluid solver convergence
- [ ] Drag force computation validation
- [ ] Barrier contact impulse conservation

#### Level 2: Integration Tests
- [ ] Single barrier impact force reproduction
- [ ] Overflow trajectory validation
- [ ] Two-phase coupling stability
- [ ] Mass conservation throughout simulation

#### Level 3: Experimental Validation
- [ ] Impact force time history RMSE ≤ 10% (Fig. 5)
- [ ] Flow kinematics visual comparison (Fig. 4)
- [ ] Landing distance accuracy ≤ 15%
- [ ] Fluidization ratio trends reproduction

#### Level 4: Parametric Study Reproduction
- [ ] Froude number effects on impact force (Fig. 12)
- [ ] Barrier spacing optimization curves (Fig. 12)
- [ ] Flow depth and velocity profiles (Figs. 10-11)
- [ ] Two-phase vs equivalent fluid comparison

### Production-Readiness Checklist

1. **Performance & Scalability**:
   - [ ] GPU memory optimization and profiling
   - [ ] Load balancing across multiple GPUs
   - [ ] Checkpoint/restart capability

2. **Reliability & Observability**:
   - [ ] Comprehensive error handling and logging
   - [ ] Performance metrics and monitoring
   - [ ] Memory leak detection and prevention

3. **Documentation & Usability**:
   - [ ] API documentation with examples
   - [ ] User guide with parameter selection
   - [ ] Developer guide for extensions

4. **Testing & Quality**:
   - [ ] Continuous integration pipeline
   - [ ] Regression test suite
   - [ ] Code coverage >80%

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Impact Force Error | ≤ 10% RMSE | vs Fig. 5 experimental data |
| Flow Kinematics | Visual match | vs Fig. 4 photographs |
| Performance | >500 MLUPS | A100 GPU, 64k particles |
| Memory Efficiency | <8GB | 500m³ simulation volume |
| Code Coverage | >80% | pytest + coverage |
| Documentation | Complete | All APIs + examples |

### Dependencies & External Resources

1. **Core Libraries**:
   - Taichi ≥ 1.7.4 (GPU computing framework)
   - NumPy ≥ 1.24 (numerical arrays)
   - PyVTK (visualization output)

2. **Validation Data**:
   - Experimental impact force measurements
   - High-speed camera flow kinematics
   - Particle size distribution data

3. **Reference Implementations**:
   - Original paper supplementary material
   - Related MPM solvers for benchmarking

### Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CUDA compatibility issues | Medium | High | Maintain CPU fallback, test across GPU generations |
| Performance bottlenecks | High | Medium | Early profiling, algorithm optimization |
| Validation data accuracy | Low | High | Multiple experimental datasets, error bounds |
| Complex physics debugging | High | Medium | Unit test coverage, visualization tools |

This requirements specification provides the foundation for the TDD development protocol implementation of the two-phase MPM debris flow solver.