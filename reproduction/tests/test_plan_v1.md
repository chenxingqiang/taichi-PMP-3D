# Two-Phase MPM Solver Test Plan v1.0

**Project**: Ng et al. (2023) Two-phase MPM Reproduction  
**Test-Driven Development Protocol**: Phase 1 Test Design  
**Testing Framework**: pytest + Taichi + CUDA/YICA  
**Coverage Target**: >95% unit tests, >80% overall code coverage

## Testing Philosophy & Strategy

Following the TDD Development Protocol, all tests are written **before implementation** to ensure:

1. **Precision in Early Stages**: Each equation and algorithm is verified against analytical solutions
2. **Controlled Evolution**: Implementation changes are validated by comprehensive test suites
3. **Evidence-Based Modifications**: All changes are supported by quantitative test results
4. **Real-System Validation**: Tests use production-equivalent configurations

## Test Hierarchy & Organization

```
tests/
├── unit/                           # Single component isolation tests
│   ├── test_solid_phase.py         # Drucker-Prager plasticity tests
│   ├── test_fluid_phase.py         # Incompressible flow solver tests  
│   ├── test_drag_model.py          # Inter-phase drag force tests
│   ├── test_barrier_model.py       # Contact detection tests
│   └── test_mpm_domain.py          # Domain-level coordination tests
├── integration/                    # Multi-component interaction tests
│   ├── test_two_phase_coupling.py  # Solid-fluid coupling validation
│   ├── test_barrier_impact.py      # Single barrier impact tests
│   ├── test_overflow_landing.py    # Overflow mechanics tests
│   └── test_dual_barriers.py       # Full dual barrier system tests
├── validation/                     # Experimental reproduction tests
│   ├── test_impact_forces.py       # Force time history validation (Fig. 5)
│   ├── test_flow_kinematics.py     # Visual kinematics validation (Fig. 4)  
│   ├── test_parametric_studies.py  # Froude number effects (Figs. 10-12)
│   └── test_fluidization_ratio.py  # Landing mechanics validation (Fig. 8)
├── performance/                    # Computational efficiency tests
│   ├── test_memory_usage.py        # GPU memory optimization tests
│   ├── test_throughput.py          # MLUPS benchmark tests
│   └── test_scalability.py         # Multi-GPU scaling tests
└── fixtures/                       # Test data and utilities
    ├── analytical_solutions.py     # Known analytical results
    ├── experimental_data.py        # Ng et al. (2023) validation data
    └── test_utilities.py           # Common testing functions
```

## Unit Test Specifications

### 1. SolidPhase Unit Tests (`test_solid_phase.py`)

#### 1.1 Drucker-Prager Yield Surface Validation

```python
class TestDruckerPragerYieldSurface:
    """Test Drucker-Prager plasticity model accuracy."""
    
    def test_yield_surface_equation_17(self):
        """Verify fshear = √J₂ - μₚp' within machine precision."""
        # Setup: Known stress state
        stress_tensor = create_known_stress_state()
        j2 = compute_second_invariant(stress_tensor)
        mean_pressure = compute_mean_effective_pressure(stress_tensor)
        friction_coeff = 0.49
        
        # Expected: Analytical yield function value  
        expected_yield = np.sqrt(j2) - friction_coeff * mean_pressure
        
        # Actual: Computed by SolidPhase implementation
        solid_phase = SolidPhase(1000, 32, 32, 32, 0.1)
        computed_yield = solid_phase.evaluate_yield_function(stress_tensor)
        
        # Validation: Machine precision accuracy
        assert abs(computed_yield - expected_yield) < 1e-14
        
    def test_friction_coefficient_equation_18(self):
        """Validate μₚ = μ₁ + (μ₂-μ₁)/(1+b/Im) + 5φIᵥ/(2aIm)."""
        # Parameters from physics_config.yaml
        mu1, mu2 = 0.49, 1.4
        a, b = 1.23, 0.31
        solid_fraction = 0.56
        
        # Test range of inertial numbers
        for I in [0.01, 0.1, 1.0, 10.0]:
            for Iv in [0.001, 0.01, 0.1]:
                Im = np.sqrt(I**2 + 2*Iv)
                
                expected_mu = (mu1 + (mu2-mu1)/(1+b/Im) + 
                              5*solid_fraction*Iv/(2*a*Im))
                
                computed_mu = solid_phase.compute_friction_coefficient(
                    I, Iv, solid_fraction)
                
                assert abs(computed_mu - expected_mu) < 1e-12
```

#### 1.2 Return Mapping Algorithm Validation

```python
def test_return_mapping_convergence(self):
    """Ensure return mapping converges within 10 iterations."""
    # Test with various stress states (elastic, plastic, tension)
    test_cases = [
        create_elastic_stress_state(),
        create_plastic_stress_state(),  
        create_tensile_stress_state()
    ]
    
    for initial_stress in test_cases:
        solid_phase = SolidPhase(1000, 32, 32, 32, 0.1)
        
        result = solid_phase.return_mapping_algorithm(
            initial_stress, strain_increment=0.001, dt=1e-4)
        
        # Convergence requirements
        assert result.converged == True
        assert result.iterations <= 10
        assert result.residual < 1e-10
        
def test_plastic_consistency(self):
    """Verify plastic flow consistency condition."""
    # If plastic flow occurs, yield function should be zero
    stress_states = generate_plastic_stress_states(n_samples=100)
    
    for stress in stress_states:
        solid_phase = SolidPhase(1000, 32, 32, 32, 0.1)
        updated_stress = solid_phase.return_mapping_algorithm(stress, 0.01, 1e-4)
        
        if solid_phase.plastic_multiplier > 1e-12:  # Plastic flow occurred
            yield_value = solid_phase.evaluate_yield_function(updated_stress)
            assert abs(yield_value) < 1e-10  # Must be on yield surface
```

#### 1.3 P2G/G2P Transfer Conservation

```python
def test_momentum_conservation_p2g(self):
    """Validate momentum conservation in particle-to-grid transfer."""
    # Setup: Known particle configuration
    solid_phase = SolidPhase(1000, 64, 64, 64, 0.1)
    grid_fields = create_empty_grid_fields()
    
    # Initialize particles with known momenta
    initial_momentum = initialize_test_particles(solid_phase)
    
    # Execute P2G transfer
    solid_phase.particle_to_grid_transfer(grid_fields)
    
    # Compute total grid momentum
    final_momentum = compute_total_grid_momentum(grid_fields)
    
    # Conservation check (accounting for boundary effects)
    momentum_error = np.linalg.norm(final_momentum - initial_momentum)
    assert momentum_error < 1e-12
    
def test_mass_conservation_p2g(self):
    """Verify mass conservation in P2G transfer.""" 
    solid_phase = SolidPhase(1000, 64, 64, 64, 0.1)
    
    initial_mass = compute_total_particle_mass(solid_phase)
    solid_phase.particle_to_grid_transfer(grid_fields)
    final_mass = compute_total_grid_mass(grid_fields)
    
    assert abs(final_mass - initial_mass) < 1e-14
```

### 2. FluidPhase Unit Tests (`test_fluid_phase.py`)

#### 2.1 Incompressible Flow Solver Validation

```python
class TestIncompressibleFlowSolver:
    """Test incompressible fluid phase implementation."""
    
    def test_poiseuille_flow_analytical(self):
        """Reproduce analytical Poiseuille flow solution."""
        # Setup: Channel flow with pressure gradient
        fluid_phase = FluidPhase(5000, 64, 32, 32, 0.05)
        
        # Boundary conditions: dp/dx = constant, no-slip walls
        pressure_gradient = 100.0  # Pa/m
        channel_width = 1.6  # m (32 cells × 0.05 m)
        
        # Run steady-state simulation
        for step in range(1000):
            fluid_phase.solve_pressure_poisson(dt=1e-4)
            if fluid_phase.check_steady_state(): break
        
        # Extract velocity profile at channel center
        velocity_profile = fluid_phase.get_velocity_profile(x_center=True)
        
        # Analytical solution: u = (1/2μ)(dp/dx)y(h-y)
        h = channel_width / 2
        y = np.linspace(-h, h, 32)
        mu = 1e-3  # Pa·s
        analytical_u = (1/(2*mu)) * pressure_gradient * y * (h - np.abs(y))
        
        # Validation: L2 error < 1%
        l2_error = np.linalg.norm(velocity_profile - analytical_u) / np.linalg.norm(analytical_u)
        assert l2_error < 0.01
        
    def test_divergence_free_condition(self):
        """Verify ∇·vf = 0 enforcement."""
        fluid_phase = FluidPhase(10000, 64, 64, 64, 0.1)
        
        # Initialize with divergent velocity field
        initialize_divergent_velocity_field(fluid_phase)
        
        # Apply incompressibility projection
        initial_divergence = compute_velocity_divergence(fluid_phase)
        fluid_phase.solve_pressure_poisson(dt=1e-4, tolerance=1e-10)
        final_divergence = compute_velocity_divergence(fluid_phase)
        
        # Validation: Divergence reduced by factor >1000
        assert np.max(np.abs(final_divergence)) < 1e-10
        assert np.max(np.abs(final_divergence)) < np.max(np.abs(initial_divergence)) / 1000
```

#### 2.2 Pressure Poisson Solver Tests

```python
def test_conjugate_gradient_convergence(self):
    """Test CG solver convergence for pressure Poisson equation."""
    fluid_phase = FluidPhase(1000, 32, 32, 32, 0.1)
    
    # Setup: Known RHS with analytical solution  
    rhs = create_analytical_poisson_rhs()
    analytical_solution = solve_poisson_analytical(rhs)
    
    # Solve using CG implementation
    iterations = fluid_phase.solve_pressure_poisson_cg(rhs, tolerance=1e-12)
    computed_solution = fluid_phase.get_pressure_solution()
    
    # Convergence checks
    assert iterations <= 100  # Should converge quickly for smooth RHS
    
    solution_error = np.linalg.norm(computed_solution - analytical_solution)
    assert solution_error < 1e-10
```

### 3. DragModel Unit Tests (`test_drag_model.py`)

#### 3.1 Van der Hoef Drag Correlation

```python
class TestDragModel:
    """Test inter-phase drag force computation."""
    
    def test_van_der_hoef_correlation(self):
        """Validate F̂(Re, φ) drag coefficient correlation."""
        drag_model = DragModel()
        
        # Test cases from Van der Hoef et al. (2005) paper
        test_cases = [
            (0.1, 0.1, 18.0),    # (Re, φ, expected_F_hat)
            (1.0, 0.2, 24.5),
            (10.0, 0.4, 35.2),
            (100.0, 0.5, 48.6)
        ]
        
        for reynolds, solid_fraction, expected_f_hat in test_cases:
            computed_f_hat = drag_model.compute_drag_coefficient(
                solid_fraction, reynolds)
            
            # Tolerance based on correlation accuracy
            relative_error = abs(computed_f_hat - expected_f_hat) / expected_f_hat
            assert relative_error < 0.05  # Within 5% of reference
            
    def test_drag_force_equation_22(self):
        """Verify f_d = 18φ(1-φ)ηf/d² F̂(vs - vf)."""
        drag_model = DragModel()
        
        # Known parameters
        phi = 0.4
        eta_f = 1e-3  # Pa·s
        d = 1e-3      # m
        v_rel = np.array([1.0, 0.5, -0.3])  # m/s
        
        # Compute drag force
        drag_force = drag_model.compute_interphase_drag_force(
            phi, eta_f, d, v_rel)
        
        # Analytical expectation
        f_hat = drag_model.compute_drag_coefficient(phi, compute_reynolds(v_rel, d))
        expected_drag = 18 * phi * (1-phi) * eta_f / (d*d) * f_hat * v_rel
        
        np.testing.assert_array_almost_equal(drag_force, expected_drag, decimal=12)
```

#### 3.2 Fluidization Ratio Computation

```python
def test_fluidization_ratio_equation_26(self):
    """Validate λ = pbed/(pbed + σ'bed)."""
    drag_model = DragModel()
    
    # Test cases with known pressure and stress states
    test_cases = [
        (1000.0, 5000.0, 0.2),     # (pbed, sigma_bed, expected_lambda)
        (2000.0, 3000.0, 0.4),
        (0.0, 1000.0, 0.0),        # No pore pressure
        (1000.0, 0.0, 1.0),        # No effective stress
    ]
    
    for p_bed, sigma_bed, expected_lambda in test_cases:
        computed_lambda = drag_model.compute_fluidization_ratio(p_bed, sigma_bed)
        assert abs(computed_lambda - expected_lambda) < 1e-14
        
def test_friction_reduction_equation_25(self):
    """Test Ffric = μbed × σbed × (1-λ)."""
    drag_model = DragModel()
    
    mu_bed = 0.4
    lambda_values = [0.0, 0.3, 0.6, 0.9]
    sigma_bed = 1000.0
    
    for lam in lambda_values:
        friction_force = drag_model.compute_bed_friction(mu_bed, sigma_bed, lam)
        expected_friction = mu_bed * sigma_bed * (1 - lam)
        
        assert abs(friction_force - expected_friction) < 1e-12
```

### 4. BarrierModel Unit Tests (`test_barrier_model.py`)

#### 4.1 Contact Detection & Forces

```python
class TestBarrierModel:
    """Test rigid barrier contact mechanics."""
    
    def test_penalty_contact_forces(self):
        """Validate penalty method contact force computation."""
        barrier_model = BarrierModel(barrier_height=2.0, barrier_spacing=5.0)
        
        # Test particle positions (some penetrating barrier)
        particle_positions = np.array([
            [1.0, 1.0, 0.0],   # No contact
            [2.0, 0.5, 0.0],   # Penetrating barrier
            [2.1, 1.5, 0.0],   # Just touching
        ])
        
        contact_forces = barrier_model.compute_contact_forces(particle_positions)
        
        # Validation checks
        assert np.linalg.norm(contact_forces[0]) < 1e-12  # No force for non-contact
        assert contact_forces[1, 0] < -1000  # Repulsive force in x-direction
        assert abs(contact_forces[2, 0]) < 100  # Small force at contact threshold
        
    def test_coulomb_friction_contact(self):
        """Test Coulomb friction law in contact.""" 
        barrier_model = BarrierModel(barrier_height=2.0, barrier_spacing=5.0)
        
        # Sliding contact scenario
        normal_force = 1000.0  # N
        tangent_velocity = np.array([0.0, 0.5, 0.0])  # m/s
        friction_coeff = 0.4
        
        friction_force = barrier_model.compute_friction_force(
            normal_force, tangent_velocity, friction_coeff)
        
        # Should equal μ × |N| in opposite direction to sliding
        expected_magnitude = friction_coeff * normal_force
        computed_magnitude = np.linalg.norm(friction_force)
        
        assert abs(computed_magnitude - expected_magnitude) < 1e-10
        
        # Direction opposite to sliding velocity
        friction_direction = friction_force / computed_magnitude  
        sliding_direction = tangent_velocity / np.linalg.norm(tangent_velocity)
        assert np.dot(friction_direction, sliding_direction) < -0.99  # Nearly opposite
```

#### 4.2 Overflow Trajectory Validation

```python
def test_trajectory_equation_3(self):
    """Validate analytical landing distance calculation.""" 
    barrier_model = BarrierModel(barrier_height=2.0, barrier_spacing=5.0)
    
    # Test parameters from paper
    v_launch = 7.4  # m/s
    theta = 20 * np.pi/180  # slope angle
    H_B = 2.0  # barrier height
    g = 9.81  # m/s²
    
    # Analytical calculation (Equation 3)
    expected_xi = ((v_launch**2)/(g*np.cos(theta)) * 
                   (np.tan(theta) + np.sqrt(np.tan(theta)**2 + 
                    2*g*H_B/(v_launch**2 * np.cos(theta)))) + 
                   H_B * np.tan(theta))
    
    computed_xi = barrier_model.calculate_theoretical_landing_distance(
        v_launch, theta, H_B)
    
    assert abs(computed_xi - expected_xi) < 1e-10
    
def test_landing_velocity_equation_4(self):
    """Test vi = R × vr × cos(θland) calculation."""
    barrier_model = BarrierModel(barrier_height=2.0, barrier_spacing=5.0)
    
    v_r = 8.5  # velocity just before landing
    theta_land = 30 * np.pi/180  # landing angle  
    R = 1.0  # velocity correction factor
    
    expected_vi = R * v_r * np.cos(theta_land)
    computed_vi = barrier_model.compute_landing_velocity(v_r, theta_land, R)
    
    assert abs(computed_vi - expected_vi) < 1e-14
```

## Integration Test Specifications

### 5. Two-Phase Coupling Tests (`test_two_phase_coupling.py`)

#### 5.1 Momentum Exchange Validation

```python
class TestTwoPhaseCoupling:
    """Test solid-fluid momentum exchange."""
    
    def test_total_momentum_conservation(self):
        """Verify total system momentum conservation with drag."""
        domain = MPM2PDomain(64, 64, 64, 0.1, 5000, 5000)
        
        # Initialize with relative motion between phases
        initialize_counter_flow(domain.solid_phase, domain.fluid_phase)
        
        initial_momentum = (compute_solid_momentum(domain.solid_phase) +
                           compute_fluid_momentum(domain.fluid_phase))
        
        # Run simulation with drag coupling
        for step in range(100):
            domain.step(dt=1e-4)
        
        final_momentum = (compute_solid_momentum(domain.solid_phase) +
                         compute_fluid_momentum(domain.fluid_phase))
        
        # Total momentum should be conserved (no external forces)
        momentum_error = np.linalg.norm(final_momentum - initial_momentum)
        assert momentum_error < 1e-10
        
    def test_drag_equilibrium(self):
        """Test approach to drag equilibrium state."""
        domain = MPM2PDomain(32, 32, 32, 0.1, 1000, 1000) 
        
        # Initialize with velocity difference
        v_rel_initial = 2.0  # m/s relative velocity
        initialize_relative_velocity(domain, v_rel_initial)
        
        # Run until equilibrium
        for step in range(1000):
            domain.step(dt=1e-5)
            v_rel = compute_relative_velocity(domain)
            if v_rel < 0.01 * v_rel_initial: break  # 99% reduction
        
        # Should reach equilibrium in reasonable time
        assert step < 500
        assert v_rel < 0.01
```

### 6. Barrier Impact Tests (`test_barrier_impact.py`)

#### 6.1 Single Barrier Validation

```python
def test_single_barrier_impact_force(self):
    """Reproduce single barrier impact force from experiments."""
    # Configure domain matching experimental setup (0.2m wide, 5m long flume)
    domain = MPM2PDomain(100, 40, 40, 0.05, 2000, 2000)
    
    # Load experimental configuration
    config = load_experimental_config("sand_water_mixture")
    setup_dam_break_initial_condition(domain, config)
    
    # Run simulation until barrier impact
    impact_forces = []
    times = []
    
    for step in range(2000):  # 2 seconds at dt=1e-3
        dt = 1e-3
        domain.step(dt)
        
        # Record impact force on barrier
        force = domain.barrier_model.get_total_impact_force()
        impact_forces.append(force)
        times.append(step * dt)
        
        if step * dt > 2.0: break
    
    # Load experimental data for comparison
    exp_times, exp_forces = load_experimental_data("Fig5_sand_water_mixture")
    
    # Interpolate simulation results to experimental time points
    sim_forces_interp = np.interp(exp_times, times, impact_forces)
    
    # Validation: RMSE < 10% of peak force
    peak_force = np.max(exp_forces)
    rmse = np.sqrt(np.mean((sim_forces_interp - exp_forces)**2))
    relative_rmse = rmse / peak_force
    
    assert relative_rmse < 0.10  # Within 10% RMSE tolerance
```

## Performance Test Specifications

### 7. Memory Usage Tests (`test_memory_usage.py`)

```python
class TestMemoryUsage:
    """Test GPU memory efficiency and optimization."""
    
    def test_500m3_simulation_memory_footprint(self):
        """Verify 500m³ simulation fits within 8GB GPU memory."""
        # Large-scale configuration
        domain = MPM2PDomain(128, 128, 128, 0.2, 50000, 50000)
        
        # Initialize full-scale debris flow
        setup_500m3_debris_flow(domain)
        
        # Measure GPU memory usage
        initial_memory = get_gpu_memory_usage()
        domain.step(dt=1e-4)  # Single step to allocate all arrays
        peak_memory = get_gpu_memory_usage()
        
        memory_used = peak_memory - initial_memory
        
        # Validation: Memory usage < 8GB target
        assert memory_used < 8 * 1024**3  # 8 GB in bytes
        
    def test_memory_leak_detection(self):
        """Detect memory leaks over extended simulation."""
        domain = MPM2PDomain(64, 64, 64, 0.1, 5000, 5000)
        
        memory_samples = []
        for step in range(1000):
            domain.step(dt=1e-4)
            
            if step % 100 == 0:
                memory_samples.append(get_gpu_memory_usage())
        
        # Memory usage should be stable after initial allocation
        memory_growth = memory_samples[-1] - memory_samples[1] 
        assert memory_growth < 100 * 1024**2  # Less than 100 MB growth
```

### 8. Throughput Tests (`test_throughput.py`)

```python
def test_mlups_benchmark(self):
    """Measure computational throughput in MLUPS."""
    domain = MPM2PDomain(64, 64, 64, 0.1, 10000, 10000)
    
    # Warmup runs
    for _ in range(10):
        domain.step(dt=1e-4)
    
    # Timed benchmark
    n_steps = 100
    n_particles = domain.get_total_particle_count()
    
    start_time = time.time()
    for _ in range(n_steps):
        domain.step(dt=1e-4)
    end_time = time.time()
    
    # Calculate MLUPS (Million Lattice Updates Per Second)
    total_updates = n_steps * n_particles
    elapsed_time = end_time - start_time
    mlups = (total_updates / 1e6) / elapsed_time
    
    # Performance target: >500 MLUPS on A100
    if get_gpu_type() == "A100":
        assert mlups > 500.0
    else:
        # Relaxed target for other GPUs
        assert mlups > 100.0
```

## Validation Test Specifications

### 9. Experimental Reproduction Tests (`test_impact_forces.py`)

```python
class TestExperimentalValidation:
    """Reproduce experimental results from Ng et al. (2023)."""
    
    def test_figure_5_impact_force_reproduction(self):
        """Reproduce Figure 5 impact force time histories."""
        test_cases = [
            "dry_sand",
            "water", 
            "sand_water_mixture"
        ]
        
        for case in test_cases:
            # Setup simulation matching experimental conditions
            domain = create_experimental_setup(case)
            
            # Run simulation and record impact forces
            sim_times, sim_forces = run_impact_simulation(domain)
            
            # Load experimental data
            exp_times, exp_forces = load_experimental_data(f"Fig5_{case}")
            
            # Compute validation metrics
            rmse = compute_rmse(sim_forces, exp_forces, sim_times, exp_times)
            peak_error = compute_peak_force_error(sim_forces, exp_forces)
            
            # Acceptance criteria from requirements
            assert rmse < 0.10 * np.max(exp_forces)  # RMSE < 10% of peak
            assert peak_error < 0.15  # Peak force error < 15%
            
    def test_figure_4_flow_kinematics(self):
        """Validate flow kinematics visual comparison."""
        domain = create_experimental_setup("sand_water_mixture")
        
        # Key time points from Figure 4
        validation_times = [0.0, 0.2, 0.4, 2.0]
        
        for t in validation_times:
            # Run to specific time
            run_to_time(domain, t)
            
            # Extract flow kinematics
            velocity_field = domain.get_velocity_field()
            particle_positions = domain.get_particle_positions()
            
            # Load experimental benchmarks
            exp_data = load_experimental_kinematics(f"Fig4_t{t}s")
            
            # Quantitative comparison metrics
            front_velocity_error = compare_front_velocity(velocity_field, exp_data)
            overflow_angle_error = compare_overflow_angle(particle_positions, exp_data)
            
            # Visual comparison acceptance criteria
            assert front_velocity_error < 0.20  # Within 20% of measured
            assert overflow_angle_error < 5.0   # Within 5 degrees
```

## Test Data & Fixtures

### 10. Analytical Solutions (`fixtures/analytical_solutions.py`)

```python
def poiseuille_flow_solution(y, dp_dx, mu, h):
    """Analytical velocity profile for Poiseuille flow."""
    return (1/(2*mu)) * dp_dx * y * (h - np.abs(y))

def couette_flow_solution(y, U_wall, h):
    """Linear velocity profile for Couette flow."""
    return U_wall * (y + h) / (2 * h)

def drucker_prager_yield_function(J2, p_prime, mu_p):
    """Analytical Drucker-Prager yield function."""
    return np.sqrt(J2) - mu_p * p_prime

def trajectory_landing_distance(v_launch, theta, H_B, g=9.81):
    """Analytical projectile motion landing distance."""
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)
    
    xi = ((v_launch**2)/(g*cos_theta) * 
          (tan_theta + np.sqrt(tan_theta**2 + 2*g*H_B/(v_launch**2 * cos_theta))) + 
          H_B * tan_theta)
    return xi
```

### 11. Experimental Data Loading (`fixtures/experimental_data.py`)

```python
def load_experimental_data(dataset_name):
    """Load experimental validation data from Ng et al. (2023)."""
    data_files = {
        "Fig5_dry_sand": "fig5_dry_sand_force.csv",
        "Fig5_water": "fig5_water_force.csv", 
        "Fig5_sand_water_mixture": "fig5_mixture_force.csv",
        "Fig4_t0.0s": "fig4_kinematics_t0.json",
        "Fig4_t0.2s": "fig4_kinematics_t0p2.json",
        # ... additional datasets
    }
    
    file_path = os.path.join("experimental_data", data_files[dataset_name])
    
    if dataset_name.startswith("Fig5"):
        return load_force_time_series(file_path)
    elif dataset_name.startswith("Fig4"):
        return load_kinematics_data(file_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
```

## Test Execution Strategy

### Continuous Integration Pipeline

```yaml
# .github/workflows/test_pipeline.yml
name: Two-Phase MPM Test Suite

on: [push, pull_request]

jobs:
  unit_tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        backend: [cpu, cuda]
        
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        pip install taichi pytest numpy
        pip install -e .
        
    - name: Run unit tests
      run: pytest tests/unit/ -v --backend=${{ matrix.backend }}
      
  integration_tests:
    needs: unit_tests
    runs-on: gpu-runner
    
    steps:
    - name: Run integration tests
      run: pytest tests/integration/ -v --backend=cuda
      
  validation_tests:
    needs: integration_tests
    runs-on: gpu-runner
    
    steps:
    - name: Run experimental validation
      run: pytest tests/validation/ -v --backend=cuda --slow
```

### Test Coverage Requirements

| Test Category | Coverage Target | Acceptance Criteria |
|---------------|-----------------|-------------------|
| Unit Tests | >95% pass rate | All physics equations validated |
| Integration Tests | >90% pass rate | Multi-component interactions verified |
| Validation Tests | >80% pass rate | Experimental data reproduced within tolerance |
| Performance Tests | 100% pass rate | Memory and throughput targets met |

## Conclusion

This comprehensive test plan provides the foundation for Test-Driven Development of the two-phase MPM solver. The test suite ensures:

1. **Mathematical Accuracy**: Every equation from the paper is validated against analytical solutions
2. **Physical Realism**: Conservation laws and constitutive relationships are verified
3. **Experimental Fidelity**: Simulation results match published experimental data
4. **Production Readiness**: Performance, memory, and reliability requirements are met

**Next Steps**: Implement the test scaffolding (Phase 2) to create the initial failing tests that will guide the implementation of each component.