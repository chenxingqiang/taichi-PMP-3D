# solver for the CLEAR board examples

# main references - papers:
# 1. [Jiang et al., APIC, 2015] (https://dl.acm.org/doi/10.1145/2766996)
# 2. [Jiang et al. JCP, 2017] (https://www.sciencedirect.com/science/article/pii/S0021999117301535)
# 3. [MPM SIGGRAPH 2016 course] (https://dl.acm.org/doi/10.1145/2897826.2927348)
# 4. [Hu et al., MLS-MPM] (https://dl.acm.org/doi/10.1145/3197517.3201293)
# 5. [Klar et al., Drucker-Prager sand simulation] (https://dl.acm.org/doi/10.1145/2897824.2925906)
# 6. [Klar et al., Drucker-Prager sand simulation - supplementary file] (https://www.seas.upenn.edu/~cffjiang/research/sand/tech-doc.pdf)
# 7. [Neto, Borja, ActaGeo, 2018] (https://link.springer.com/article/10.1007/s11440-018-0700-3)
# 8. [YUN (RAYMOND) FEI et al., SIGGRAPH 2021] (http://yunfei.work/asflip/)

# main references - code:
# 1. mpm99.py: from Taichi/examples/mpm99.py


import taichi as ti
import numpy as np
import time
import sys
import os

sys.path.append("..")

# materials
# from materials.drucker_prager import *
from materials.drucker_prager_rheology import *
# from materials.dem_pillar_3D import *
from materials.dem_pillar_box_3D import *

pi = 3.141592653
inv_pi = 1 / pi


def parametersetting(friction,
                     friction_side,
                     FLIPcoeff,
                     friction_angle,
                     mu_2,
                     xifactor,
                     sand_E=2.016e4,
                     sand_nu=0.3,
                     board_inclination=20,
                     timestep=None,
                     DEM_contact=False,
                     DEMlist=[[1.86, 2.86 / 2], [1.86, 2.86 / 2]],
                     columnR=0.01):
    # ============ PARTICLES, GRID QUANTITIES DECLARATION ============
    global n_s_particles, n_l_particles, n_grid_x, n_grid_y, n_grid_z, max_x, dx, inv_dx, max_y, max_z, dt, inv_dt, timelimit, dim \
        , inclination, standard_gravity, gravity, boundary_friction_coeff, boundary_friction_coeff_side, tol, w0, l0, d0, h0, w1, l1, d1, h1, start_x, start_y \
        , n_particles_per_direction_x, n_particles_per_direction_y, n_particles_per_direction_z, p_vol, mass_s, mass_l, FLIP_blending_coeff \
        , APIC_flag, beta_max, potential_coeff_k, DEM_r, rho_s, s_dia, rho_l, rho_critical, calmstep, dx, n_grid_x, n_grid_y, iContainerOffset, k_n_crit, my_DEM_pillar, DEMx, DEMy

    n_s_particles = ti.field(dtype=int, shape=())
    n_l_particles = ti.field(dtype=int, shape=())
    # Define computational domain
    max_num_s_particles = 148120  # maximum particle number (should >= real particle number used)
    n_grid_x = 150  # grid number along the x dir
    n_grid_y = 40
    n_grid_z = 150

    max_x = 1.3  # maximum x/y coordinate
    dx, inv_dx = max_x / n_grid_x, float(n_grid_x / max_x)  # dx = dy = dz
    iContainerOffset = int(0.0 * inv_dx)
    max_y = dx * n_grid_y
    max_z = dx * n_grid_z
    dim = 3  # problem dimension
    inclination = board_inclination
    standard_gravity = ti.Vector([0, -9.81, 0])
    gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
    boundary_friction_coeff = friction  # friction coefficient for the bottom boundary
    boundary_friction_coeff_side = friction_side
    tol = 1e-10
    # initial sample dimensions (d0 is not used)
    w0 = 0.2
    w0 -= w0 % dx  # algin w0 with grids
    l0 = 0.4
    h0 = 0.7
    w1 = 0.2
    w1 -= w0 % dx  # algin w0 with grids
    l1 = 0.4
    h1 = 0.7
    start_x = 2 * dx
    start_y = max_y / 2 - w0 / 2  # startx
    start_y -= start_y % dx  # align starty_y with grids
    n_particles_per_direction_x = 2
    n_particles_per_direction_y = 2
    n_particles_per_direction_z = 2
    PPC = n_particles_per_direction_x * n_particles_per_direction_y * n_particles_per_direction_z  # particle per cell
    p_vol = dx * dx * dx / PPC
    s_dia = (3 / 4 * inv_pi * p_vol) ** (1/3)
    DEM_r = (p_vol * 3 / 4 * inv_pi) ** (1 / 3)
    rho_s = 1.25  # bulk density, t/m3
    rho_l = 1.00  # bulk density, t/m3
    mass_s = p_vol * rho_s
    mass_l = p_vol * rho_l
    rho_critical = rho_s
    FLIP_blending_coeff = FLIPcoeff  # coeff * FLIP + (1-coeff) * PIC
    APIC_flag = False
    # calculate k
    global k_s, DEM_fric_coef, DEM_contact_flag, C_n, C_s
    DEM_contact_flag = DEM_contact
    DEM_fric_coef = 0.5  # tangential friction angle
    xee_bar = 0.1 * DEM_r
    c = ti.sqrt(sand_E * (1 - sand_nu) / ((1 + sand_nu) * (1 - 2 * sand_nu) * rho_s))
    k_n_crit = (4 * mass_s * pi ** 2 * c ** 2 / dx ** 2)  # critical kn
    #c_l = ti.sqrt(sand_E * (1 - 0.5) / ((1 + 0.5) * (1 - 2 * 0.5) * rho_l))
    #k_n_crit = (4 * mass_s * pi ** 2 * c ** 2 / dx ** 2)
    k_s = k_n_crit * 0 / 4  # tangential stiffness
    potential_coeff_k = (-k_n_crit * 1 / (
                2 * ti.log(xee_bar / DEM_r) + (xee_bar - DEM_r) * (3 * xee_bar + DEM_r) / xee_bar ** 2))
    CFL_coeff = 0.05
    C_n = 0.6
    # C_s = 0.6
    # DEM parameters
    if DEM_contact_flag:
        DEMx = ti.field(dtype=float, shape=DEMlist[:, 1].shape[0])
        DEMy = ti.field(dtype=float, shape=DEMlist[:, 1].shape[0])
        DEMx.from_numpy(DEMlist[:, 1])
        DEMy.from_numpy(DEMlist[:, 0])
    # Define time step
    if timestep is None:
        dt_CFL1 = dx / (c)
        dt_CFL2 = 2 * pi * ti.sqrt(mass_s / k_n_crit)
        dt = CFL_coeff * min(dt_CFL1, dt_CFL2)
    else:
        dt = timestep  # time step dt
    inv_dt = 1 / dt
    timelimit = 0.8
    calmstep = 0.1 / dt
    # Sand particles quantities
    global x_s, x_s_0, n_s, n_s_0, x_s_plot, v_s, Affine_C_s, DEM_force, sig_principle_s, sig_shear_s, beta_in_bd_flag, grid_delta_J \
        , grid_total_old_J, grid_weight_sum, DEM_shear_force, s_FLIP_a, s_If_a, s_DEMres_a, DEM_force_deuction_flag, s_rho_gasification, localFlipBlendingCoeff
    x_s = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)

    n_s = ti.field(dtype=float, shape=max_num_s_particles)  # sand fraction
    n_s_0 = ti.field(dtype=float, shape=max_num_s_particles)  # sand initial fraction

    x_s_0 = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # sand initial position
    x_s_plot = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # sand position for drawing
    v_s = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # sand velocity
    Affine_C_s = ti.Matrix.field(dim, dim, dtype=float,
                                 shape=max_num_s_particles)  # Sand affine velocity C matrix, see section 5.3 in [Jiang et al., APIC, 2015] or section 2.2 in [Jiang et al. JCP, 2017] for the definition.
    DEM_force = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # the force given by the DEM
    DEM_force_deuction_flag = ti.field(dtype=float,
                                       shape=max_num_s_particles)  # document if the particle overlaps with DEM
    localFlipBlendingCoeff = ti.field(dtype=float, shape=max_num_s_particles)  # document if the particles are gasified

    DEM_shear_force = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # the shear force by the DEM
    sig_principle_s = ti.Vector.field(3, dtype=float,
                                      shape=max_num_s_particles)  # sand stress sigma 11 sigma 22 sigma 33
    sig_shear_s = ti.Vector.field(3, dtype=float, shape=max_num_s_particles)  # sand stress sigma 12 sigma 13 sigma 23
    beta_in_bd_flag = ti.field(dtype=float,
                               shape=max_num_s_particles)  # check if the particle is in boundary so that the beta can be updated during G2P
    s_FLIP_a = ti.Vector.field(3, dtype=float,
                               shape=max_num_s_particles)  # sand equivalent acceleration projected back from FLIP
    s_If_a = ti.Vector.field(3, dtype=float,
                             shape=max_num_s_particles)  # sand equivalent internal acceleration projected back from the grid
    s_DEMres_a = ti.Vector.field(3, dtype=float,
                                 shape=max_num_s_particles)  # sand resultant acceleration due to the DEM force
    s_rho_gasification = ti.Vector.field(2, dtype=float,
                                         shape=max_num_s_particles)  # trace the rho (first), and gasification (second)

    global x_l, x_l_0, x_l_plot, v_l, F_l, pore_3D, Affine_C_l, DEM_force_l, sig_principle_l, sig_shear_l, beta_in_bd_flag_l, grid_delta_J_l \
        , grid_total_old_J_l, grid_weight_sum_l, DEM_shear_force_l, l_FLIP_a, l_If_a, l_DEMres_a, DEM_force_deuction_flag_l, l_rho_gasification
    x_l = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # liquid position
    x_l_0 = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # liquid initial position
    x_l_plot = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # liquid position for drawing
    v_l = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # liquid velocity
    F_l = ti.Matrix.field(dim, dim, dtype=float, shape=max_num_s_particles)  # liquid gradient deformation
    pore_3D = ti.field(dtype=float, shape=max_num_s_particles)  # liquid pore pressure
    Affine_C_l = ti.Matrix.field(dim, dim, dtype=float,
                                 shape=max_num_s_particles)  # Sand affine velocity C matrix, see section 5.3 in [Jiang et al., APIC, 2015] or section 2.2 in [Jiang et al. JCP, 2017] for the definition.
    DEM_force_l = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # the force given by the DEM
    DEM_force_deuction_flag_l = ti.field(dtype=float,
                                       shape=max_num_s_particles)  # document if the particle overlaps with DEM
    # localFlipBlendingCoeff = ti.field(dtype=float, shape=max_num_s_particles)  # document if the particles are gasified

    DEM_shear_force_l = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)  # the shear force by the DEM
    sig_principle_l = ti.Vector.field(3, dtype=float,
                                      shape=max_num_s_particles)  # isotropic pore pressure sigma 11 sigma 22 sigma 33
    sig_shear_l = ti.Vector.field(3, dtype=float, shape=max_num_s_particles)  # deviatoric pore pressure sigma 12 sigma 13 sigma 23
    beta_in_bd_flag_l = ti.field(dtype=float,
                               shape=max_num_s_particles)  # check if the particle is in boundary so that the beta can be updated during G2P
    l_FLIP_a = ti.Vector.field(3, dtype=float,
                               shape=max_num_s_particles)  # liquid equivalent acceleration projected back from FLIP
    l_If_a = ti.Vector.field(3, dtype=float,
                             shape=max_num_s_particles)  # liquid equivalent internal acceleration projected back from the grid
    l_DEMres_a = ti.Vector.field(3, dtype=float,
                                 shape=max_num_s_particles)  # liquid resultant acceleration due to the DEM force
    l_rho_gasification = ti.Vector.field(2, dtype=float,
                                         shape=max_num_s_particles)  # trace the rho (first), and gasification (second)
    # tracking particle flag
    global tracking_particle_flag
    tracking_particle_flag = ti.field(dtype=int, shape=max_num_s_particles)
    # Energy
    global g_potential, k_energy, e_potential
    g_potential = ti.field(ti.f32, shape=())
    k_energy = ti.field(ti.f32, shape=())
    e_potential = ti.field(ti.f32, shape=())

    # Sand grids quantities
    global grid_sv, grid_s_old_v, grid_sm, grid_sf, grid_if, grid_DEMf, grid_TopPrj, grid_TopMap, grid_phi_s, grid_weight, grid_drag
    grid_sv = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid sand momentum/velocity
    grid_s_old_v = ti.Vector.field(dim, dtype=float, shape=(
    n_grid_x, n_grid_z, n_grid_y))  # grid sand old momentum/velocity, used in FLIP scheme
    grid_sm = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y)) # grid sand mass
    grid_phi_s = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid solid fraction
    grid_weight = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid solid fraction weight sum
    grid_drag = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid drag force
    grid_sf = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid sand force
    grid_if = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid internal sand force
    grid_DEMf = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid DEM force
    grid_TopPrj = ti.field(dtype=int, shape=(n_grid_x, n_grid_y))  # grid sand mass
    grid_TopMap = ti.field(dtype=int, shape=(n_grid_x - iContainerOffset, n_grid_y))
    # nodal projction quantities
    grid_delta_J = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))
    grid_total_old_J = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))
    grid_weight_sum = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))

    global grid_lv, grid_l_old_v, grid_lm, grid_lf, grid_if_l, grid_DEMf_l, grid_TopPrj_l, grid_TopMap_l, grid_k, grid_func, grid_Re, cell_pore_3D, cell_particle_count
    grid_lv = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid liquid momentum/velocity
    grid_l_old_v = ti.Vector.field(dim, dtype=float, shape=(
        n_grid_x, n_grid_z, n_grid_y))  # grid liquid old momentum/velocity, used in FLIP scheme
    grid_lm = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid liquid mass
    #grid_k = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid permeability
    grid_func = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid function for drag force calculation
    grid_Re = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid Reynolds number
    grid_lf = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid liquid force
    grid_if_l = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid internal liquid force
    grid_DEMf_l = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))  # grid DEM liquid force
    grid_TopPrj_l = ti.field(dtype=int, shape=(n_grid_x, n_grid_y))  # grid sand mass
    grid_TopMap_l = ti.field(dtype=int, shape=(n_grid_x - iContainerOffset, n_grid_y))
    # nodal projction quantities
    grid_delta_J_l = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))
    grid_total_old_J_l = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))
    grid_weight_sum_l = ti.field(dtype=float, shape=(n_grid_x, n_grid_z, n_grid_y))
    cell_pore_3D = ti.Matrix.field(dim, dim, dtype=float, shape=(n_grid_x - 1, n_grid_z - 1, n_grid_y - 1))
    cell_particle_count = ti.field(dtype=float, shape=(n_grid_x - 1, n_grid_z - 1, n_grid_y - 1))
    # DEM position
    global sand_material, my_DEM_pillar
    # material
    sand_material = DruckerPragerRheology(n_particles=max_num_s_particles, dim=3, E=float(sand_E),
                                          nu=float(sand_nu), friction_angle=friction_angle, cohesion=0.0, mu_2=mu_2,
                                          xi=xifactor * ti.sqrt(mass_s * 1000), implicit_rheology_flag=True)
    columnR = columnR
    # setup DEM
    if DEM_contact_flag:
        my_DEM_pillar = DemPB3D(DEMx.shape[0] + 5)
        DEM_config_(columnR)
    else:
        my_DEM_pillar = None

    global color_map, color_s, scale_factor
    color_s = ti.field(dtype=int, shape=max_num_s_particles)
    scale_factor = 1.0 / max_x * 3.0  # for ratio = 1.0
    # halt
    global out_of_domain, out_of_domain_mass, out_of_domain_l, out_of_domain_mass_l
    out_of_domain = ti.field(ti.f32, shape=())
    out_of_domain_mass = ti.field(ti.f32, shape=())
    out_of_domain_l = ti.field(ti.f32, shape=())
    out_of_domain_mass_l = ti.field(ti.f32, shape=())
    # debug
    global PICV, stresslist, gradvlist, dFlist
    PICV = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles)
    stresslist = ti.Matrix.field(dim, dim, dtype=float,
                                 shape=max_num_s_particles)  # Sand affine velocity C matrix, see section 5.3 in [Jiang et al., APIC, 2015] or section 2.2 in [Jiang et al. JCP, 2017] for the definition.
    gradvlist = ti.Matrix.field(dim, dim, dtype=float, shape=max_num_s_particles)
    dFlist = ti.Matrix.field(dim, dim, dtype=float, shape=max_num_s_particles)


def fetchinfo():
    return x_s, v_s, dt, inv_dt, sig_principle_s, sig_shear_s, DEM_force, timelimit, \
        grid_sm, grid_sv, PICV, out_of_domain, s_FLIP_a, s_If_a, s_DEMres_a, s_rho_gasification, my_DEM_pillar
def fetchinfo_l():
    return x_l, v_l, dt, inv_dt, sig_principle_l, sig_shear_l, DEM_force, timelimit, \
        grid_lm, grid_lv, PICV, out_of_domain, l_FLIP_a, l_If_a, l_DEMres_a, l_rho_gasification, my_DEM_pillar

def FetchTopViewContour_PY():  # py scope
    FetchTopViewContour_TI()
    return grid_TopMap
    FetchTopViewContour_TI_l()
    return grid_TopMap_l



@ti.kernel
def DEM_config_(columnR: float):  # generated via ChatGPT, modified by Zhengyu Liang
    print("config DEM...")
    # set pillars
    ti.loop_config(serialize=True)
    scalex = 0.02 / 2
    scaley = 0.02 / 2
    scalez = 0.02 / 2
    vol = scalex * 2 * scaley * 2 * scalez * 2
    rho = 8.9  # t / m^3
    mass = vol * rho  # t
    # ti.loop_config(serialize=True)
    n_pillar = DEMx.shape[0]
    for i in range(n_pillar):
        pos = ti.Vector([DEMx[i] + start_x, DEMy[i] + 2 * dx, start_y + w0 / 2])
        # print("pos", pos)
        my_DEM_pillar.sf_mass[i] = mass
        my_DEM_pillar.set_model_pos_quat(i, ti.Vector([scalex, scaley, scalez]), pos, ti.Vector([0.0, 0.0, 0.0, 1.0]))
        my_DEM_pillar.sf_type[i] = 1

    # set floor
    pos_floor = ti.Vector([1.44, -0.1 + 2 * dx, start_y + w0 / 2])
    my_DEM_pillar.set_model_pos_quat(n_pillar, ti.Vector([1.5, 0.1, 1.5]), pos_floor, ti.Vector([0.0, 0.0, 0.0, 1.0]))
    my_DEM_pillar.sf_type[n_pillar] = 1
    # set walls
    sidewall_x = ti.Vector([-0.1, 1.3 + 0.1, 0.55, 0.55])
    sidewall_z = ti.Vector([start_y + w0 / 2, start_y + w0 / 2, start_y - 0.1, start_y + w0 + 0.1])
    sidewall_sz_x = ti.Vector([0.1, 0.1, 1.3, 1.3])
    sidewall_sz_z = ti.Vector([0.2, 0.2, 0.1, 0.1])
    idx = 0
    ti.loop_config(serialize=True)
    for i in range(n_pillar + 1, my_DEM_pillar.n_pillar):
        pos_floor = ti.Vector([sidewall_x[idx], 0.5, sidewall_z[idx]])
        scale_vec = ti.Vector([sidewall_sz_x[idx], 0.5, sidewall_sz_z[idx]])
        my_DEM_pillar.set_model_pos_quat(i, scale_vec, pos_floor, ti.Vector([0.0, 0.0, 0.0, 1.0]))
        my_DEM_pillar.sf_type[i] = 1
        idx += 1
    print("DEM config finished...")


@ti.kernel
def FetchTopViewContour_TI():  # ti scope
    for p in range(n_s_particles[None]):
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        if base[0] - iContainerOffset >= 0:
            grid_TopMap[base[0] - iContainerOffset, base[2]] = 1

@ti.kernel
def FetchTopViewContour_TI_l():  # ti scope
    for p in range(n_l_particles[None]):
        base = (x_l[p] * inv_dx - 0.5).cast(int)
        if base[0] - iContainerOffset >= 0:
            grid_TopMap_l[base[0] - iContainerOffset, base[2]] = 1

# ====================================================
# ============ I/0 FUNCTION ============
# write info to files (python scope)
def info2txt(filename, infolist):
    f = open(filename, 'w+')
    f.write('pos_sx pos_sz pos_sy pos_lx pos_lz pos_ly vel_sx vel_sz vel_sy vel_lx vel_lz vel_ly sig11 sig22 sig33 sig12 sig13 sig23 sigl11 sigl22 sigl33 sigl12 sigl13 sigl23\n')
    for info in infolist:
        for i in info:
            f.write('%f ' % i)
        f.write('\n')
    f.close()

def info2txt_l(filename, infolist):
    f = open(filename, 'w+')
    f.write('pos_x pos_z pos_y vel_x vel_z vel_y sig11 sig22 sig33 sig12 sig13 sig23\n')
    for info in infolist:
        for i in info:
            f.write('%f ' % i)
        f.write('\n')
    f.close()


# ============ LOAD STEP FUNCTION ============
def load_step(total_step):
    reset_grid(total_step)
    P2G()
    Phase_Couple()
    BCs_old_v(total_step)
    Momentum_exchange()
    BCs_v(total_step)
    G2P()
    G2P_l()


# ============ Solver kernels ============
@ti.kernel
def reset_grid(step: ti.i32):
    gravity[None] = ti.min(calmstep, step) / calmstep * standard_gravity  # add gravity in 500 steps
    # reset grid quantities
    for i, j, k in grid_sm:
        # 3D
        grid_sv[i, j, k] = [0, 0, 0]
        grid_s_old_v[i, j, k] = [0, 0, 0]
        grid_sm[i, j, k] = 0
        grid_phi_s[i, j, k] = 0
        grid_weight[i, j, k] = 0
        grid_sf[i, j, k] = [0, 0, 0]
        grid_if[i, j, k] = [0, 0, 0]
        grid_DEMf[i, j, k] = [0, 0, 0]

        grid_delta_J[i, j, k] = 0.0
        grid_total_old_J[i, j, k] = 0.0
        grid_weight_sum[i, j, k] = 0.0
        out_of_domain_mass[None] = 0.0
    for i, j, k in grid_lm:
        # 3D
        grid_lv[i, j, k] = [0, 0, 0]
        grid_l_old_v[i, j, k] = [0, 0, 0]
        grid_Re[i, j, k] = 0
        grid_func[i, j, k] = 0
        grid_lm[i, j, k] = 0
        grid_lf[i, j, k] = [0, 0, 0]
        grid_if_l[i, j, k] = [0, 0, 0]
        grid_DEMf_l[i, j, k] = [0, 0, 0]

        grid_delta_J_l[i, j, k] = 0.0
        grid_total_old_J_l[i, j, k] = 0.0
        grid_weight_sum_l[i, j, k] = 0.0
        out_of_domain_mass_l[None] = 0.0
    for i, j, k in cell_pore_3D:
        cell_pore_3D [i, j, k] = ti.Matrix.zero(float, 3, 3)
        cell_particle_count[i, j, k] = 0.0
    for i, j in grid_TopMap:
        grid_TopMap[i, j] = 0
        grid_TopMap_l[i, j] = 0
    if DEM_contact_flag:
        # DEM time integration (frozen for now)
        for i in range(my_DEM_pillar.n_pillar):
            my_DEM_pillar.reset_status(i)


@ti.kernel
def P2G():
    # P2G
    for p in range(n_s_particles[None]):
        base = (x_s[p] * inv_dx - 0.5).cast(int)  # get grid index
        fx = x_s[p] * inv_dx - base.cast(float)  # get reference coordinate, fx belongs to [0.5, 1.5]
        # Quadratic B-spline kernels, Eq. (123) in [MPM SIGGRAPH 2016 course]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        grad_w = [(fx - 1.5) * inv_dx, (2 - 2 * fx) * inv_dx, (fx - 0.5) * inv_dx]

        # get stress
        # stress_3D = sand_material.get_Kirchhoff_stress(p) # get sand Kirchhoff stress (3D)
        stress_3D = sand_material.get_Cauchy_stress(p)  # get Cauchy stress (3D)

        # get affine
        affine = mass_s * Affine_C_s[p]

        # get particle current volume
        this_particle_new_J_no_bar = sand_material.get_J_no_bar(p)
        this_particle_new_volume = p_vol * this_particle_new_J_no_bar

        # Consider gasification (ref: [Dunatunga, Kamrin, JFM, 2015])
        this_particle_new_density = mass_s / this_particle_new_volume
        s_rho_gasification[p][0] = this_particle_new_density
        s_rho_gasification[p][1] = 0.0

        if (this_particle_new_density < rho_critical):  # gassified
            # f_DEM_to_p = ti.Vector.zero(float, dim)
            stress_3D = ti.Matrix.zero(float, dim, dim)
            s_rho_gasification[p][1] = 1.0
            localFlipBlendingCoeff[p] = FLIP_blending_coeff
        else:
            localFlipBlendingCoeff[p] = 1.0

        # stresslist[p] = stress_3D # debug
        sig_principle_s[p] = ti.Vector(
            [stress_3D[0, 0], stress_3D[1, 1], stress_3D[2, 2]])  # sand stress sigma 11 sigma 22 sigma 33 (principle)
        sig_shear_s[p] = ti.Vector(
            [stress_3D[0, 1], stress_3D[0, 2], stress_3D[1, 2]])  # sand stress sigma 12 sigma 13 sigma 23

        # get DEM force
        #f_DEM_to_p = DEM_force[p]

        # execute P2G - sand
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx  # difference between grid node position and particle position
            weight = w[i][0] * w[j][1] * w[k][2]
            weight_grad = ti.Vector(
                [grad_w[i][0] * w[j][1] * w[k][2], w[i][0] * grad_w[j][1] * w[k][2], w[i][0] * w[j][1] * grad_w[k][2]])
            if APIC_flag:
                grid_sv[base + offset] += weight * (mass_s * v_s[p] + affine @ dpos)
            else:
                grid_sv[base + offset] += weight * (mass_s * v_s[p])
            grid_phi_s[base + offset] += weight * (1 - n_s[p])
            grid_weight[base + offset] += weight
            grid_sm[base + offset] += weight * mass_s
            # grid_sf[base + offset] += -p_vol * stress_3D @ weight_grad
            grid_sf[base + offset] += -this_particle_new_volume * stress_3D @ weight_grad
            grid_if[
                base + offset] += -this_particle_new_volume * stress_3D @ weight_grad  # record only the force induced by stress

            #grid_sf[base + offset] += weight * f_DEM_to_p
            #grid_DEMf[base + offset] += weight * f_DEM_to_p  # record only the grid DEM force

        # reset DEM force
        #DEM_force[p] = ti.Vector([0.0, 0.0, 0.0])

        ## check if the particle is in bd ##
        beta_in_bd_flag[p] = 0  # init
        # floor
        if x_s[p][1] < 3 * dx and v_s[p][1] < 0:          beta_in_bd_flag[p] = 1
        # roller BCs for the back wall of container
        if x_s[p][0] <= start_x:                        beta_in_bd_flag[p] = 1
        # side walls
        if (x_s[p][2] <= start_y + 0.5 * dx or x_s[p][2] >= start_y + w0 - 0.5 * dx):
            # back wall
            if x_s[p][2] <= start_y + 0.5 * dx and v_s[p][2] <= 0:
                beta_in_bd_flag[p] = 1
                # front wall
            elif x_s[p][2] >= start_y + w0 - 0.5 * dx and v_s[p][2] >= 0:
                beta_in_bd_flag[p] = 1
        # the gate
        # if (x_s[p][0] >= 0.5 and x_s[p][0] <= 0.55) and x_s[p][1] >= 0.21:
        #     # hit from inside
        #     if x_s[p][0] >= 0.5 and v_s[p][0] >= 0:
        #         beta_in_bd_flag[p] = 1
        #         # hit form outside
        #     elif x_s[p][0] <= 0.55 and v_s[p][0] <= 0:
        #         beta_in_bd_flag[p] = 1

    for p in range(n_l_particles[None]):
        base = (x_l[p] * inv_dx - 0.5).cast(int)  # get grid index
        fx = x_l[p] * inv_dx - base.cast(float)  # get reference coordinate, fx belongs to [0.5, 1.5]
        # Quadratic B-spline kernels, Eq. (123) in [MPM SIGGRAPH 2016 course]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        grad_w = [(fx - 1.5) * inv_dx, (2 - 2 * fx) * inv_dx, (fx - 0.5) * inv_dx]

        # get stress
        # stress_3D = sand_material.get_Kirchhoff_stress(p) # get sand Kirchhoff stress (3D)
        #pore_3D = sand_material.get_Cauchy_stress(p)  # get Cauchy stress (3D)

        # get affine
        affine = mass_l * Affine_C_l[p]

        # get particle current volume
        #this_particle_new_J_no_bar = sand_material.get_J_no_bar(p)
        this_particle_new_volume = p_vol

        # Consider gasification (ref: [Dunatunga, Kamrin, JFM, 2015])
        # this_particle_new_density = mass_l / this_particle_new_volume
        # s_rho_gasification[p][0] = this_particle_new_density
        # s_rho_gasification[p][1] = 0.0
        #
        # if (this_particle_new_density < rho_critical):  # gassified
        #     # f_DEM_to_p = ti.Vector.zero(float, dim)
        #     stress_3D = ti.Matrix.zero(float, dim, dim)
        #     s_rho_gasification[p][1] = 1.0
        #     localFlipBlendingCoeff[p] = FLIP_blending_coeff
        # else:
        #     localFlipBlendingCoeff[p] = 1.0
        localFlipBlendingCoeff[p] = 1.0
        # stresslist[p] = stress_3D # debug
        # sig_principle_l[p] = ti.Vector(
        #     [pore_3D[p][0, 0], pore_3D[p][1, 1], pore_3D[p][2, 2]])  # sand stress sigma 11 sigma 22 sigma 33 (principle)
        # sig_shear_l[p] = ti.Vector(
        #     [pore_3D[p][0, 1], pore_3D[p][0, 2], pore_3D[p][1, 2]])

        # get DEM force
        #f_DEM_to_p = DEM_force_l[p]

        # execute P2G - liquid
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx  # difference between grid node position and particle position
            weight = w[i][0] * w[j][1] * w[k][2]
            weight_grad = ti.Vector(
                [grad_w[i][0] * w[j][1] * w[k][2], w[i][0] * grad_w[j][1] * w[k][2], w[i][0] * w[j][1] * grad_w[k][2]])
            if APIC_flag:
                grid_lv[base + offset] += weight * (mass_s * v_l[p] + affine @ dpos)
            else:
                grid_lv[base + offset] += weight * (mass_s * v_l[p])
            grid_lm[base + offset] += weight * mass_l
            # grid_lf[base + offset] += -p_vol * pore_3D @ weight_grad

            grid_if_l[
                base + offset] += -this_particle_new_volume * pore_3D[p] * weight_grad  # record only the force induced by stress

            #grid_lf[base + offset] += weight * f_DEM_to_p
            #grid_DEMf_l[base + offset] += weight * f_DEM_to_p  # record only the grid DEM force

        # reset DEM force
        #DEM_force_l[p] = ti.Vector([0.0, 0.0, 0.0])

        ## check if the particle is in bd ##
        beta_in_bd_flag_l[p] = 0  # init
        # floor
        if x_l[p][1] < 3 * dx and v_l[p][1] < 0:          beta_in_bd_flag_l[p] = 1
        # roller BCs for the back wall of container
        if x_l[p][0] <= start_x:                        beta_in_bd_flag_l[p] = 1
        # side walls
        if (x_l[p][2] <= start_y + 0.5 * dx or x_l[p][2] >= start_y + w0 - 0.5 * dx):
            # back wall
            if x_l[p][2] <= start_y + 0.5 * dx and v_l[p][2] <= 0:
                beta_in_bd_flag_l[p] = 1
                # front wall
            elif x_l[p][2] >= start_y + w0 - 0.5 * dx and v_l[p][2] >= 0:
                beta_in_bd_flag_l[p] = 1
        # the gate
        # if (x_l[p][0] >= 0.5 and x_l[p][0] <= 0.55) and x_l[p][1] >= 0.21:
        #     # hit from inside
        #     if x_l[p][0] >= 0.5 and v_l[p][0] >= 0:
        #         beta_in_bd_flag[p] = 1
        #         # hit form outside
        #     elif x_l[p][0] <= 0.55 and v_l[p][0] <= 0:
        #         beta_in_bd_flag[p] = 1

    # Momentum to velocity
    for i, j, k in grid_sm:
        if grid_sm[i, j, k] > 0:
            grid_sv[i, j, k] = (1 / grid_sm[i, j, k]) * grid_sv[i, j, k]  # Momentum to velocity

            grid_s_old_v[i, j, k] = grid_sv[i, j, k]

    for i, j, k in grid_lm:
        if grid_lm[i, j, k] > 0:
            grid_lv[i, j, k] = (1 / grid_lm[i, j, k]) * grid_lv[i, j, k]  # Momentum to velocity

            grid_l_old_v[i, j, k] = grid_lv[i, j, k]

@ti.kernal
def Phase_Couple():
    for i, j, k in grid_sm:
        grid_phi_s[i, j, k] = grid_phi_s[i, j, k] * (1 / grid_weight[i, j, k])
        grid_Re[i, j, k] = (1 - grid_phi_s[i ,j ,k]) * rho_l * s_dia * 1000 * (grid_sv[i, j ,k] - grid_lv[i, j, k]).norm()
        grid_func[i, j, k] = 10 * grid_phi_s[i ,j ,k] / ((1 - grid_phi_s[i ,j ,k]) ** 2)
        grid_func[i, j, k] += ((1 - grid_phi_s[i ,j ,k]) ** 2) * (1 + 1.5 * (grid_phi_s[i ,j ,k] ** 0.5))
        temp_co1 = 0.413 * grid_Re[i, j, k] / (((1 - grid_phi_s[i ,j ,k]) ** 2) * 24)
        temp_co2 = 1 / (1 - grid_phi_s[i ,j ,k]) + 3 * (1 - grid_phi_s[i ,j ,k]) * grid_phi_s[i ,j ,k] + 8.4 * grid_Re[i, j, k] ** -0.343
        temp_co3 = 1 + (10 ** (grid_phi_s[i, j, k] * 3)) / (grid_Re[i, j, k] ** (0.5 + 2 * grid_phi_s[i ,j ,k]))
        grid_func[i, j, k] += temp_co1 * temp_co2 / temp_co3
        # grid_func[i, j, k] += grid_Re[i, j, k] * (0.03365 * (1 - grid_phi_s[i ,j ,k]) + 0.106 * grid_phi_s[i ,j ,k] * (1 - grid_phi_s[i ,j ,k]))
        # grid_func[i, j, k] += grid_Re[i, j, k] * (0.0116 / ((1 - grid_phi_s[i ,j ,k]) ** 4))
        # grid_func[i, j, k] += (grid_phi_s[i ,j ,k] * (6 - 10 * grid_phi_s[i ,j ,k])) / ((1 - grid_phi_s[i ,j ,k]) ** 2)
        grid_drag[i, j, k] = 18.0 * grid_phi_s[i, j, k] * (1 - grid_phi_s[i, j, k]) * grid_func[i, j, k] / (
                    s_dia ** 2) * 1e-3 * (grid_sv[i, j, k] - grid_lv[i, j, k])
        grid_sf[i, j, k] += grid_if_l[i, j, k] * grid_phi_s[i, j, k]
        grid_sf[i, j, k] += -grid_drag[i, j, k]
        grid_lf[i, j, k] += grid_if_l[i, j, k] * (1 - grid_phi_s[i, j, k])
        grid_lf[i, j, k] += grid_drag[i, j, k]


@ti.kernel
def BCs_old_v(step: ti.i32):
    # Friction and BCs for old v
    for i, j, k in grid_sm:
        # Add boundary friction for sand old v
        boundary_normal = ti.Vector.zero(float, dim)
        if grid_sm[i, j, k] > 0:
            if j < 3 and grid_s_old_v[i, j, k][1] < 0:          boundary_normal = ti.Vector([0, 1, 0])
        if boundary_normal[1] != 0:
            v_normal_mag = grid_s_old_v[i, j, k].dot(boundary_normal)
            v_normal = v_normal_mag * boundary_normal
            v_tangent = grid_s_old_v[i, j, k] - v_normal
            v_tangent_norm = v_tangent.norm()
            if v_tangent_norm > tol:
                # Coulomb friction
                if v_tangent_norm < abs(boundary_friction_coeff * v_normal_mag):
                    grid_s_old_v[i, j, k] = ti.Vector([0, 0, 0])
                else:
                    grid_s_old_v[i, j, k] = v_tangent
                    grid_s_old_v[i, j, k] -= abs(boundary_friction_coeff * v_normal_mag) * (v_tangent / v_tangent_norm)

    for i, j, k in grid_sm:
        boundary_normal = ti.Vector.zero(float, dim)
        # BCs (change normal v to zero)
        if grid_sm[i, j, k] > 0:
            if i * dx <= start_x:                                              grid_s_old_v[i, j, k][
                0] = 0  # roller BCs for the back wall of container
            # side walls
            if (k * dx <= start_y + 0.5 * dx or k * dx >= start_y + w0 - 0.5 * dx):
                # back wall
                if k * dx <= start_y + 0.5 * dx:
                    boundary_normal = ti.Vector([0, 0, 1])
                # front wall
                else:
                    boundary_normal = ti.Vector([0, 0, -1])
                v_normal_mag = grid_s_old_v[i, j, k].dot(boundary_normal)
                v_normal = v_normal_mag * boundary_normal
                v_tangent = grid_s_old_v[i, j, k] - v_normal
                v_tangent_norm = v_tangent.norm()
                if v_tangent_norm > tol:
                    # Coulomb friction
                    if v_tangent_norm < abs(boundary_friction_coeff_side * v_normal_mag):
                        grid_s_old_v[i, j, k] = ti.Vector([0, 0, 0])
                    else:
                        grid_s_old_v[i, j, k] = v_tangent
                        grid_s_old_v[i, j, k] -= abs(boundary_friction_coeff_side * v_normal_mag) * (
                                    v_tangent / v_tangent_norm)
            if (k * dx <= start_y + 0.5 * dx or k * dx >= start_y + w0 - 0.5 * dx):
                grid_s_old_v[i, j, k][2] = 0  # the side walls of the container (make sure there is no component)

            if j < 3 and grid_s_old_v[i, j, k][1] < 0:                       grid_s_old_v[i, j, k][1] = 0  # the floor
            if j >= n_grid_z - 4 and grid_s_old_v[i, j, k][1] > 0:             grid_s_old_v[i, j, k][
                1] = 0  # the ceiling

            # if (i * dx >= 0.5 and i * dx <= 0.55) and j * dx >= 0.21:  # the gate
            #     grid_s_old_v[i, j, k][0] = 0

            if (i * dx >= start_x + l0) and (step < calmstep):                      grid_s_old_v[i, j, k][
                0] = 0  # Add initial BCs for the box

            # out of domain #
            if (i >= n_grid_x - 3) or (k >= n_grid_y - 3) or (k == 3) or (j >= n_grid_z - 3):

                grid_s_old_v[i, j, k][0] = 0
                grid_s_old_v[i, j, k][1] = 0
                grid_s_old_v[i, j, k][2] = 0
                out_of_domain_mass[None] += grid_sm[i, j, k]
                if out_of_domain_mass[None] > 0.0001:
                    out_of_domain[None] = 1

    for i, j, k in grid_lm:
        # Add boundary friction for liquid old v
        boundary_normal = ti.Vector.zero(float, dim)
        if grid_lm[i, j, k] > 0:
            if j < 3 and grid_l_old_v[i, j, k][1] < 0:          boundary_normal = ti.Vector([0, 1, 0])
        if boundary_normal[1] != 0:
            v_normal_mag = grid_l_old_v[i, j, k].dot(boundary_normal)
            v_normal = v_normal_mag * boundary_normal
            if v_normal_mag < 0:
                v_tangent = grid_l_old_v[i, j, k] - v_normal
                grid_l_old_v[i, j, k] = v_tangent

            #if v_tangent_norm > tol:

                # Coulomb friction
                # if v_tangent_norm < abs(boundary_friction_coeff * v_normal_mag):
                #     grid_l_old_v[i, j, k] = ti.Vector([0, 0, 0])
                # else:
                #     grid_l_old_v[i, j, k] = v_tangent
                #     grid_l_old_v[i, j, k] -= abs(boundary_friction_coeff * v_normal_mag) * (v_tangent / v_tangent_norm)

    for i, j, k in grid_lm:
        boundary_normal = ti.Vector.zero(float, dim)
        # BCs (change normal v to zero)
        if grid_lm[i, j, k] > 0:
            if i * dx <= start_x:                                              grid_l_old_v[i, j, k][
                0] = 0  # roller BCs for the back wall of container
            # side walls
            if (k * dx <= start_y + 0.5 * dx or k * dx >= start_y + w0 - 0.5 * dx):
                # back wall
                if k * dx <= start_y + 0.5 * dx:
                    boundary_normal = ti.Vector([0, 0, 1])
                # front wall
                else:
                    boundary_normal = ti.Vector([0, 0, -1])
                v_normal_mag = grid_l_old_v[i, j, k].dot(boundary_normal)
                v_normal = v_normal_mag * boundary_normal
                if v_normal_mag < 0:
                    v_tangent = grid_l_old_v[i, j, k] - v_normal
                    grid_l_old_v[i, j, k] = v_tangent
                # v_tangent_norm = v_tangent.norm()
                # if v_tangent_norm > tol:
                #     # Coulomb friction
                #     if v_tangent_norm < abs(boundary_friction_coeff_side * v_normal_mag):
                #         grid_l_old_v[i, j, k] = ti.Vector([0, 0, 0])
                #     else:
                #         grid_l_old_v[i, j, k] = v_tangent
                #         grid_l_old_v[i, j, k] -= abs(boundary_friction_coeff_side * v_normal_mag) * (
                #                     v_tangent / v_tangent_norm)
            if (k * dx <= start_y + 0.5 * dx or k * dx >= start_y + w0 - 0.5 * dx):
                grid_l_old_v[i, j, k][2] = 0  # the side walls of the container (make sure there is no component)

            if j < 3 and grid_l_old_v[i, j, k][1] < 0:                       grid_l_old_v[i, j, k][1] = 0  # the floor
            if j >= n_grid_z - 4 and grid_l_old_v[i, j, k][1] > 0:             grid_l_old_v[i, j, k][
                1] = 0  # the ceiling

            # if (i * dx >= 0.5 and i * dx <= 0.55) and j * dx >= 0.21:  # the gate
            #     grid_l_old_v[i, j, k][0] = 0

            if (i * dx >= start_x + l0) and (step < calmstep):                      grid_l_old_v[i, j, k][
                0] = 0  # Add initial BCs for the box

            # out of domain #
            if (i >= n_grid_x - 3) or (k >= n_grid_y - 3) or (k == 3) or (j >= n_grid_z - 3):

                grid_l_old_v[i, j, k][0] = 0
                grid_l_old_v[i, j, k][1] = 0
                grid_l_old_v[i, j, k][2] = 0
                out_of_domain_mass[None] += grid_lm[i, j, k]
                if out_of_domain_mass[None] > 0.0001:
                    out_of_domain[None] = 1


@ti.kernel
def Momentum_exchange():
    # Explicit solver
    for i, j, k in grid_sm:
        # Velocity update
        if grid_sm[i, j, k] > 0:
            delta_sv = dt * (gravity[None] + grid_sf[i, j, k] / grid_sm[i, j, k])
            grid_sv[i, j, k] += delta_sv
    for i, j, k in grid_lm:
        # Velocity update
        if grid_lm[i, j, k] > 0:
            delta_lv = dt * (gravity[None] + grid_lf[i, j, k] / grid_lm[i, j, k])
            grid_lv[i, j, k] += delta_lv


@ti.kernel
def BCs_v(step: ti.i32):
    # Friction and BCs for old v
    for i, j, k in grid_sm:
        # Add boundary friction for sand old v
        boundary_normal = ti.Vector.zero(float, dim)
        if grid_sm[i, j, k] > 0:
            if j < 3 and grid_sv[i, j, k][1] < 0:          boundary_normal = ti.Vector([0, 1, 0])
        if boundary_normal[1] != 0:
            v_normal_mag = grid_sv[i, j, k].dot(boundary_normal)
            v_normal = v_normal_mag * boundary_normal
            v_tangent = grid_sv[i, j, k] - v_normal
            v_tangent_norm = v_tangent.norm()
            if v_tangent_norm > tol:
                # Coulomb friction
                if v_tangent_norm < abs(boundary_friction_coeff * v_normal_mag):
                    grid_sv[i, j, k] = ti.Vector([0, 0, 0])
                else:
                    grid_sv[i, j, k] = v_tangent
                    grid_sv[i, j, k] -= abs(boundary_friction_coeff * v_normal_mag) * (v_tangent / v_tangent_norm)

    for i, j, k in grid_sm:
        # BCs (change normal v to zero)
        if grid_sm[i, j, k] > 0:
            boundary_normal = ti.Vector.zero(float, dim)
            if i * dx <= start_x:                                               grid_sv[i, j, k][
                0] = 0  # roller BCs for the back wall of container
            # side walls
            if (k * dx <= start_y + 0.5 * dx or k * dx >= start_y + w0 - 0.5 * dx):
                # back wall
                if k * dx <= start_y + 0.5 * dx:
                    boundary_normal = ti.Vector([0, 0, 1])
                # front wall
                else:
                    boundary_normal = ti.Vector([0, 0, -1])
                v_normal_mag = grid_sv[i, j, k].dot(boundary_normal)
                v_normal = v_normal_mag * boundary_normal
                v_tangent = grid_sv[i, j, k] - v_normal
                v_tangent_norm = v_tangent.norm()
                if v_tangent_norm > tol:
                    # Coulomb friction
                    if v_tangent_norm < abs(boundary_friction_coeff_side * v_normal_mag):
                        grid_sv[i, j, k] = ti.Vector([0, 0, 0])
                    else:
                        grid_sv[i, j, k] = v_tangent
                        grid_sv[i, j, k] -= abs(boundary_friction_coeff_side * v_normal_mag) * (
                                    v_tangent / v_tangent_norm)
            if (k * dx <= start_y + 0.5 * dx or k * dx >= start_y + w0 - 0.5 * dx):
                grid_sv[i, j, k][2] = 0  # the side walls of the container (make sure there is no component)

            if j < 3 and grid_sv[i, j, k][1] < 0:                             grid_sv[i, j, k][1] = 0  # the floor
            if j >= n_grid_z - 4 and grid_sv[i, j, k][1] > 0:             grid_sv[i, j, k][1] = 0  # the ceiling

            # if (i * dx >= 0.5 and i * dx <= 0.55) and j * dx >= 0.21:  # the gate
            #     grid_sv[i, j, k][0] = 0

            if (i * dx >= start_x + l0) and (step < calmstep):                       grid_sv[i, j, k][
                0] = 0  # Add initial BCs for the box
            # out of domain #
            if (i >= n_grid_x - 3) or (k >= n_grid_y - 3) or (k == 3) or (j >= n_grid_z - 3):
                grid_sv[i, j, k][0] = 0
                grid_sv[i, j, k][1] = 0
                grid_sv[i, j, k][2] = 0

    for i, j, k in grid_lm:
        # Add boundary friction for sand old v
        boundary_normal = ti.Vector.zero(float, dim)
        if grid_lm[i, j, k] > 0:
            if j < 3 and grid_lv[i, j, k][1] < 0:          boundary_normal = ti.Vector([0, 1, 0])
        if boundary_normal[1] != 0:
            v_normal_mag = grid_lv[i, j, k].dot(boundary_normal)
            v_normal = v_normal_mag * boundary_normal
            if v_normal_mag < 0:
                v_tangent = grid_lv[i, j, k] - v_normal
                grid_lv[i, j, k] = v_tangent
            # v_tangent_norm = v_tangent.norm()
            # if v_tangent_norm > tol:
            #     # Coulomb friction
            #     if v_tangent_norm < abs(boundary_friction_coeff * v_normal_mag):
            #         grid_lv[i, j, k] = ti.Vector([0, 0, 0])
            #     else:
            #         grid_lv[i, j, k] = v_tangent
            #         grid_lv[i, j, k] -= abs(boundary_friction_coeff * v_normal_mag) * (v_tangent / v_tangent_norm)

    for i, j, k in grid_lm:
        # BCs (change normal v to zero)
        if grid_lm[i, j, k] > 0:
            boundary_normal = ti.Vector.zero(float, dim)
            if i * dx <= start_x:                                               grid_lv[i, j, k][
                0] = 0  # roller BCs for the back wall of container
            # side walls
            if (k * dx <= start_y + 0.5 * dx or k * dx >= start_y + w0 - 0.5 * dx):
                # back wall
                if k * dx <= start_y + 0.5 * dx:
                    boundary_normal = ti.Vector([0, 0, 1])
                # front wall
                else:
                    boundary_normal = ti.Vector([0, 0, -1])
                v_normal_mag = grid_lv[i, j, k].dot(boundary_normal)
                v_normal = v_normal_mag * boundary_normal
                if v_normal_mag < 0:
                    v_tangent = grid_lv[i, j, k] - v_normal
                    grid_lv[i, j, k] = v_tangent
                # v_tangent_norm = v_tangent.norm()
                # if v_tangent_norm > tol:
                #     # Coulomb friction
                #     if v_tangent_norm < abs(boundary_friction_coeff_side * v_normal_mag):
                #         grid_lv[i, j, k] = ti.Vector([0, 0, 0])
                #     else:
                #         grid_lv[i, j, k] = v_tangent
                #         grid_lv[i, j, k] -= abs(boundary_friction_coeff_side * v_normal_mag) * (
                #                     v_tangent / v_tangent_norm)
            if (k * dx <= start_y + 0.5 * dx or k * dx >= start_y + w0 - 0.5 * dx):
                grid_lv[i, j, k][2] = 0  # the side walls of the container (make sure there is no component)

            if j < 3 and grid_lv[i, j, k][1] < 0:                             grid_lv[i, j, k][1] = 0  # the floor
            if j >= n_grid_z - 4 and grid_lv[i, j, k][1] > 0:             grid_lv[i, j, k][1] = 0  # the ceiling

            # if (i * dx >= 0.5 and i * dx <= 0.55) and j * dx >= 0.21:  # the gate
            #     grid_lv[i, j, k][0] = 0

            if (i * dx >= start_x + l0) and (step < calmstep):                       grid_lv[i, j, k][
                0] = 0  # Add initial BCs for the box
            # out of domain #
            if (i >= n_grid_x - 3) or (k >= n_grid_y - 3) or (k == 3) or (j >= n_grid_z - 3):
                grid_lv[i, j, k][0] = 0
                grid_lv[i, j, k][1] = 0
                grid_lv[i, j, k][2] = 0



@ti.kernel
def G2P():
    # G2P
    for p in range(n_s_particles[None]):
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        fx = x_s[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        grad_w = [(fx - 1.5) * inv_dx, (2 - 2 * fx) * inv_dx, (fx - 0.5) * inv_dx]
        FLIP_a = ti.Vector.zero(float, dim)  # reset FLIP a
        If_a = ti.Vector.zero(float, dim)  # reset If_a a
        DEMf_a = ti.Vector.zero(float, dim)  # reset DEMf_a
        new_FLIP_v = v_s[p]
        new_PIC_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)
        grad_v = ti.Matrix.zero(float, dim, dim)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            g_v = grid_sv[base + offset]
            g_old_v = grid_s_old_v[base + offset]
            weight = w[i][0] * w[j][1] * w[k][2]
            weight_grad = ti.Vector(
                [grad_w[i][0] * w[j][1] * w[k][2], w[i][0] * grad_w[j][1] * w[k][2], w[i][0] * w[j][1] * grad_w[k][2]])
            FLIP_a += weight * (g_v - g_old_v) * inv_dt  # calculate the FLIP_a from FLIP acceleration
            ##### grid_sv[i, j, k] += dt * (gravity + grid_sf[i, j, k] / grid_sm[i, j, k])
            If_a += weight * grid_if[base + offset] / grid_sm[base + offset]  # calculate the If_a from the grid
            #DEMf_a += weight * grid_DEMf[base + offset] / grid_sm[base + offset]  # calculate the DEMf_a from the grid
            new_FLIP_v += weight * (g_v - g_old_v)
            new_PIC_v += weight * g_v

            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx ** 2  # get new particle affine velocity matrix
            grad_v += g_v.outer_product(weight_grad)

        # new_FLIP_v -=  DEMf_a*dt*DEM_deduction_flag # remove DEM a if necessary

        delta_F = (ti.Matrix.identity(float, dim) + dt * grad_v)

        # Project delta_J to nodes
        delta_J = delta_F.determinant()

        this_particle_old_J = sand_material.get_J(p)
        sand_material.update_J_no_bar(delta_J, p)

        this_particle_new_J_no_bar = sand_material.get_J_no_bar(p)

        this_particle_new_volume = p_vol * this_particle_new_J_no_bar

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i][0] * w[j][1] * w[k][2]

            grid_weight_sum[base + offset] += weight * this_particle_new_volume

            grid_delta_J[base + offset] += weight * this_particle_new_volume * delta_J
            grid_total_old_J[base + offset] += weight * this_particle_new_volume * this_particle_old_J

        # particle advection
        if APIC_flag:
            v_s[p] = new_PIC_v
        else:
            v_s[p] = localFlipBlendingCoeff[p] * new_FLIP_v + \
                     (1 - localFlipBlendingCoeff[p]) * new_PIC_v
        Affine_C_s[p] = new_C
        s_FLIP_a[p] = FLIP_a
        s_If_a[p] = If_a
        #s_DEMres_a[p] = DEMf_a

    # Solve for grid J
    for i, j, k in grid_delta_J:
        this_grid_weight_sum = grid_weight_sum[i, j, k]
        if this_grid_weight_sum > 0:
            grid_delta_J[i, j, k] /= this_grid_weight_sum
            grid_total_old_J[i, j, k] /= this_grid_weight_sum

    # Particles loop to update material
    for p in range(n_s_particles[None]):
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        fx = x_s[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        grad_w = [(fx - 1.5) * inv_dx, (2 - 2 * fx) * inv_dx, (fx - 0.5) * inv_dx]

        new_PIC_v = ti.Vector.zero(float, dim)
        grad_v = ti.Matrix.zero(float, dim, dim)

        delta_J_bar = 0.0
        total_old_J_bar = 0.0

        this_particle_old_J = sand_material.get_J(p)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            weight = w[i][0] * w[j][1] * w[k][2]
            weight_grad = ti.Vector(
                [grad_w[i][0] * w[j][1] * w[k][2], w[i][0] * grad_w[j][1] * w[k][2], w[i][0] * w[j][1] * grad_w[k][2]])

            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            g_v = grid_sv[base + offset]

            g_delta_J = grid_delta_J[base + offset]
            g_total_old_J = grid_total_old_J[base + offset]

            new_PIC_v += g_v * weight
            grad_v += g_v.outer_product(weight_grad)

            delta_J_bar += weight * g_delta_J
            total_old_J_bar += weight * g_total_old_J

        delta_F = ti.Matrix.identity(float, dim) + dt * grad_v
        delta_J = delta_F.determinant()
        n_s[p] = 1 - ((1 - n_s_0) / delta_J)
        # Modified F-bar (consider gasification)
        this_particle_new_J_no_bar = sand_material.get_J_no_bar(p)
        this_particle_new_volume = p_vol * this_particle_new_J_no_bar
        this_particle_new_density = mass_s / this_particle_new_volume

        if (this_particle_new_density >= rho_critical):
            sand_material.update_deformation_gradient(
                (total_old_J_bar * delta_J_bar / (this_particle_old_J * delta_J)) ** (1.0 / 3.0) * delta_F, p, dt)

        # Particle advection
        x_s[p] = x_s[p] + dt * new_PIC_v

    # DEM coupling #
    if DEM_contact_flag:
        a = 1.0
        # DEM-DEM collision
        n_DEM = my_DEM_pillar.n_pillar
        n_pillar = n_DEM - 6
        kn_pp = k_n_crit / 10.0
        kn_pf = k_n_crit / 10.0  # kn for pillar-floor
        for i in range(n_pillar):
            # print("i: ", i)
            # ti.loop_config(serialize=True)
            for j in range(i + 1, n_DEM):
                if (my_DEM_pillar.sf_type[i] == 1 or my_DEM_pillar.sf_type[j] == 1):
                    contact, vec = my_DEM_pillar.gjk(i, j)
                    if (contact == 1.0):
                        # force computation for i
                        if (my_DEM_pillar.sf_type[i] == 1):  # only care about box
                            aci_n = my_DEM_pillar.get_contact_manifold_box(i, j)
                            if (my_DEM_pillar.sf_type[j] == 1):
                                my_DEM_pillar.contact_f_via_cur_act_cp_box(i, j, aci_n, vec, dt=dt, kn=kn_pf,
                                                                           knd=kn_pf * 0.1, ks=k_n_crit / 10, ksd=80,
                                                                           mu=0.65)
                        # force computation for j
                        if (my_DEM_pillar.sf_type[j] == 1):  # only care about box
                            acj_n = my_DEM_pillar.get_contact_manifold_box(j, i)
                            my_DEM_pillar.contact_f_via_cur_act_cp_box(j, i, acj_n, -vec, dt=dt, kn=kn_pp,
                                                                       knd=kn_pp * 0.1, ks=k_n_crit / 10, ksd=80,
                                                                       mu=0.65)
        # gravity
        for i in range(n_pillar):
            if (my_DEM_pillar.sf_type[i] == 1):
                my_DEM_pillar.v3f_fn[i] += gravity[None] * my_DEM_pillar.sf_mass[i]

        # MP-DEM coupling
        for p in range(n_s_particles[None]):
            can_contact = my_DEM_pillar.bounding_sphere_check(x_p=x_s[p], bs_r=0.05)
            if (can_contact):
                f_DEM_to_p_total = my_DEM_pillar.get_contact_force_MPM_potential(x_p=x_s[p],
                                                                                 v_p=v_s[p],
                                                                                 potential_coeff_k=potential_coeff_k,
                                                                                 r=DEM_r,
                                                                                 Cn=C_n,
                                                                                 m=mass_s,
                                                                                 targettype=1)
                DEM_force[p] = f_DEM_to_p_total
                DEM_shear_force[p] = ti.Vector.zero(float, dim)  # we do not compute shear force for now

        # DEM time integration (frozen for now)
        for i in range(n_pillar):
            if (my_DEM_pillar.sf_type[i] == 1):
                my_DEM_pillar.time_integration_semi_explicit(i, dt)
    else:
        for p in range(n_s_particles[None]):
            DEM_force[p] = ti.Vector.zero(float, dim)
            DEM_shear_force[p] = ti.Vector.zero(float, dim)

@ti.kernel
def G2P_l():
    # G2P

    for p in range(n_l_particles[None]):
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        fx = x_s[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        grad_w = [(fx - 1.5) * inv_dx, (2 - 2 * fx) * inv_dx, (fx - 0.5) * inv_dx]
        FLIP_a_l = ti.Vector.zero(float, dim)  # reset FLIP a
        If_a_l = ti.Vector.zero(float, dim)  # reset If_a a
        DEMf_a_l = ti.Vector.zero(float, dim)  # reset DEMf_a
        new_FLIP_v_l = v_s[p]
        new_PIC_v_l = ti.Vector.zero(float, dim)
        new_C_l = ti.Matrix.zero(float, dim, dim)
        grad_v_l = ti.Matrix.zero(float, dim, dim)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            g_v = grid_lv[base + offset]
            mixture_g_v = grid_lv[base + offset] * (1 - grid_phi_s[base + offset]) - grid_phi_s[base + offset] * grid_sv[base + offset]
            g_old_v = grid_l_old_v[base + offset]
            weight = w[i][0] * w[j][1] * w[k][2]
            weight_grad = ti.Vector(
                [grad_w[i][0] * w[j][1] * w[k][2], w[i][0] * grad_w[j][1] * w[k][2], w[i][0] * w[j][1] * grad_w[k][2]])
            FLIP_a_l += weight * (g_v - g_old_v) * inv_dt  # calculate the FLIP_a from FLIP acceleration
            ##### grid_sv[i, j, k] += dt * (gravity + grid_sf[i, j, k] / grid_sm[i, j, k])
            If_a_l += weight * grid_if[base + offset] / grid_lm[base + offset]  # calculate the If_a from the grid
            #DEMf_a_l += weight * grid_DEMf[base + offset] / grid_lm[base + offset]  # calculate the DEMf_a from the grid
            new_FLIP_v_l += weight * (g_v - g_old_v)
            new_PIC_v_l += weight * g_v

            new_C_l += 4 * weight * g_v.outer_product(dpos) * inv_dx ** 2  # get new particle affine velocity matrix
            grad_v_l += mixture_g_v.outer_product(weight_grad)

        # new_FLIP_v -=  DEMf_a*dt*DEM_deduction_flag # remove DEM a if necessary

        F_l[p] = (ti.Matrix.identity(float, dim) + dt * grad_v_l) @ F_l[p] #F = (I+L*dt)*F
        delta_J = F_l[p].determinant()

        D_l = (grad_v_l + grad_v_l.transpose()) * 0.5 #D = (L + LT) / 2
        Dprime = D_l - ((1 / 3 * D_l.trace()) * ti.Matrix.identity(float, dim))
        Shear = 2 * 0.5 * Dprime
        K = 2.0e6
        pore_iso = K * (delta_J ** 7.0 - 1)
        pore_3D[p] = pore_iso * ti.Matrix.identity(float, dim) + Shear
        # Project delta_J to nodes


        #this_particle_old_J = sand_material.get_J(p)
        #sand_material.update_J_no_bar(delta_J, p)

        #this_particle_new_J_no_bar = sand_material.get_J_no_bar(p)

        #this_particle_new_volume = p_vol * this_particle_new_J_no_bar

        # for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
        #     offset = ti.Vector([i, j, k])
        #     weight = w[i][0] * w[j][1] * w[k][2]
        #
        #     grid_weight_sum[base + offset] += weight * this_particle_new_volume
        #
        #     grid_delta_J[base + offset] += weight * this_particle_new_volume * delta_J
        #     grid_total_old_J[base + offset] += weight * this_particle_new_volume * this_particle_old_J

        # particle advection
        if APIC_flag:
            v_l[p] = new_PIC_v_l
        else:
            v_l[p] = localFlipBlendingCoeff[p] * new_FLIP_v_l + \
                     (1 - localFlipBlendingCoeff[p]) * new_PIC_v_l
        Affine_C_l[p] = new_C_l
        l_FLIP_a[p] = FLIP_a_l
        l_If_a[p] = If_a_l
        #l_DEMres_a[p] = DEMf_a_l

    # Solve for grid J
    # for i, j, k in grid_delta_J:
    #     this_grid_weight_sum = grid_weight_sum[i, j, k]
    #     if this_grid_weight_sum > 0:
    #         grid_delta_J[i, j, k] /= this_grid_weight_sum
    #         grid_total_old_J[i, j, k] /= this_grid_weight_sum
    #
    # # Particles loop to update material
    # for p in range(n_s_particles[None]):
    #     base = (x_s[p] * inv_dx - 0.5).cast(int)
    #     fx = x_s[p] * inv_dx - base.cast(float)
    #     w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    #     grad_w = [(fx - 1.5) * inv_dx, (2 - 2 * fx) * inv_dx, (fx - 0.5) * inv_dx]
    #
    #     new_PIC_v = ti.Vector.zero(float, dim)
    #     grad_v = ti.Matrix.zero(float, dim, dim)
    #
    #     delta_J_bar = 0.0
    #     total_old_J_bar = 0.0
    #
    #     this_particle_old_J = sand_material.get_J(p)
    #     for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
    #         weight = w[i][0] * w[j][1] * w[k][2]
    #         weight_grad = ti.Vector(
    #             [grad_w[i][0] * w[j][1] * w[k][2], w[i][0] * grad_w[j][1] * w[k][2], w[i][0] * w[j][1] * grad_w[k][2]])
    #
    #         offset = ti.Vector([i, j, k])
    #         dpos = (offset.cast(float) - fx) * dx
    #         g_v = grid_sv[base + offset]
    #
    #         g_delta_J = grid_delta_J[base + offset]
    #         g_total_old_J = grid_total_old_J[base + offset]
    #
    #         new_PIC_v += g_v * weight
    #         grad_v += g_v.outer_product(weight_grad)
    #
    #         delta_J_bar += weight * g_delta_J
    #         total_old_J_bar += weight * g_total_old_J
    #
    #     delta_F = ti.Matrix.identity(float, dim) + dt * grad_v
    #     delta_J = delta_F.determinant()
    #
    #     # Modified F-bar (consider gasification)
    #     this_particle_new_J_no_bar = sand_material.get_J_no_bar(p)
    #     this_particle_new_volume = p_vol * this_particle_new_J_no_bar
    #     this_particle_new_density = mass_s / this_particle_new_volume
    #
    #     if (this_particle_new_density >= rho_critical):
    #         sand_material.update_deformation_gradient(
    #             (total_old_J_bar * delta_J_bar / (this_particle_old_J * delta_J)) ** (1.0 / 3.0) * delta_F, p, dt)

        # Particle advection
        x_l[p] = x_l[p] + dt * new_PIC_v_l

    # DEM coupling #
    if DEM_contact_flag:
        a = 1.0
        # DEM-DEM collision
        n_DEM = my_DEM_pillar.n_pillar
        n_pillar = n_DEM - 6
        kn_pp = k_n_crit / 10.0
        kn_pf = k_n_crit / 10.0  # kn for pillar-floor
        for i in range(n_pillar):
            # print("i: ", i)
            # ti.loop_config(serialize=True)
            for j in range(i + 1, n_DEM):
                if (my_DEM_pillar.sf_type[i] == 1 or my_DEM_pillar.sf_type[j] == 1):
                    contact, vec = my_DEM_pillar.gjk(i, j)
                    if (contact == 1.0):
                        # force computation for i
                        if (my_DEM_pillar.sf_type[i] == 1):  # only care about box
                            aci_n = my_DEM_pillar.get_contact_manifold_box(i, j)
                            if (my_DEM_pillar.sf_type[j] == 1):
                                my_DEM_pillar.contact_f_via_cur_act_cp_box(i, j, aci_n, vec, dt=dt, kn=kn_pf,
                                                                           knd=kn_pf * 0.1, ks=k_n_crit / 10, ksd=80,
                                                                           mu=0.65)
                        # force computation for j
                        if (my_DEM_pillar.sf_type[j] == 1):  # only care about box
                            acj_n = my_DEM_pillar.get_contact_manifold_box(j, i)
                            my_DEM_pillar.contact_f_via_cur_act_cp_box(j, i, acj_n, -vec, dt=dt, kn=kn_pp,
                                                                       knd=kn_pp * 0.1, ks=k_n_crit / 10, ksd=80,
                                                                       mu=0.65)
        # gravity
        for i in range(n_pillar):
            if (my_DEM_pillar.sf_type[i] == 1):
                my_DEM_pillar.v3f_fn[i] += gravity[None] * my_DEM_pillar.sf_mass[i]

        # MP-DEM coupling
        for p in range(n_l_particles[None]):
            can_contact = my_DEM_pillar.bounding_sphere_check(x_p=x_s[p], bs_r=0.05)
            if (can_contact):
                f_DEM_to_p_total = my_DEM_pillar.get_contact_force_MPM_potential(x_p=x_s[p],
                                                                                 v_p=v_s[p],
                                                                                 potential_coeff_k=potential_coeff_k,
                                                                                 r=DEM_r,
                                                                                 Cn=C_n,
                                                                                 m=mass_s,
                                                                                 targettype=1)
                DEM_force[p] = f_DEM_to_p_total
                DEM_shear_force[p] = ti.Vector.zero(float, dim)  # we do not compute shear force for now

        # DEM time integration (frozen for now)
        for i in range(n_pillar):
            if (my_DEM_pillar.sf_type[i] == 1):
                my_DEM_pillar.time_integration_semi_explicit(i, dt)
    else:
        for p in range(n_s_particles[None]):
            DEM_force[p] = ti.Vector.zero(float, dim)
            DEM_shear_force[p] = ti.Vector.zero(float, dim)




# ============ INITIALIZATION ============
@ti.kernel
def initialize():
    n_s_particles[None] = 0
    # material initialization
    sand_material.initialize()
    # Particle preparation
    new_particle_id = 0
    for i, j, k in grid_sm:
        if i * dx > start_x - 0.5 * dx and \
                i * dx < start_x + l0 - 0.5 * dx and \
                j >= 2 and j * dx < 2 * dx + h0 - 0.5 * dx and \
                k * dx > start_y - 0.5 * dx and \
                k * dx < start_y + w0 - 0.5 * dx:  # (a box from star_x ~ star_x+lo star_y ~ star_y+w0 0~h0)
            for index_x in range(0, n_particles_per_direction_x):
                for index_z in range(0, n_particles_per_direction_z):
                    for index_y in range(0, n_particles_per_direction_y):
                        new_particle_id = ti.atomic_add(n_s_particles[None], 1)
                        x_s[new_particle_id] = [i * dx + (0.5 + index_x) * dx / n_particles_per_direction_x, \
                                                j * dx + (0.5 + index_z) * dx / n_particles_per_direction_z, \
                                                k * dx + (
                                                            0.5 + index_y) * dx / n_particles_per_direction_y]  # arrange particles on all the vertexes of the grid
                        x_s_0[new_particle_id] = x_s[new_particle_id]  # initial particle arrangement
                        # v_s[new_particle_id] = ti.Matrix([0,0,0])
                        tracking_particle_flag[new_particle_id] = 0
    print('Particle number: ', n_s_particles[None])
    for p in range(n_s_particles[None]):
        n_s_0[p] = 0.4
        n_s[p] = n_s_0[p]

    n_l_particles[None] = 0
    # material initialization
    # Particle preparation
    new_particle_id = 0
    for i, j, k in grid_sm:
        if i * dx > start_x - 0.5 * dx and \
                i * dx < start_x + l0 - 0.5 * dx and \
                j >= 2 and j * dx < 2 * dx + h0 - 0.5 * dx and \
                k * dx > start_y - 0.5 * dx and \
                k * dx < start_y + w0 - 0.5 * dx:  # (a box from star_x ~ star_x+lo star_y ~ star_y+w0 0~h0)
            for index_x in range(0, n_particles_per_direction_x):
                for index_z in range(0, n_particles_per_direction_z):
                    for index_y in range(0, n_particles_per_direction_y):
                        new_particle_id = ti.atomic_add(n_l_particles[None], 1)
                        x_l[new_particle_id] = [i * dx + (0.5 + index_x) * dx / n_particles_per_direction_x, \
                                                j * dx + (0.5 + index_z) * dx / n_particles_per_direction_z, \
                                                k * dx + (
                                                        0.5 + index_y) * dx / n_particles_per_direction_y]  # arrange particles on all the vertexes of the grid
                        x_l_0[new_particle_id] = x_l[new_particle_id]  # initial particle arrangement
                        # v_s[new_particle_id] = ti.Matrix([0,0,0])
                        tracking_particle_flag[new_particle_id] = 0
    print('Particle number: ', n_l_particles[None])
    for p in range(n_l_particles[None]):
        F_l[p] = ti.Matrix.identity(float, dim)
        pore_3D[p] = ti.zero(float)


@ti.kernel
def MaterialInitialization():
    n_s_particles[None] = 0
    # material initialization
    sand_material.initialize()










