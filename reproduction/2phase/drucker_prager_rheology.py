# Drucker-Prager with \mu I rheology


# Authors:
# Yidong Zhao (ydzhao@hku.hk)
# Yupeng Jiang (yupjiang@hku.hk)


# main references - papers:
# 1. [Klar et al., Drucker-Prager sand simulation] (https://dl.acm.org/doi/10.1145/2897824.2925906)
# 2. [Klar et al., Drucker-Prager sand simulation - supplementary file] (https://www.seas.upenn.edu/~cffjiang/research/sand/tech-doc.pdf)
# 3. [Tampubolon et al., Multi-species simulation of porous sand and water mixtures] (https://dl.acm.org/doi/10.1145/3072959.3073651)
# 4. [Dunatunga, Kamrin, JFM, 2015]


import taichi as ti
import numpy as np

# ti.init(arch=ti.gpu)
# constants
pi = 3.141592653
tol = 1e-8

@ti.data_oriented
class DruckerPragerRheology:
    def __init__(self,
                 n_particles, # number of particles
                 dim, # problem dimension, 2 = 2D (plain strain assumption), 3 = 3D
                 E, # Young's modulus
                 nu, # Poisson's ratio
                 friction_angle, # friction angle in degree unit
                 cohesion, # cohesion TODO: check
                 mu_2, # mu_2 parameter for rheology
                 xi, # xi parameter for rheology
                 implicit_rheology_flag # implicit or explicit flag for rheology
                 ):
        self.n_particles = n_particles
        self.dim = dim
        self.E = E
        self.nu = nu
        self.lame_mu, self.lame_lambda = E / (2*(1+nu)), E*nu / ((1+nu) * (1-2*nu)) # Lame parameters
        # lame_mu aka G (shear modulus)
        self.friction_angle = friction_angle

        # Rheology parameters
        self.mu_2 = mu_2
        self.xi = xi
        self.implicit_rheology_flag = implicit_rheology_flag

        # Quantities declaration (some of them will be initialized in another funciton)
        self.F_elastic_array = ti.Matrix.field(3, 3, dtype = float, shape = n_particles)
        self.J_array = ti.field(dtype = float, shape = n_particles)
        self.J_no_bar_array = ti.field(dtype = float, shape = n_particles)
        self.friction_coeff_array = ti.field(dtype = float, shape = n_particles)
        self.plastic_multiplier = ti.field(dtype = float, shape = n_particles)
        self.plastic_rate = ti.field(dtype = float, shape = n_particles)
        self.SFLIP_beta_FLAG = ti.field(dtype = float, shape = n_particles)

    # ============ MEMBER FUNCTIONS - taichi scope ============
    @ti.func
    def initialize(self):
        F_elastic_init = ti.Matrix.identity(float, 3)

        for p in range(self.n_particles):
            self.F_elastic_array[p] = F_elastic_init

            self.J_array[p] = 1.0
            self.J_no_bar_array[p] = 1.0

            self.friction_coeff_array[p] = 2.0*ti.sqrt(6)*ti.sin(self.friction_angle*pi/180) / (3-ti.sin(self.friction_angle*pi/180)) # B in Eq. (4.76) of [Borja, Plasticity Modeling & Computation]
            self.plastic_multiplier[p] = 0.0
            self.plastic_rate[p] = 0.0
            self.SFLIP_beta_FLAG[p] = 0 # 1:BETA MIN 2:BETA MAX



    @ti.func
    def return_mapping(self, e_trial, p, dt):
        # # old implementation
        # e_trial = e_trial_0 # not consider volume correction

        # ehat = e_trial - e_trial.trace()/3 * ti.Matrix.identity(float, 3) # Eq. (27) in [ref paper 1]
        # ehat_norm = ti.sqrt(ehat[0, 0] ** 2 + ehat[1, 1] ** 2 + ehat[2, 2] ** 2)

        # delta_lambda = ehat_norm + (3 * self.lame_lambda + 2 * self.lame_mu) / (2 * self.lame_mu) * e_trial.trace() * self.friction_coeff_array[p] # Eq. (27) in [ref paper 1]

        # new_e = ti.Matrix.zero(float, 3, 3)
        # delta_q = 0.0
        
        
        # # three cases
        # if ehat_norm <= 0 or e_trial.trace() > 0: # case II, project to the tip
        #     self.SFLIP_beta_FLAG[p] = 1 # if not tension, we use beta min to approximate FLIP
        #     new_e = ti.Matrix.zero(float, 3, 3)
        #     e_trial_norm = ti.sqrt(e_trial[0, 0] ** 2 + e_trial[1, 1] ** 2 + e_trial[2, 2] ** 2)
        #     delta_q = e_trial_norm

        #     self.plastic_multiplier[p] += delta_lambda
        #     self.plastic_rate[p] = delta_lambda / dt
        #     # TODO: state
        # elif delta_lambda <= 0: # case I, elastic
        #     self.SFLIP_beta_FLAG[p] = 1 # if elastic, we use beta min to approximate FLIP
        #     new_e = e_trial_0
        #     delta_q = 0.0
        #     self.plastic_rate[p] = 0.0
        #     # TODO: state
        # else: # case III, plastic

        #     # self.plastic_multiplier[p] += delta_lambda
        #     # # TODO: state
        #     self.SFLIP_beta_FLAG[p] = 2 # if plastic, we use beta max to approximate NFLIP
        #     # Rheology
        #     eps_v = e_trial.trace()
        #     tau_trial = self.lame_lambda * eps_v * ti.Matrix.identity(float, 3) + 2 * self.lame_mu * e_trial

        #     p_trial = 1/3 * tau_trial.trace() # compression negative
        #     S_trial = tau_trial - p_trial * ti.Matrix.identity(float, 3)
        #     S_trial_norm = ti.sqrt(S_trial[0, 0] ** 2 + S_trial[1, 1] ** 2 + S_trial[2, 2] ** 2)
        #     equiv_tau_trial = S_trial_norm / ti.sqrt(2)


        #     mu_s = 1.0/ti.sqrt(2) * 3 * self.friction_coeff_array[p]

        #     new_equiv_tau = 0.0

        #     # new_equiv_tau  = (-p_trial) * mu_s / ti.sqrt(2) # no rheology, just for checking the implementation

        #     # # implicit rheoloty
        #     if (self.implicit_rheology_flag):
        #         S0 = mu_s * (-p_trial)
        #         S2 = self.mu_2 * (-p_trial)
        #         alpha = self.xi * self.lame_mu * dt * ti.sqrt(-p_trial)
        #         B = S2 + equiv_tau_trial + alpha
        #         H = S2 * equiv_tau_trial + S0 * alpha

        #         # solve for tau_n+1
        #         new_equiv_tau = (B-ti.sqrt(B*B - 4*H)) / 2


        #     else:
        #         # explicit rheology
        #         current_mu = mu_s
        #         if (self.plastic_rate[p] > tol):
        #             current_mu = mu_s + (self.mu_2 - mu_s) / (self.xi*ti.sqrt(-p_trial)/(ti.sqrt(2)*self.plastic_rate[p]) + 1)
        #         new_equiv_tau = (-p_trial) * current_mu / ti.sqrt(2)


        #     # get new strains
        #     delta_lambda = (equiv_tau_trial - new_equiv_tau) / (ti.sqrt(2)*self.lame_mu)
        #     new_e = e_trial - delta_lambda / ehat_norm * ehat

        #     self.plastic_rate[p] = delta_lambda / dt

        #     self.plastic_multiplier[p] += delta_lambda


        # return new_e







        # new implementation
        # Calculate trial stress
        eps_v = e_trial.trace()
        tau_trial = self.lame_lambda * eps_v * ti.Matrix.identity(float, 3) + 2 * self.lame_mu * e_trial

        # Get P and S
        P = 1/3 * tau_trial.trace()
        S_trial = tau_trial - P * ti.Matrix.identity(float, 3)
        S_trial_norm = ti.sqrt(S_trial[0, 0] ** 2 + S_trial[1, 1] ** 2 + S_trial[2, 2] ** 2)
        equiv_tau_trial = S_trial_norm / ti.sqrt(2) # see the definition in (2.10) of [Dunatunga, Kamrin, JFM, 2015]
        Q_trial = ti.sqrt(3.0/2.0) * S_trial_norm

        new_e = ti.Matrix.zero(float, 3, 3)


        mu_s = self.friction_coeff_array[p] / ti.sqrt(2)


        # return mapping
        yield_function = ti.sqrt(2.0/3.0) * Q_trial + self.friction_coeff_array[p] * P

        if yield_function <= 0: # elastic
            new_e = e_trial
        elif yield_function > 0 and e_trial.trace() > 0: # plasticity, return mapping to the tip
            delta_lambda = ti.sqrt(e_trial[0,0]**2 + e_trial[1,1]**2 + e_trial[2,2]**2) # TODO: confirm this

            new_e = ti.Matrix.zero(float, 3, 3)

            self.plastic_multiplier[p] += delta_lambda
            self.plastic_rate[p] = delta_lambda / dt
            self.J_array[p] = 1.0
            self.J_no_bar_array[p] = 1.0

        elif yield_function > 0: # plasticity, radial return mapping with dilation angle = 0
            
            # Implicit rheology
            new_equiv_tau = 0.0

            S0 = mu_s * (-P)
            S2 = self.mu_2 * (-P)
            alpha = self.xi * self.lame_mu * dt * ti.sqrt(-P)
            B = S2 + equiv_tau_trial + alpha
            H = S2 * equiv_tau_trial + S0 * alpha

            # solve for tau_n+1
            # new_equiv_tau = (B-ti.sqrt(B*B - 4*H)) / 2 # the solution of Eq. (3.14) in [Dunatunga, Kamrin, JFM, 2015] 
            new_equiv_tau = 2*H/(B+ti.sqrt(B*B - 4*H)) # the solution of Eq. (3.15) in [Dunatunga, Kamrin, JFM, 2015] 
            # get new strains
            delta_lambda = (equiv_tau_trial - new_equiv_tau) / (ti.sqrt(2)*self.lame_mu)

            # radial direction
            n = ti.Matrix.zero(float, 3, 3)
            if S_trial_norm > 0:
                n = S_trial / S_trial_norm

            new_e = e_trial - delta_lambda * n

            self.plastic_rate[p] = delta_lambda / dt
            self.plastic_multiplier[p] += delta_lambda


        return new_e




    @ti.func
    def update_deformation_gradient(self, delta_F, p, dt):
        # new_F_elastic_trial = (ti.Matrix.identity(float, self.dim) + dt * new_C) @ self.F_elastic_array[p]
        # new_C_3D = ti.Matrix.zero(float, 3, 3)
        delta_F_3D = ti.Matrix.zero(float, 3, 3)
        if self.dim == 2:
            for i, j in ti.static(ti.ndrange(2, 2)):
                # new_C_3D[i, j] = new_C[i, j]
                delta_F_3D[i, j] = delta_F[i, j]
            # new_C_3D[2, 2] = 0.0
            delta_F_3D[2, 2] = 1.0
        else:
            for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
                # new_C_3D[i, j] = new_C[i, j]
                delta_F_3D[i, j] = delta_F[i, j]

        # new_F_elastic_trial = (ti.Matrix.identity(float, 3) + dt * new_C_3D) @ self.F_elastic_array[p]
        new_F_elastic_trial = delta_F_3D @ self.F_elastic_array[p]

        # update J
        delta_J = delta_F_3D.determinant()
        self.J_array[p] *= delta_J


        U, sig, V = ti.svd(new_F_elastic_trial)
        e = ti.Matrix.zero(float, 3, 3)
        for d in ti.static(range(3)):
            e[d, d] = ti.log(sig[d, d])
        self.SFLIP_beta_FLAG[p] = 2 # by default, we use beta max to approximate NFLIP
        new_e = self.return_mapping(e, p, dt)
        # get new elastic deformation gradient
        exp_new_e = ti.Matrix.zero(float, 3, 3)
        for d in ti.static(range(3)):
            exp_new_e[d, d] = ti.exp(new_e[d, d])
        new_F_elastic = U @ exp_new_e @ V.transpose()
        # update elastic deformation gradient
        self.F_elastic_array[p] = new_F_elastic


    @ti.func
    def update_J_no_bar(self, delta_J, p):
        self.J_no_bar_array[p] *= delta_J


    @ti.func
    def get_Kirchhoff_stress(self, p):
        # # old implementation
        # U, sig, V = ti.svd(self.F_elastic_array[p])
        # inv_sig = sig.inverse()
        # e = ti.Matrix.zero(float, 3, 3)
        # for d in ti.static(range(3)):
        #     e[d, d] = ti.log(sig[d, d])
        # stress = U @ (2 * self.lame_mu * inv_sig @ e + self.lame_lambda * e.trace() * inv_sig) @ V.transpose() # formula (26) in Klar et al., pk1 stress
        # stress = stress @ self.F_elastic_array[p].transpose() # Kirchhoff stress

        # return stress


        # new implementation
        U, sig, V = ti.svd(self.F_elastic_array[p])
        inv_sig = sig.inverse()
        e = ti.Matrix.zero(float, 3, 3)
        for d in ti.static(range(3)):
            e[d, d] = ti.log(sig[d, d]) # get principle logarithmic strain
        e_trace = e.trace()

        Kirchhoff_principal = ti.Matrix.zero(float, 3, 3)
        for d in ti.static(range(3)):
            Kirchhoff_principal[d, d] = self.lame_lambda * e_trace + 2 * self.lame_mu * e[d, d]

        Kirchhoff_stress = U @ (Kirchhoff_principal) @ U.transpose()

        return Kirchhoff_stress





    @ti.func
    def get_Cauchy_stress(self, p):
        # # old implementation
        # U, sig, V = ti.svd(self.F_elastic_array[p])
        # inv_sig = sig.inverse()
        # e = ti.Matrix.zero(float, 3, 3)
        # for d in ti.static(range(3)):
        #     e[d, d] = ti.log(sig[d, d])
        # stress = U @ (2 * self.lame_mu * inv_sig @ e + self.lame_lambda * e.trace() * inv_sig) @ V.transpose() # formula (26) in Klar et al., pk1 stress
        # stress = stress @ self.F_elastic_array[p].transpose() # Kirchhoff stress

        # return stress/self.J_array[p]



        # new implementation
        U, sig, V = ti.svd(self.F_elastic_array[p])
        inv_sig = sig.inverse()
        e = ti.Matrix.zero(float, 3, 3)
        for d in ti.static(range(3)):
            e[d, d] = ti.log(sig[d, d]) # get principle logarithmic strain
        e_trace = e.trace()

        Kirchhoff_principal = ti.Matrix.zero(float, 3, 3)
        for d in ti.static(range(3)):
            Kirchhoff_principal[d, d] = self.lame_lambda * e_trace + 2 * self.lame_mu * e[d, d]

        Kirchhoff_stress = U @ (Kirchhoff_principal) @ U.transpose()

        return Kirchhoff_stress/self.J_array[p]







    @ti.func
    def get_plastic_multiplier(self, p):
        return self.plastic_multiplier[p]

    @ti.func
    def get_plastic_rate(self, p):
        return self.plastic_rate[p]

    @ti.func
    def get_J(self, p):
        return self.J_array[p]

    @ti.func
    def get_J_no_bar(self, p):
        return self.J_no_bar_array[p]











