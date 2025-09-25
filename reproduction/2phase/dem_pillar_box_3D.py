# DEM 3D objects, including pillar (for trees) and box (for walls and floors)

# Ref: 
# // "Implementing GJK" by Casey Muratori:
# // The best description of the algorithm from the ground up
# // https://www.youtube.com/watch?v=Qupqu1xe7Io

# // "Implementing a GJK Intersection Query" by Phill Djonov
# // Interesting tips for implementing the algorithm
# // http://vec3.ca/gjk/implementation/

# // "GJK Algorithm 3D" by Sergiu Craitoiu
# // Has nice diagrams to visualise the tetrahedral case
# // http://in2gpu.com/2014/05/18/gjk-algorithm-3d/

# // "GJK + Expanding Polytope Algorithm - Implementation and Visualization"
# // Good breakdown of EPA with demo for visualisation
# // https://www.youtube.com/watch?v=6rgiPrzqt9w

# the original version of this code is written in C++, the GJK and EPA part is modified from:
# kevinmoran/GJK:https://github.com/kevinmoran/GJK
# i.e., Kevin's implementation of the Gilbert-Johnson-Keerthi intersection algorithm and the Expanding Polytope Algorithm

# Author:
# Zhengyu Liang (lzyhku@connect.hku.hk)

import taichi as ti
import numpy as np
import sys
import os

sys.path.append("..")
sys.path.append(".\sample_points")
from math_util.matrix_generation import*
#ti.init(arch=ti.gpu,debug=True, default_fp=ti.f64,device_memory_GB=7,random_seed=41) 

@ti.data_oriented
class DemPB3D:
# All DEM elements are assumed to be transformed from a basic element:
# For pillars, the basic element is a cylinder positioned at [0, 0, 0]
# with a radius of 1 and height of 2.
# For boxes, the basic element is a cube positioned at [0, 0, 0] with 
# a length of 2.
# A naming convention is used in this class to hint at data types:
# v[x]f represents a vector field with x dimensions.
# m[x]f represents a matrix field with x*x dimensions.
# sf represents a scalar field.
# note that we assume y vector[1] as the "up" direction
    def __init__(self,
                #  v3f_lin_mv, # linear momentum
                #  v3f_ang_mv, # angular momentum
                #  v3f_pos, # position
                #  v4f_quat, # quaternion to describe rotation
                #  m3f_mat_s, # scale matrix
                #  m3f_mat_sr, # scale_rotation matrix
                #  m3f_mat_sr_inv, # inverse matrix of mat_sr
                #  sf_mass, # contact coefficient
                #  sf_type, # type of the DEM element (0: pillar, 1: box)
                 n_pillar = 1 # number of pillars 
                ):
        # some global parameters
        self.EPSILON = 1e-6
        self.GJK_MAX_NUM_ITERATIONS = 64
        self.EPA_TOLERANCE = 0.005
        self.EPA_MAX_NUM_FACES = 300
        self.EPA_MAX_NUM_LOOSE_EDGES = 300
        self.EPA_MAX_NUM_ITERATIONS = 300
        self.v3f_EPA_poly = ti.Vector.field(3, dtype=float, shape=(n_pillar, self.EPA_MAX_NUM_FACES, 4))
        self.v3f_EPA_loose_edges = ti.Vector.field(3, dtype=float, shape=(n_pillar, self.EPA_MAX_NUM_LOOSE_EDGES, 2))

        print("init DemPB3D")
        # dynamic
        self.n_pillar = n_pillar
        self.v3f_lin_mv = ti.Vector.field(3, dtype=float, shape=n_pillar) # linear momentum
        self.v3f_ang_mv = ti.Vector.field(3, dtype=float, shape=n_pillar) # angular momentum (local coordinate)
        self.v3f_pos = ti.Vector.field(3, dtype=float, shape=n_pillar) # position
        self.v4f_quat = ti.Vector.field(4, dtype=float, shape=n_pillar) # quaternion to describe rotation
        self.m3f_mat_s = ti.Matrix.field(3, 3, dtype=float, shape=n_pillar) # scale matrix
        self.m3f_mat_r = ti.Matrix.field(3, 3, dtype=float, shape=n_pillar) # rotation matrix
        self.m3f_mat_sr = ti.Matrix.field(3, 3, dtype=float, shape=n_pillar) # scale_rotation matrix
        self.m3f_mat_sr_inv = ti.Matrix.field(3, 3, dtype=float, shape=n_pillar) # inverse matrix of mat_sr
        self.sf_mass = ti.field(dtype=float, shape=n_pillar) # DEM mass
        self.sf_type = ti.field(dtype=int, shape=n_pillar) # type of the DEM element (0: cylinder, 1: box)
        self.sf_touch_MP = ti.field(dtype=int, shape=n_pillar)
        self.sf_exceed_critical_moment = ti.field(dtype=int, shape=n_pillar)


        self.v3f_moment = ti.Vector.field(3, dtype=float, shape=n_pillar) # rotation moment
        self.v3f_moment_d = ti.Vector.field(3, dtype=float, shape=n_pillar) # rotation damping moment
        self.v3f_fn = ti.Vector.field(3, dtype=float, shape=n_pillar) # normal force
        self.v3f_fnd = ti.Vector.field(3, dtype=float, shape=n_pillar) # normal damping force
        self.v3f_fs = ti.Vector.field(3, dtype=float, shape=n_pillar) # shear force

        # sample points
        # for cylinder
        nparr_sample_points_cylinder = np.loadtxt(".\\materials\\sample_points\\cylinder_sample.txt")
        self.n_sp_cylinder = nparr_sample_points_cylinder.shape[0]
        self.v3f_sample_points_cylinder = ti.Vector.field(3, dtype = float, shape = self.n_sp_cylinder)
        self.v3f_sample_points_cylinder.from_numpy(nparr_sample_points_cylinder)
        self.f_active_sp_cylinder = ti.field(dtype = int, shape = (n_pillar, self.n_sp_cylinder)) # document which sp is used to form the contact manifold
        self.f_active_contact_dist_cylinder = ti.field(dtype = float, shape = (n_pillar, self.n_sp_cylinder)) # document which sp is used to form the contact manifold
        self.f_active_contact_vec_cylinder = ti.Vector.field(3, dtype = float, shape = (n_pillar, self.n_sp_cylinder)) # document which sp is used to form the contact manifold
        self.v3f_cached_fs_local_cylinder = ti.Vector.field(3, dtype = float, shape = (n_pillar, self.n_sp_cylinder)) # chached shear force (each sample point)
        self.f_active_sp_table_cylinder = ti.field(dtype = int, shape = (n_pillar, self.n_sp_cylinder)) # active table of contact points for resetting shear force

        # for cube
        nparr_sample_points_box = np.loadtxt(".\\materials\\sample_points\\box_sample.txt")
        self.n_sp_box = nparr_sample_points_box.shape[0]
        self.v3f_sample_points_box = ti.Vector.field(3, dtype = float, shape = self.n_sp_box)
        self.v3f_sample_points_box.from_numpy(nparr_sample_points_box)
        self.f_active_sp_box = ti.field(dtype = int, shape=(n_pillar, self.n_sp_box)) # document which sp is used to form the contact manifold
        self.f_active_contact_dist_box = ti.field(dtype = float, shape=(n_pillar, self.n_sp_box)) # document which sp is used to form the contact manifold
        self.f_active_contact_vec_box = ti.Vector.field(3, dtype = float, shape = (n_pillar, self.n_sp_box)) # document which sp is used to form the contact manifold
        self.v3f_cached_fs_local_box = ti.Vector.field(3, dtype = float, shape = (n_pillar, self.n_sp_box)) # chached shear force (each sample point)
        self.f_active_sp_table_box = ti.field(dtype = int, shape = (n_pillar, self.n_sp_box)) # active table of contact points for resetting shear force
    

        self.data_prep_()
        print("init complete")



    # ============ GJK Utilities ============
    @ti.kernel
    def data_prep_(self):
        for i in range(self.n_pillar):
            self.m3f_mat_s[i] = ti.Matrix.identity(n=3, dt=ti.f32)
            self.m3f_mat_r[i] = ti.Matrix.identity(n=3, dt=ti.f32)
            self.m3f_mat_sr[i] = ti.Matrix.identity(n=3, dt=ti.f32)
            self.m3f_mat_sr_inv[i] = ti.Matrix.identity(n=3, dt=ti.f32)
            self.sf_mass[i] = 1.0
            self.v4f_quat[i] = ti.Vector([0.0, 0.0, 0.0, 1.0])
        

        # prepare cylinder sample points to reverse y and z so as to follow the OPENGL convention
        for ii in range(self.n_sp_cylinder):
            pt = self.v3f_sample_points_cylinder[ii]
            temp = pt.z
            pt.z = pt.y
            pt.y = temp
            self.v3f_sample_points_cylinder[ii] = pt
        # prepare box sample points to reverse y and z so as to follow the OPENGL convention
        for ii in range(self.n_sp_box):
            pt = self.v3f_sample_points_box[ii]
            temp = pt.z
            pt.z = pt.y
            pt.y = temp
            self.v3f_sample_points_box[ii] = pt

    @ti.func
    def set_mat_via_quat_(self, i):
        self.m3f_mat_r[i] = quat_to_mat(self.v4f_quat[i])
        self.m3f_mat_sr[i] = self.m3f_mat_r[i] @ self.m3f_mat_s[i]
        self.m3f_mat_sr_inv[i] = ti.math.inverse(self.m3f_mat_sr[i])

    @ti.func
    def set_model_pos_quat(self, i, scale_vec, pos, q): # set the status of an DEM element
        self.v3f_pos[i] = pos
        self.m3f_mat_s[i] = scale_mat3(scale_vec)
        self.v4f_quat[i] = q.normalized()
        self.set_mat_via_quat_(i)
    
    @ti.func
    def support_box_(self, i, v3_dir):
        mat_sr_inv = self.m3f_mat_sr_inv[i]
        dir = (mat_sr_inv @ v3_dir).normalized()
        x = 0.0
        y = 0.0
        z = 0.0
        if(dir.x > 0.0):
            x = 1.0
        else:
            x = -1.0

        if(dir.y > 0.0):
            y = 1.0
        else:
            y = -1.0

        if(dir.z > 0.0):
            z = 1.0
        else:
            z = -1.0
        result = ti.Vector([x, y, z])
        return self.m3f_mat_sr[i] @ result + self.v3f_pos[i]
    
    @ti.func
    def support_pillar_(self, i, v3_dir):
        mat_sr_inv = self.m3f_mat_sr_inv[i]
        dir = (mat_sr_inv @ v3_dir).normalized()
        
        # print("/n")
        result = ti.Vector([0.0, 0.0, 0.0])
        if(dir.y > 1.0 - self.EPSILON):
            # vertically up
            result = ti.Vector([0.0, 1.0, 0.0])
        elif(dir.y < -1.0 + self.EPSILON):
            # vertically down
            #print("case_vertically down")
            result = ti.Vector([0.0, -1.0, 0.0])
            #print("result", result)
        else:
            dir_xz = ti.Vector([dir.x, 0.0, dir.z]).normalized()
            if(dir.y > 0.0):
                dir_xz.y = 1.0
            else:
                dir_xz.y = -1.0
            result = dir_xz
        # print("support pt: ", i, sep = ' ', end = ' | ')
        # print(" ", result)
        return self.m3f_mat_sr[i] @ result + self.v3f_pos[i]
    
    @ti.func
    def support(self, i, search_dir):
        ti.static_print("compile support: start")
        c1 = ti.Vector([0.0, 0.0, 0.0])
        if(self.sf_type[i] == 0):
            c1 = self.support_pillar_(i, search_dir)
        elif(self.sf_type[i] == 1):
            c1 = self.support_box_(i, search_dir)
        ti.static_print("compile support: end")
        return c1
    
    @ti.func
    def update_simplex3_(self, v3f_simplex_list):
        ti.static_print("compile update_simplex3_: start")
        # /* Required winding order:
        # //  b
        # //  |\
        # //  | \
        # //  |  a
        # //  | /
        # //  |/
        # //  c
        # i.e., C.C.W.
        a = v3f_simplex_list[0, 0:3]
        b = v3f_simplex_list[1, 0:3]
        c = v3f_simplex_list[2, 0:3]
        n = ti.math.cross(b - a, c - a) # triangle's normal
        #print("normal: ", n)
        ao = -a # direction to origin
        # assume the new simplex is 2d
        simp_dim = 2
        pass_flag = 0.0
        search_dir = ti.Vector([0.0, 0.0, 0.0])
        if(ti.math.dot(ti.math.cross(b - a, n), ao) > 0.0):
            #print("case1")
            v3f_simplex_list[2, 0:3] = v3f_simplex_list[0, 0:3] # eliminate c and make room for new point
            search_dir =  ti.math.cross(ti.math.cross(b-a, ao), b-a)
            pass_flag = 1.0
        if(pass_flag == 0.0):
            if(ti.math.dot(ti.math.cross(n, c - a), ao) > 0.0):
                #print("case2")
                v3f_simplex_list[1, 0:3] = v3f_simplex_list[0, 0:3] # eliminate b and make room for new point
                search_dir =  ti.math.cross(ti.math.cross(c-a, ao), c-a)
                pass_flag = 1.0
            if(pass_flag == 0.0):
                #print("case3")
                # assume the new simplex is 3d
                simp_dim = 3
                d = v3f_simplex_list[3, 0:3]
                if(ti.math.dot(n, ao) > 0.0): # above triangle
                    d = c
                    c = b
                    b = a
                    v3f_simplex_list[1, 0:3] = b
                    v3f_simplex_list[2, 0:3] = c
                    v3f_simplex_list[3, 0:3] = d
                    search_dir = n
                    pass_flag = 1.0
                if(pass_flag == 0.0):
                    # else below triangle
                    d = b
                    b = a
                    v3f_simplex_list[1, 0:3] = b
                    v3f_simplex_list[3, 0:3] = d
                    search_dir = -n
        ti.static_print("compile update_simplex3_: end")
        return v3f_simplex_list, simp_dim, search_dir
    
    @ti.func
    def update_simplex4_(self, v3f_simplex_list):
        ti.static_print("compile update_simplex4_: start")
        a = v3f_simplex_list[0, 0:3]
        b = v3f_simplex_list[1, 0:3]
        c = v3f_simplex_list[2, 0:3]
        d = v3f_simplex_list[3, 0:3]

        ABC = ti.math.cross(b-a, c-a) # normal vector of the face ABC
        ACD = ti.math.cross(c-a, d-a) # normal vector of the face ACD
        ADB = ti.math.cross(d-a, b-a) # normal vector of the face ADB
        ao = -a # direction to origin
        # assume the new simplex is 3d
        simp_dim = 3
        # comment from the original author:
        # //Plane-test origin with 3 faces
        # /*
        # // Note: Kind of primitive approach used here; If origin is in front of a face, just use it as the new simplex.
        # // We just go through the faces sequentially and exit at the first one which satisfies dot product. Not sure this 
        # // is optimal or if edges should be considered as possible simplices? Thinking this through in my head I feel like 
        # // this method is good enough. Makes no difference for AABBS, should test with more complex colliders.
        # */
        contact_flag = 1.0
        search_dir = ti.Vector([0.0, 0.0, 0.0])
        if(contact_flag == 1.0):
            if(ti.math.dot(ABC, ao) > 0.0):  # In front of ABC
                d = c
                c = b
                b = a
                v3f_simplex_list[1, 0:3] = b
                v3f_simplex_list[2, 0:3] = c
                v3f_simplex_list[3, 0:3] = d
                search_dir = ABC
                contact_flag = 0.0
            if(contact_flag == 1.0):
                if(ti.math.dot(ACD, ao) > 0.0):  # In front of ACD
                    b = a
                    v3f_simplex_list[1, 0:3] = b
                    search_dir = ACD
                    contact_flag = 0.0
                if(contact_flag == 1.0):
                    if(ti.math.dot(ADB, ao) > 0.0):  # In front of ADB
                        c = d
                        d, b = b, a
                        v3f_simplex_list[1, 0:3] = b
                        v3f_simplex_list[2, 0:3] = c
                        v3f_simplex_list[3, 0:3] = d
                        search_dir = ADB
                        contact_flag = 0.0
        # else inside tetrahedron; enclosed!
        ti.static_print("compile update_simplex4_: end")
        return contact_flag, v3f_simplex_list, simp_dim, search_dir
    
    @ti.func
    def pt_isequal(self, vec1, vec2):
        ans = 0.0
        if ti.math.length(vec1 - vec2) < self.EPSILON:
            ans = 1.0
        return ans
    
    # ============ EPA for computing contact normal ============
    @ti.func
    def EPA(self, v3f_simplex_list, i, j):
        ti.static_print("compile EPA: start")
        # print("EPA: 1st")
        a = v3f_simplex_list[0, 0:3]
        b = v3f_simplex_list[1, 0:3]
        c = v3f_simplex_list[2, 0:3]
        d = v3f_simplex_list[3, 0:3]
        # init with the final simplex from GJK
        self.v3f_EPA_poly[i, 0, 0][0:3] = a
        self.v3f_EPA_poly[i, 0, 1][0:3] = b
        self.v3f_EPA_poly[i, 0, 2][0:3] = c
        self.v3f_EPA_poly[i, 0, 3][0:3] = ti.math.cross(b - a, c - a).normalized()

        self.v3f_EPA_poly[i, 1, 0][0:3] = a
        self.v3f_EPA_poly[i, 1, 1][0:3] = c
        self.v3f_EPA_poly[i, 1, 2][0:3] = d
        self.v3f_EPA_poly[i, 1, 3][0:3] = ti.math.cross(c - a, d - a).normalized()

        self.v3f_EPA_poly[i, 2, 0][0:3] = a
        self.v3f_EPA_poly[i, 2, 1][0:3] = d
        self.v3f_EPA_poly[i, 2, 2][0:3] = b
        self.v3f_EPA_poly[i, 2, 3][0:3] = ti.math.cross(d - a, b - a).normalized()

        self.v3f_EPA_poly[i, 3, 0][0:3] = b
        self.v3f_EPA_poly[i, 3, 1][0:3] = d
        self.v3f_EPA_poly[i, 3, 2][0:3] = c
        self.v3f_EPA_poly[i, 3, 3][0:3] = ti.math.cross(d - b, c - b).normalized()


        num_faces = 4
        closest_face = -1
        p1 = ti.Vector([0.0, 0.0, 0.0])
        p2 = ti.Vector([0.0, 0.0, 0.0])
        contact_vec = ti.Vector([0.0, 0.0, 0.0])
        converge_flag = 0
        ti.loop_config(serialize=True) # Serializes the next for loop
        for iterations in range(self.EPA_MAX_NUM_ITERATIONS):
            ti.static_print("compile EPA: check point1")
            if converge_flag == 0:
                #Find face that's closest to origin
                min_dist = ti.math.dot(self.v3f_EPA_poly[i, 0, 0], self.v3f_EPA_poly[i, 0, 3])
                closest_face = 0
                ti.loop_config(serialize=True) # Serializes the next for loop
                for ii in range(num_faces):
                    dist = ti.math.dot(self.v3f_EPA_poly[i, ii, 0], self.v3f_EPA_poly[i, ii, 3])
                    if dist < min_dist:
                        min_dist = dist
                        closest_face = ii
                # search towards the normal of the closest face
                search_dir = self.v3f_EPA_poly[i, closest_face, 3]
                p1 = self.support(i, -search_dir)
                p2 = self.support(j, search_dir)
                p = p2 - p1
                ti.static_print("compile EPA: check point2")
                if(ti.math.dot(p, search_dir) - min_dist < self.EPA_TOLERANCE):
                    converge_flag = 1
                    contact_vec = self.v3f_EPA_poly[i, closest_face, 3] * ti.math.dot(p, search_dir)
                if converge_flag == 0:
                    # remove facing edges to make the poly convex
                    num_loose_edges = 0
                    # find all triangles facing p
                    ti.static_print("compile EPA: check point3")
                    ii = 0
                    facelimit = num_faces
                    ti.loop_config(serialize=True) # Serializes the next for loop
                    while(True):
                        if ii >= facelimit:
                            break
                        # if triangle ii faces p, remove it
                        if(ti.math.dot(self.v3f_EPA_poly[i, ii, 3], p - self.v3f_EPA_poly[i, ii, 0]) > 0.0):
                            # add removed triangle's edges to the loose edge list
                            # if it is already there, remove it (both triangles linked to this edge are gone)
                            ti.loop_config(serialize=True)
                            for jj in range(3):
                                cur_edge0 = self.v3f_EPA_poly[i, ii, jj]
                                cur_edge1 = self.v3f_EPA_poly[i, ii, (jj + 1) % 3]
                                found_edge = 0.0
                                ti.loop_config(serialize=True) # Serializes the next for loop
                                for k in range(num_loose_edges):
                                    if(self.pt_isequal(self.v3f_EPA_loose_edges[i, k, 1], cur_edge0) == 1.0 and
                                       self.pt_isequal(self.v3f_EPA_loose_edges[i, k, 0], cur_edge1) == 1.0):
                                        # Edge is already in the list, remove it
                                        # THIS ASSUMES EDGE CAN ONLY BE SHARED BY 2 TRIANGLES (which should be true)
                                        # THIS ALSO ASSUMES SHARED EDGE WILL BE REVERSED IN THE TRIANGLES (which
                                        # should be true provided every triangle is wound CCW) 
                                        self.v3f_EPA_loose_edges[i, k, 0] = self.v3f_EPA_loose_edges[i, num_loose_edges - 1, 0]
                                        self.v3f_EPA_loose_edges[i, k, 1] = self.v3f_EPA_loose_edges[i, num_loose_edges - 1, 1]
                                        # Overwrite current edge with last edge in list like swap with last and pop_back
                                        num_loose_edges -= 1
                                        found_edge = 1.0
                                        break
                                        # k = num_loose_edges # exit loop because edge can only be shared once
                                ## for k in range(num_loose_edges)

                                if(found_edge == 0.0):
                                    if num_loose_edges >= self.EPA_MAX_NUM_LOOSE_EDGES:
                                        print("num_loose_edges >= self.EPA_MAX_NUM_LOOSE_EDGES")
                                        break
                                    self.v3f_EPA_loose_edges[i, num_loose_edges, 0] = cur_edge0
                                    self.v3f_EPA_loose_edges[i, num_loose_edges, 1] = cur_edge1
                                    num_loose_edges += 1
                            ## for jj in range(3):
                            #Remove triangle i from list
                            self.v3f_EPA_poly[i, ii, 0] = self.v3f_EPA_poly[i, facelimit - 1, 0]
                            self.v3f_EPA_poly[i, ii, 1] = self.v3f_EPA_poly[i, facelimit - 1, 1]
                            self.v3f_EPA_poly[i, ii, 2] = self.v3f_EPA_poly[i, facelimit - 1, 2]
                            self.v3f_EPA_poly[i, ii, 3] = self.v3f_EPA_poly[i, facelimit - 1, 3]
                            facelimit -= 1
                            ii -= 1
                        ii += 1
                    num_faces = facelimit
                    ## for ii in range(num_faces):
                    # Reconstruct polytope with p added
                    ti.static_print("compile EPA: check point4")
                    ti.loop_config(serialize=True) # Serializes the next for loop
                    for jj in range(num_loose_edges):
                        if num_faces >= self.EPA_MAX_NUM_FACES:
                            break
                        self.v3f_EPA_poly[i, num_faces, 0] = self.v3f_EPA_loose_edges[i, jj, 0] 
                        self.v3f_EPA_poly[i, num_faces, 1] = self.v3f_EPA_loose_edges[i, jj, 1] 
                        self.v3f_EPA_poly[i, num_faces, 2] = p
                        self.v3f_EPA_poly[i, num_faces, 3] = ti.math.cross(self.v3f_EPA_loose_edges[i, jj, 0] - self.v3f_EPA_loose_edges[i, jj, 1], self.v3f_EPA_loose_edges[i, jj, 0] - p).normalized()
                        # Check for wrong normal to maintain CCW winding
                        if(ti.math.dot(self.v3f_EPA_poly[i, num_faces, 0], self.v3f_EPA_poly[i, num_faces, 3]) < -self.EPSILON):
                            temp = ti.Vector([0.0, 0.0, 0.0])
                            temp = self.v3f_EPA_poly[i, num_faces, 0][0:3]
                            self.v3f_EPA_poly[i, num_faces, 0] = self.v3f_EPA_poly[i, num_faces, 1]
                            self.v3f_EPA_poly[i, num_faces, 1] = temp
                            self.v3f_EPA_poly[i, num_faces, 3] = -self.v3f_EPA_poly[i, num_faces, 3]
                        num_faces += 1
        ## end main iterations
        if(converge_flag == 0):
            print("EPA did not converge", sep='', end=': ')  
            print("i:", i, sep=' ', end=' | ')
            print("j:", j, sep=' ', end=' | ')
            print("i type:", self.sf_type[i], sep=' ', end=' | ')
            print("j type:", self.sf_type[j])
            if(i == 0):
                print("self.v3f_EPA_poly[i, closest_face, 0]: ", self.v3f_EPA_poly[i, closest_face, 0])
                print("self.v3f_EPA_poly[i, closest_face, 1]: ", self.v3f_EPA_poly[i, closest_face, 1])
                print("self.v3f_EPA_poly[i, closest_face, 2]: ", self.v3f_EPA_poly[i, closest_face, 2])
                print("self.v3f_EPA_poly[i, closest_face, 3]: ", self.v3f_EPA_poly[i, closest_face, 3])
                print("num_faces: ", num_faces)



            contact_vec = self.v3f_EPA_poly[i, closest_face, 3] * ti.math.dot(self.v3f_EPA_poly[i, closest_face, 0], self.v3f_EPA_poly[i, closest_face, 3])
        ti.static_print("compile EPA: end")
        return contact_vec


    # ============ The GJK algorithm ============
    @ti.func
    def gjk(self, i, j):
        # v3f_simplex_list = self.v3f_simplex_list_field[i] # see how to deal with this

        ti.static_print("compile gjk: start")
        # print("gjk_invoke")
        v3f_simplex_list = ti.Matrix([[0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0]])
        #search_dir = self.v3f_pos[i] - self.v3f_pos[j]

        # pick a random initial search dir
        search_dir = ti.Vector([ti.random(ti.f32) + self.EPSILON, ti.random(ti.f32) + self.EPSILON, ti.random(ti.f32) + self.EPSILON])

        # get the first point of simplex
        c1 = self.support(i, -search_dir)
        c2 = self.support(j, search_dir)

        c = c2 - c1
        v3f_simplex_list[2, 0:3] = c
        search_dir = -c

        # get the second point of simplex
        b1 = self.support(i, -search_dir)
        b2 = self.support(j, search_dir)
        b = b2 - b1

        v3f_simplex_list[1, 0:3] = b
        # we now have 1D simplex
        # c------------b
        contact_flag = 0.0
        contact_vec = ti.Vector([0.0, 0.0, 0.0]) 
        if(ti.math.dot(v3f_simplex_list[1, 0:3], search_dir) < 0.0):
            # we didn't reach the origin, won't enclose it
            contact_flag = -2.0 # early exist
        # search perpendicular to line segment towards origin
        if(contact_flag == 0.0): # skip if contact_flag indicates early exist
            search_dir = ti.math.cross(
                            ti.math.cross(v3f_simplex_list[2, 0:3] - v3f_simplex_list[1, 0:3], 
                            -v3f_simplex_list[2, 0:3]), 
                            v3f_simplex_list[2, 0:3] - v3f_simplex_list[1, 0:3])

            if(ti.math.length(search_dir) < self.EPSILON):
                # //origin is on this line segment
                # //Apparently any normal search vector will do?
                search_dir = ti.math.cross(v3f_simplex_list[2, 0:3] - v3f_simplex_list[1, 0:3], 
                                ti.Vector([1.0, 0.0, 0.0])
                                )
                if(ti.math.length(search_dir) < self.EPSILON):
                    search_dir = ti.math.cross(v3f_simplex_list[2, 0:3] - v3f_simplex_list[1, 0:3], 
                                ti.Vector([0.0, 0.0, -1.0])
                                )
            simp_dim = 2
            #ti.loop_config(serialize=True)
            for loop_num in range(self.GJK_MAX_NUM_ITERATIONS):
                if(contact_flag == 0.0): # skip if contact_flag indicates early exist
                    a1 = self.support(i, -search_dir)
                    # print("a1", a1)
                    a2 = self.support(j, search_dir)
                    # print("a2", a2)
                    v3f_simplex_list[0, 0:3] = a2 - a1

                    if(ti.math.dot(v3f_simplex_list[0, 0:3], search_dir)<0.0):
                        contact_flag = -1.0
                    if(contact_flag == 0.0):
                        simp_dim += 1
                        if(simp_dim == 3):

                            # print("3th: v3f_simplex_list: ", v3f_simplex_list)
                            # print("search_dir: ", search_dir)
                            # print("num: ", simp_dim)

                            v3f_simplex_list, simp_dim, search_dir = self.update_simplex3_(v3f_simplex_list)

                            # print("3th: v3f_simplex_list after: ", v3f_simplex_list)
                            # print("search_dir: ", search_dir)
                            # print("num: ", simp_dim)

                        else:

                            # print("4th: v3f_simplex_list: ", v3f_simplex_list)
                            # print("search_dir: ", search_dir)
                            # print("search_dir: ", simp_dim)

                            contact, v3f_simplex_list, simp_dim, search_dir = self.update_simplex4_(v3f_simplex_list)
                            
                            # print("4th: v3f_simplex_list after: ", v3f_simplex_list)
                            # print("search_dir: ", search_dir)
                            # print("search_dir: ", simp_dim)

                            if(contact == 1.0):
                                contact_flag = 1.0
                            #     # contact_vec = self.EPA(v3f_simplex_list, i, j)
        ti.static_print("compile gjk: end")
        return contact_flag, contact_vec      
    
    # ============ Sampling point level-set algorithm ============
    @ti.func
    def query_sdf_cylinder_(self, i, pt_world):
        #using the sampling point pt_world to query the sdf of cylinder i 
        inv_matrot = ti.math.inverse(self.m3f_mat_r[i])
        pos = self.v3f_pos[i]
        pt_scaled = inv_matrot @ (pt_world - pos)
        mat_s = self.m3f_mat_s[i]
        scale_x_vec = mat_s[0:3, 0]
        scale_y_vec = mat_s[0:3, 1]
        scale_z_vec = mat_s[0:3, 2]
        # from -1.0 to 1.0
        ws_cap = ti.math.length(scale_y_vec)
        ws_base = - ws_cap
        # standard r = 1.0
        ws_r = (ti.math.length(scale_x_vec) + ti.math.length(scale_z_vec)) / 2.0 # radius
        dist = 0.0 # assume a large number first for the answer
        vec = ti.Vector([0.0, 0.0, 0.0])
        pt_r_vec = ti.Vector([pt_scaled.x, 0, pt_scaled.z])
        pt_r_vec_n = pt_r_vec.normalized()
        pt_r = ti.math.length(pt_r_vec)
        if (pt_scaled.y > ws_cap or pt_scaled.y < ws_base or pt_r > ws_r): # outside column
            # 5 situations
            if(pt_r <= ws_r):
                if(pt_scaled.y > ws_cap):
                    dist = ws_cap - pt_scaled.y
                    vec = ti.Vector([0.0, 1.0, 0.0])
                else:
                    dist = pt_scaled.y - ws_base
                    vec = ti.Vector([0.0, -1.0, 0.0])
            else:
                if(pt_scaled.y > ws_cap):
                    dist = -ti.math.length(ti.Vector([pt_scaled.y - ws_cap, pt_r - ws_r]))
                    vec = (pt_r_vec_n * (pt_r - ws_r) + ti.Vector([0.0, 1.0, 0.0]) * (pt_scaled.y - ws_cap)).normalized()
                elif (pt_scaled.y < ws_base):
                    dist = -ti.math.length(ti.Vector([ws_base - pt_scaled.y, pt_r - ws_r]))
                    vec = (pt_r_vec_n * (pt_r - ws_r) + ti.Vector([0.0, -1.0, 0.0]) * (ws_base - pt_scaled.y)).normalized()
                else:
                    dist = ws_r - pt_r
                    vec = pt_r_vec_n * (pt_r - ws_r)
        else: # in column
            dist_vec = ti.Vector([
                        ws_cap - pt_scaled.y, 
                        pt_scaled.y - ws_base,
                        ws_r - pt_r
                        ])
            norm_vec = ti.Matrix([
                        [0.0 ,1.0, 0.0], 
                        [0.0 ,-1.0, 0.0],
                        [0.0, 0.0, 0.0]
                        ])
            norm_vec[2, 0:3] = pt_r_vec_n
            dist = dist_vec[0]
            vec = norm_vec[0, 0:3]
            ti.loop_config(serialize=True)
            for ii in range(3):
                if(dist_vec[ii] < dist):
                    dist = dist_vec[ii]
                    vec = norm_vec[ii, 0:3]
        return dist, self.m3f_mat_r[i] @ vec
    
    @ti.func
    def query_sdf_box_(self, i, pt_world):
        #using the sampling point pt_world to query the sdf of cylinder i 
        inv_matrot = ti.math.inverse(self.m3f_mat_r[i])
        pos = self.v3f_pos[i]
        pt_scaled = inv_matrot @ (pt_world - pos)
        mat_s = self.m3f_mat_s[i]
        scale_x_vec = mat_s[0:3, 0]
        scale_y_vec = mat_s[0:3, 1]
        scale_z_vec = mat_s[0:3, 2]
        x_limit = ti.math.length(scale_x_vec)
        y_limit = ti.math.length(scale_y_vec)
        z_limit = ti.math.length(scale_z_vec)
        dist = ti.math.max(ti.math.max(x_limit, y_limit), z_limit) * 10
        vec = ti.Vector([0.0, 0.0, 0.0])

        if(pt_scaled.x > x_limit or pt_scaled.x < -x_limit or
        pt_scaled.y > y_limit or pt_scaled.y < -y_limit or
        pt_scaled.z > z_limit or pt_scaled.z < -z_limit): ## out
            signX = 1.0
            X = pt_scaled.x
            if(X < 0.0):
                signX = -1.0
                X = -X
            
            signY = 1.0
            Y = pt_scaled.y
            if(Y < 0.0):
                signY = -1.0
                Y = -Y
            signZ = 1.0
            Z = pt_scaled.z
            if(Z < 0.0):
                signZ = -1.0
                Z = -Z
            ## 7 zones
            if(X > x_limit and Y > y_limit and Z > z_limit): # corner
                temp_vec = ti.Vector([X, Y, Z]) - ti.Vector([x_limit, y_limit, z_limit])
                dist = -ti.math.length(temp_vec)
                vec = temp_vec.normalized() 
            elif(X > x_limit and Y > y_limit): # z edge
                temp_vec = ti.Vector([X, Y]) - ti.Vector([x_limit, y_limit])
                dist = -ti.math.length(temp_vec)
                vec = ti.Vector([temp_vec.x, temp_vec.y, 0.0]).normalized()
            elif(X > x_limit and Z > z_limit): # y edge
                temp_vec = ti.Vector([X, Z]) - ti.Vector([x_limit, z_limit])
                dist = -ti.math.length(temp_vec)
                vec = ti.Vector([temp_vec.x, 0.0, temp_vec.y]).normalized()
            elif(Y > y_limit and Z > z_limit): # x edge
                temp_vec = ti.Vector([Y, Z]) - ti.Vector([y_limit, z_limit])
                dist = -ti.math.length(temp_vec)
                vec = ti.Vector([0.0, temp_vec.x, temp_vec.y]).normalized()
            elif(X > x_limit):
                dist = x_limit - X
                vec = ti.Vector([1.0, 0.0, 0.0])
            elif(Y > y_limit):
                dist = y_limit - Y
                vec = ti.Vector([0.0, 1.0, 0.0])
            elif(Z > z_limit):
                dist = z_limit - Z
                vec = ti.Vector([0.0, 0.0, 1.0])
            vec.x = vec.x * signX
            vec.y = vec.y * signY
            vec.z = vec.z * signZ
        else: ## inside
            dist_vec = ti.Vector([
                        x_limit - pt_scaled.x, pt_scaled.x + x_limit,
                        y_limit - pt_scaled.y, pt_scaled.y + y_limit,
                        z_limit - pt_scaled.z, pt_scaled.z + z_limit
                        ])
            norm_vec = ti.Matrix([
                        [1.0 ,0.0, 0.0], [-1.0 ,0.0, 0.0],
                        [0.0 ,1.0, 0.0], [0.0 ,-1.0, 0.0],
                        [0.0 ,0.0, 1.0], [0.0 ,0.0, -1.0],
                        ])
            dist = ti.math.min(dist, dist_vec[0])
            vec = norm_vec[0, 0:3]
            ti.loop_config(serialize=True)
            for ii in range(6):
                if(dist_vec[ii] < dist):
                    dist = dist_vec[ii]
                    vec = norm_vec[ii, 0:3]
        return dist, self.m3f_mat_r[i] @ vec

    @ti.func
    def query_sdf(self, i, pt_world):
        dist = -1.0
        vec = ti.Vector([0.0, 0.0, 0.0])
        if(self.sf_type[i] == 0):
            dist, vec = self.query_sdf_cylinder_(i, pt_world)
        elif(self.sf_type[i] == 1):
            dist, vec = self.query_sdf_box_(i, pt_world)
        return dist, vec
    
    @ti.func
    def get_contact_manifold_cylinder(self, i, j):
        # using the sampling point of ith cylinder to query the LS of j th DEM
        # update the active sampling point table as a result
        active_count = 0
        ti.loop_config(serialize=True)
        for ii in range(self.n_sp_cylinder):
            pt_local = self.v3f_sample_points_cylinder[ii]
            pt_world = self.m3f_mat_sr[i] @ pt_local + self.v3f_pos[i]
            dist, vec = self.query_sdf(j, pt_world)
            if(dist > 0.0):
                self.f_active_sp_cylinder[i, active_count] = ii
                self.f_active_contact_dist_cylinder[i, active_count] = dist
                self.f_active_contact_vec_cylinder[i, active_count] = vec
                self.f_active_sp_table_cylinder[i, ii] = 1 # a flag indicating that the fs of the ii th sample point will be cached later
                active_count += 1
        return active_count # the right limit of active count

    @ti.func
    def get_contact_manifold_box(self, i, j):
        # using the sampling point of ith box to query the LS of j th DEM
        # update the active sampling point table as a result
        active_count = 0
        ti.loop_config(serialize=True)
        for ii in range(self.n_sp_box):
            pt_local = self.v3f_sample_points_box[ii]
            pt_world = self.m3f_mat_sr[i] @ pt_local + self.v3f_pos[i]
            dist, vec = self.query_sdf(j, pt_world)
            if(dist > 0.0):
                self.f_active_sp_box[i, active_count] = ii
                self.f_active_contact_dist_box[i, active_count] = dist
                self.f_active_contact_vec_box[i, active_count] = vec
                self.f_active_sp_table_box[i, ii] = 1 # a flag indicating that the fs of the ii th sample point will be cached later
                active_count += 1
        return active_count # the right limit of active count

    # ============ contact_resolution (DEM dynamics) ============
    @ti.func
    def get_inertia_mat_cylinder_(self, i):
        # return the local space inertia matrix of a cylinder
        mat_s = self.m3f_mat_s[i]
        scale_x_vec = mat_s[0:3, 0]
        scale_y_vec = mat_s[0:3, 1]
        scale_z_vec = mat_s[0:3, 2]
        h = 2 * ti.math.length(scale_y_vec)
        r = (ti.math.length(scale_x_vec) + ti.math.length(scale_z_vec)) / 2.0
        mass = self.sf_mass[i]
        a11 = a33 = (1.0 / 12.0) * mass * (3 * r * r + h * h)
        a22 = r * r * mass / 2.0
        ans = ti.Matrix([
            [a11, 0.0, 0.0],
            [0.0, a22, 0.0],
            [0.0, 0.0, a33]
        ])
        return ans

    @ti.func
    def get_inertia_mat_box_(self, i):
        # return the local space inertia matrix of a box
        mat_s = self.m3f_mat_s[i]
        scale_x_vec = mat_s[0:3, 0]
        scale_y_vec = mat_s[0:3, 1]
        scale_z_vec = mat_s[0:3, 2]
        x_len = ti.math.length(scale_x_vec)
        y_len = ti.math.length(scale_y_vec)
        z_len = ti.math.length(scale_z_vec)
        mass = self.sf_mass[i]
        xx = x_len * x_len
        yy = y_len * y_len
        zz = z_len * z_len
        a11 = (1.0 / 12.0) * mass * (yy + zz)
        a22 = (1.0 / 12.0) * mass * (xx + zz)
        a33 = (1.0 / 12.0) * mass * (xx + yy)
        ans = ti.Matrix([
            [a11, 0.0, 0.0],
            [0.0, a22, 0.0],
            [0.0, 0.0, a33]
        ])
        return ans   
    
    @ti.func
    def get_inertia_mat_(self, i):
        ans = ti.Matrix.identity(n=3, dt=ti.f32)
        if self.sf_type[i] == 0:
            ans = self.get_inertia_mat_cylinder_(i)
        elif self.sf_type[i] == 1:
            ans = self.get_inertia_mat_box_(i)
        return ans
    
    @ti.func
    def contact_f_via_cur_act_cp_cylinder(self, i, j, active_num, contact_vec, dt, kn, knd, ks, ksd, mu):
        # compute contact force and update the force field based on the current active contact manifold
        # should be invoked right after the contact manifold is built
        if self.sf_type[i] == 0:
            veli = self.v3f_lin_mv[i] * 1.0 / self.sf_mass[i]  # rigid body velocity
            velj = self.v3f_lin_mv[j] * 1.0 / self.sf_mass[j]  # rigid body velocity

            # TODO: compute I0j based on types

            # angle_vel for i
            I0i = self.get_inertia_mat_(i) # local space inertia matrix
            inv_I0i = ti.math.inverse(I0i)
            mat_roti = self.m3f_mat_r[i]
            inv_mat_roti = mat_roti.transpose()
            inv_Ii = mat_roti @ inv_I0i @ inv_mat_roti # global inertia tensor
            angl_veli = inv_Ii @ self.v3f_ang_mv[i] # anglular velocity
            # angle_vel for j
            I0j = self.get_inertia_mat_(j) # local space inertia matrix
            inv_I0j = ti.math.inverse(I0j)
            mat_rotj = self.m3f_mat_r[j]
            inv_mat_rotj = mat_rotj.transpose()
            inv_Ij = mat_rotj @ inv_I0j @ inv_mat_rotj # global inertia tensor
            angl_velj = inv_Ij @ self.v3f_ang_mv[j] # anglular velocity

            # normalization factor
            sum_depth = self.EPSILON
            sum_ds = self.EPSILON
            sum_n_damping_mag = self.EPSILON

            # some geometric variables
            # n = contact_vec.normalized()
            posi = self.v3f_pos[i]
            posj = self.v3f_pos[j]
            for ii in range(active_num): # loop over contact points to accumulate normalization factor
                pt_id = self.f_active_sp_cylinder[i, ii]
                pt_global = self.m3f_mat_sr[i] @ self.v3f_sample_points_cylinder[pt_id] + posi
                r_veci = pt_global - posi
                r_vecj = pt_global - posj
                contact_veli = veli + ti.math.cross(angl_veli, r_veci)
                contact_velj = velj + ti.math.cross(angl_velj, r_vecj)
                contact_vel = contact_veli - contact_velj

                d = self.f_active_contact_dist_cylinder[i, ii]
                n = self.f_active_contact_vec_cylinder[i, ii]
                # if(i == 0):
                #     print("n: ", n)
                #     print("contact_vec: ", contact_vec)
                vel_norm_mag = ti.math.dot(n, contact_vel) # normal vel component (with direction)
                tan_vel = contact_vel - vel_norm_mag * n
                sum_depth += d
                sum_ds += ti.math.length(tan_vel)
                sum_n_damping_mag += ti.abs(vel_norm_mag)
            # force
            M = ti.Vector([0.0, 0.0, 0.0]) # rotation moment
            M_d = ti.Vector([0.0, 0.0, 0.0]) # damping rotration moment
            F_n = ti.Vector([0.0, 0.0, 0.0]) # normal force
            F_nd = ti.Vector([0.0, 0.0, 0.0]) # normal damping force
            F_s = ti.Vector([0.0, 0.0, 0.0]) # shear force
            # F_sd = ti.Vector([0.0, 0.0, 0.0])# damping shear force
            # computing force
            for ii in range(active_num): # loop over contact points
                pt_id = self.f_active_sp_cylinder[i, ii]
                pt_global = self.m3f_mat_sr[i] @ self.v3f_sample_points_cylinder[pt_id] + self.v3f_pos[i]
                r_veci = pt_global - posi
                r_vecj = pt_global - posj
                contact_veli = veli + ti.math.cross(angl_veli, r_veci)
                contact_velj = velj + ti.math.cross(angl_velj, r_vecj)
                contact_vel = contact_veli - contact_velj


                d = self.f_active_contact_dist_cylinder[i, ii]
                n = self.f_active_contact_vec_cylinder[i, ii]
                vel_norm_mag = ti.math.dot(n, contact_vel) # normal vel component (with direction)
                tan_vel = contact_vel - vel_norm_mag * n

                Fs = mat_roti @ self.v3f_cached_fs_local_cylinder[i, pt_id]
                Fs += ks * -tan_vel * dt
                Fn = kn * d * n
                Fs_mag = ti.math.length(Fs)
                Fn_mag = ti.math.length(Fn)
                if (Fs_mag > Fn_mag * mu): # limited by friction coef
                    Fs = Fs.normalized() * Fn_mag * mu
                Fd = knd * -vel_norm_mag * n
                F_n += Fn * d / sum_depth
                F_s += Fs * ti.math.length(tan_vel) / sum_ds
                self.v3f_cached_fs_local_cylinder[i, pt_id] = inv_mat_roti @ Fs # cached local Fs
                F_nd += Fd * ti.abs(vel_norm_mag) / sum_n_damping_mag
                M += ti.math.cross(r_veci, Fn * d / sum_depth + Fs * ti.math.length(tan_vel) / sum_ds)
                M_d += ti.math.cross(r_veci, Fd * ti.abs(vel_norm_mag) / sum_n_damping_mag)

            self.v3f_moment[i] += M # rotation moment
            self.v3f_moment_d[i] += M_d # rotation damping moment
            self.v3f_fn[i] += F_n # normal force
            self.v3f_fnd[i] += F_nd # normal damping force
            self.v3f_fs[i] += F_s # shear force
    
    @ti.func
    def contact_f_via_cur_act_cp_box(self, i, j, active_num, contact_vec, dt, kn, knd, ks, ksd, mu):
        # compute contact force and update the force field based on the current active contact manifold
        # should be invoked right after the contact manifold is built
        if self.sf_type[i] == 1:
            veli = self.v3f_lin_mv[i] * 1.0 / self.sf_mass[i]  # rigid body velocity
            velj = self.v3f_lin_mv[j] * 1.0 / self.sf_mass[j]  # rigid body velocity
            # angle_vel for i
            I0i = self.get_inertia_mat_(i) # local space inertia matrix
            inv_I0i = ti.math.inverse(I0i)
            mat_roti = self.m3f_mat_r[i]
            inv_mat_roti = mat_roti.transpose()
            inv_Ii = mat_roti @ inv_I0i @ inv_mat_roti # global inertia tensor
            angl_veli = inv_Ii @ self.v3f_ang_mv[i] # anglular velocity
            # angle_vel for j
            I0j = self.get_inertia_mat_(j) # local space inertia matrix
            inv_I0j = ti.math.inverse(I0j)
            mat_rotj = self.m3f_mat_r[j]
            inv_mat_rotj = mat_rotj.transpose()
            inv_Ij = mat_rotj @ inv_I0j @ inv_mat_rotj # global inertia tensor
            angl_velj = inv_Ij @ self.v3f_ang_mv[j] # anglular velocity

            # normalization factor
            sum_depth = self.EPSILON
            sum_ds = self.EPSILON
            sum_n_damping_mag = self.EPSILON

            # some geometric variables
            # n = contact_vec.normalized()
            posi = self.v3f_pos[i]
            posj = self.v3f_pos[j]
            for ii in range(active_num): # loop over contact points to accumulate normalization factor
                pt_id = self.f_active_sp_box[i, ii]
                pt_global = self.m3f_mat_sr[i] @ self.v3f_sample_points_box[pt_id] + posi
                r_veci = pt_global - posi
                r_vecj = pt_global - posj
                contact_veli = veli + ti.math.cross(angl_veli, r_veci)
                contact_velj = velj + ti.math.cross(angl_velj, r_vecj)
                contact_vel = contact_veli - contact_velj

                d = self.f_active_contact_dist_box[i, ii]
                n = self.f_active_contact_vec_box[i, ii]
                # if(i == 0):
                #     print("n: ", n)
                #     print("contact_vec: ", contact_vec)
                vel_norm_mag = ti.math.dot(n, contact_vel) # normal vel component (with direction)
                tan_vel = contact_vel - vel_norm_mag * n
                sum_depth += d
                sum_ds += ti.math.length(tan_vel)
                sum_n_damping_mag += ti.abs(vel_norm_mag)
            # force
            M = ti.Vector([0.0, 0.0, 0.0]) # rotation moment
            M_d = ti.Vector([0.0, 0.0, 0.0]) # damping rotration moment
            F_n = ti.Vector([0.0, 0.0, 0.0]) # normal force
            F_nd = ti.Vector([0.0, 0.0, 0.0]) # normal damping force
            F_s = ti.Vector([0.0, 0.0, 0.0]) # shear force
            # F_sd = ti.Vector([0.0, 0.0, 0.0])# damping shear force
            # computing force
            for ii in range(active_num): # loop over contact points
                pt_id = self.f_active_sp_box[i, ii]
                pt_global = self.m3f_mat_sr[i] @ self.v3f_sample_points_box[pt_id] + self.v3f_pos[i]
                r_veci = pt_global - posi
                r_vecj = pt_global - posj
                contact_veli = veli + ti.math.cross(angl_veli, r_veci)
                contact_velj = velj + ti.math.cross(angl_velj, r_vecj)
                contact_vel = contact_veli - contact_velj


                d = self.f_active_contact_dist_box[i, ii]
                n = self.f_active_contact_vec_box[i, ii]
                vel_norm_mag = ti.math.dot(n, contact_vel) # normal vel component (with direction)
                tan_vel = contact_vel - vel_norm_mag * n

                Fs = mat_roti @ self.v3f_cached_fs_local_box[i, pt_id]
                Fs += ks * -tan_vel * dt
                Fn = kn * d * n
                Fs_mag = ti.math.length(Fs)
                Fn_mag = ti.math.length(Fn)
                if (Fs_mag > Fn_mag * mu): # limited by friction coef
                    Fs = Fs.normalized() * Fn_mag * mu
                Fd = knd * -vel_norm_mag * n
                F_n += Fn * d / sum_depth
                F_s += Fs * ti.math.length(tan_vel) / sum_ds
                self.v3f_cached_fs_local_box[i, pt_id] = inv_mat_roti @ Fs # cached local Fs
                F_nd += Fd * ti.abs(vel_norm_mag) / sum_n_damping_mag
                M += ti.math.cross(r_veci, Fn * d / sum_depth + Fs * ti.math.length(tan_vel) / sum_ds)
                M_d += ti.math.cross(r_veci, Fd * ti.abs(vel_norm_mag) / sum_n_damping_mag)

            self.v3f_moment[i] += M # rotation moment
            self.v3f_moment_d[i] += M_d # rotation damping moment
            self.v3f_fn[i] += F_n # normal force
            self.v3f_fnd[i] += F_nd # normal damping force
            self.v3f_fs[i] += F_s # shear force




    @ti.func
    def bounding_sphere_check(self, x_p, bs_r):
        ans = False
        for i in range(self.n_pillar):
            posDEM = self.v3f_pos[i]
            if(ti.math.length(posDEM - x_p) < bs_r):
                ans = True
                break
        return ans

    @ti.func
    def get_contact_force_MPM(self, x_p, v_p, kn, knd, r): # the force from DEM to particle
        # compute coupling force from MPM using LSDEM, the MP is treated as a sampling point
        # will add forces to DEM
        force_total = ti.Vector([0.0,0.0,0.0])
        for i in range(self.n_pillar):
            if self.sf_type[i] == 0:
                sdf_val, vec = self.query_sdf(i, x_p)
                dist = sdf_val + r
                if(dist > 0.0):
                    self.sf_touch_MP[i] = 1
                    velDEM = self.v3f_lin_mv[i] * 1.0 / self.sf_mass[i]  # rigid body velocity
                    # angle_vel for i
                    I0DEM = self.get_inertia_mat_(i) # local space inertia matrix

                    inv_I0DEM = ti.math.inverse(I0DEM)
                    mat_rotDEM = self.m3f_mat_r[i]
                    inv_mat_rotDEM = mat_rotDEM.transpose()
                    inv_IDEM = mat_rotDEM @ inv_I0DEM @ inv_mat_rotDEM # global inertia tensor
                    angl_velDEM = inv_IDEM @ self.v3f_ang_mv[i] # anglular velocity
                    posDEM = self.v3f_pos[i]
                    r_vec = x_p - posDEM
                    contact_velDEM = velDEM  + ti.math.cross(angl_velDEM, r_vec)
                    contact_vel = v_p - contact_velDEM
                    d = dist
                    n = vec.normalized()
                    vel_norm_mag = ti.math.dot(n, contact_vel) # normal vel component (with direction)
                    Fn = kn * ti.min(d, r) * n # cap the contact force for stability reasons
                    Fd = knd * -vel_norm_mag * n
                    M = ti.math.cross(r_vec, Fn)
                    M_d = ti.math.cross(r_vec, Fd)
                    # add reaction force to DEM 
                    self.v3f_moment[i] -= M # rotation moment
                    self.v3f_moment_d[i] -= M_d # rotation damping moment
                    self.v3f_fn[i] -= Fn # normal force
                    self.v3f_fnd[i] -= Fd # normal damping force
                    force_total += Fn + Fd
        return force_total

    @ti.func
    def get_contact_force_MPM_potential(self, x_p, v_p, potential_coeff_k, Cn, m, r, targettype = 0): # the force from DEM to particle
        force_total = ti.Vector([0.0,0.0,0.0])
        for i in range(self.n_pillar):
            if self.sf_type[i] == targettype:
                sdf_val, vec = self.query_sdf(i, x_p)
                penetration = sdf_val + r
                # potential based DEM force. See preprint by Yupeng Jiang et. al. 
                # https://www.researchgate.net/publication/353677081_Hybrid_continuum-discrete_simulation_of_granular_impact_dynamics
                xee = r - penetration
                if(xee < r):
                    xee = ti.max(xee, 0.001 * r)
                    self.sf_touch_MP[i] = 1
                    posDEM = self.v3f_pos[i]
                    r_vec = x_p - posDEM
                    n = vec.normalized()
                    # get the velocity
                    velDEM = self.v3f_lin_mv[i] * 1.0 / self.sf_mass[i]  # rigid body velocity
                    # angle_vel for i
                    I0DEM = self.get_inertia_mat_(i) # local space inertia matrix
                    inv_I0DEM = ti.math.inverse(I0DEM)

                    mat_rotDEM = self.m3f_mat_r[i]
                    inv_mat_rotDEM = mat_rotDEM.transpose()
                    inv_IDEM = mat_rotDEM @ inv_I0DEM @ inv_mat_rotDEM # global inertia tensor
                    angl_velDEM = inv_IDEM @ self.v3f_ang_mv[i] # anglular velocity
                    posDEM = self.v3f_pos[i]
                    r_vec = x_p - posDEM
                    contact_velDEM = velDEM  + ti.math.cross(angl_velDEM, r_vec)
                    contact_vel = v_p - contact_velDEM
                    contact_vel_n = ti.math.dot(n, contact_vel) * n # normal vel component (with direction)
                    # get the normal force
                    normal_force_scalar = potential_coeff_k*(xee-r)*(2*ti.log(xee/r)-r/xee+1)
                    keq = normal_force_scalar/penetration                            
                    Fn = normal_force_scalar * n
                    Fd = (-2 * Cn * ti.sqrt(m * keq)) * contact_vel_n # normal damping force
                    if Fd.dot(n) < 0: # tension
                        if Fd.norm() > Fn.norm():
                            Fd *=  Fn.norm()/Fd.norm() # no tension is allowed
                    M = ti.math.cross(r_vec, Fn)
                    M_d = ti.math.cross(r_vec, Fd)
                    # add reaction force to DEM 
                    self.v3f_moment[i] -= M # rotation moment
                    self.v3f_moment_d[i] -= M_d # rotation damping moment
                    self.v3f_fn[i] -= Fn # normal force
                    self.v3f_fnd[i] -= Fd # normal damping force
                    force_total += Fn + Fd
        return force_total

    @ti.func
    def reset_force_(self, i):
        # reset force so that it does not affect the next timestep
        self.v3f_moment[i] = ti.Vector([0.0, 0.0, 0.0]) 
        self.v3f_moment_d[i] = ti.Vector([0.0, 0.0, 0.0])
        self.v3f_fn[i] = ti.Vector([0.0, 0.0, 0.0])
        self.v3f_fnd[i] = ti.Vector([0.0, 0.0, 0.0])
        self.v3f_fs[i] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.func
    def reset_cached_fs_(self, i):
        # reset cached force if the contact manifold is not active
        for ii in range(self.n_sp_cylinder):
            if(self.f_active_sp_table_cylinder[i, ii] == 0): # no active contact, erase cached
                self.v3f_cached_fs_local_cylinder[i, ii] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self.f_active_sp_table_cylinder[i, ii] = 0 # set to zero for contact detection in next step 

        for ii in range(self.n_sp_box):
            if(self.f_active_sp_table_box[i, ii] == 0): # no active contact, erase cached
                self.v3f_cached_fs_local_box[i, ii] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self.f_active_sp_table_box[i, ii] = 0 # set to zero for contact detection in next step 

    @ti.func
    def time_integration_semi_explicit(self, i, dt):
        # semi-explicit trapezoidal time integration to update momentum and position.
        # performed after all force are computed
        # !!! will reset force after computation !!! #
        inv_mass = 1.0 / self.sf_mass[i] 
        I0 = self.get_inertia_mat_(i) # local space inertia matrix
        inv_I0 = ti.math.inverse(I0)
        mat_rot = self.m3f_mat_r[i]
        inv_mat_rot = ti.math.inverse(mat_rot)
        inv_I = mat_rot @ inv_I0 @ inv_mat_rot # global inertia tensor
        angl_vel_pre = inv_I @ self.v3f_ang_mv[i] # anglular velocity
        vel_pre = self.v3f_lin_mv[i] * inv_mass # rigid body velocity
        
        delta_lin_mv = (self.v3f_fn[i] + self.v3f_fnd[i] + self.v3f_fs[i]) * dt
        delta_ang_mv = (self.v3f_moment[i] + self.v3f_moment_d[i]) * dt

        self.v3f_lin_mv[i] += delta_lin_mv
        self.v3f_ang_mv[i] += delta_ang_mv

        angl_vel_now = inv_I @ self.v3f_ang_mv[i] # anglular velocity
        vel_now = self.v3f_lin_mv[i] * inv_mass # rigid body velocity
        # midpoint velocity
        angl_vel = (angl_vel_pre + angl_vel_now) / 2.0
        vel = (vel_pre + vel_now) / 2.0
        # rotation
        angl_vel_q = ti.Vector([angl_vel.x, angl_vel.y, angl_vel.z, 0.0])
        dq = quat_mult(angl_vel_q / 2.0, self.v4f_quat[i])
        self.v4f_quat[i] += dq * dt
        quat_len = ti.math.length(self.v4f_quat[i])
        if(quat_len != 0.0):
            self.v4f_quat[i] /= quat_len
        # position update
        self.v3f_pos[i] += vel * dt

    @ti.func
    def reset_status(self, i):
        # reset some helper variables
        self.reset_force_(i)
        self.reset_cached_fs_(i)
        self.set_mat_via_quat_(i)
        # reset flag
        # self.sf_touch_MP[i] = 0 do not set once it is activated








