import taichi as ti

@ti.func # generated via chat GPT, tobe validated
def rotation_matrix(degrees, axis):
    # """
    # Generates a 3x3 rotation matrix based on the rotation degrees and rotation axis provided.

    # Args:
    #     degrees (float): The rotation angle in degrees.
    #     axis (ti.Vector): A 3D vector representing the rotation axis.

    # Returns:
    #     A 3x3 Taichi matrix representing the rotation matrix.
    # """
    radians = ti.math.radians(degrees)
    c = ti.cos(radians)
    s = ti.sin(radians)
    v = axis.normalized()

    # Construct the components of the rotation matrix
    x = v[0]
    y = v[1]
    z = v[2]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z

    # Build the rotation matrix
    R = ti.Matrix([[xx * (1 - c) + c, xy * (1 - c) - z * s, xz * (1 - c) + y * s],
                   [xy * (1 - c) + z * s, yy * (1 - c) + c, yz * (1 - c) - x * s],
                   [xz * (1 - c) - y * s, yz * (1 - c) + x * s, zz * (1 - c) + c]])

    return R


@ti.func
def scale_mat3(scale): # generated via ChatGPT and checked by Liang Zhengyu
    # Generates a 3x3 scale matrix based on the scale vector provided.
    S = ti.Matrix.identity(ti.f64, 3)
    S[0, 0] = scale[0]
    S[1, 1] = scale[1]
    S[2, 2] = scale[2]

    return S

@ti.func
def quat_to_mat(q): # generated via ChatGPT and checked by Liang Zhengyu
    # Generates a 3x3 rotation matrix based on the q (quaternion) provided.
    # Extract quaternion components
    a, b, c, d = q[3], q[0], q[1], q[2]
    # Compute matrix elements
    ab = a * b
    ac = a * c
    ad = a * d
    bb = b * b
    bc = b * c
    bd = b * d
    cc = c * c
    cd = c * d
    dd = d * d

    # Compute the rotation matrix
    m00 = 1.0 - 2.0 * (cc + dd)
    m01 = 2.0 * (bc - ad)
    m02 = 2.0 * (bd + ac)
    m10 = 2.0 * (bc + ad)
    m11 = 1.0 - 2.0 * (bb + dd)
    m12 = 2.0 * (cd - ab)
    m20 = 2.0 * (bd - ac)
    m21 = 2.0 * (cd + ab)
    m22 = 1.0 - 2.0 * (bb + cc)

    return ti.Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])


@ti.func
def quat_mult(q1, q2):
    s1 = q1.w
    s2 = q2.w
    u1 = ti.Vector([q1.x, q1.y, q1.z])
    u2 = ti.Vector([q2.x, q2.y, q2.z])
    s = s1 * s2 - ti.math.dot(u1, u2)
    u = s1 * u2 + s2 * u1 + ti.math.cross(u1, u2)
    ans = ti.Vector([u.x, u.y, u.z, s])
    return ans

def normalize_quat(q):
    inv_mag = 1.0 / ti.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
    return ti.Vector([q.w * inv_mag, q.x * inv_mag, q.y * inv_mag, q.z * inv_mag])