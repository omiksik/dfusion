import numpy as np

# --------------------------------------------------------
def eulerAnglesToRotationMatrix(theta):
# --------------------------------------------------------

    assert(len(theta) == 3) 

    R_x = np.array([[1, 0,                  0               ],
                    [0, np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0, np.sin(theta[0]),  np.cos(theta[0]) ]])
         
    R_y = np.array([[np.cos(theta[1]),  0, np.sin(theta[1]) ],
                    [0,                 1,   0              ],
                    [-np.sin(theta[1]), 0, np.cos(theta[1]) ]])
                 
    R_z = np.array([[np.cos(theta[2]),  -np.sin(theta[2]), 0],
                    [np.sin(theta[2]),   np.cos(theta[2]), 0],
                    [0,                    0,              1]])
                     
    return np.dot(R_z, np.dot(R_y, R_x))


# --------------------------------------------------------
def compose_Rt(R, t):
# --------------------------------------------------------

	assert(R.shape == (3,3))
	assert(t.shape == (3,1))

	Rt = np.hstack([R, t])
	Rt = np.vstack([Rt, np.array([0, 0, 0, 1])])

	return Rt

# --------------------------------------------------------
def compose_Rt_from_euler(x3d, y3d, z3d, rx, ry, rz):
# --------------------------------------------------------

    R = eulerAnglesToRotationMatrix([rx, ry, rz])
    Rt = compose_Rt(R, np.array([[x3d], [y3d], [z3d]]))

    return Rt

# --------------------------------------------------------
def formCameraMatrix(K, R = np.identity(3), t = np.zeros((3, 1))):
# --------------------------------------------------------
# forms camera matrix P = K[R|t]
# Usage: P = formCameraMatrix(K, R, t)
#   input: K: 3x3 matrix (camera intrinsics)
#          R: 3x3 rotation matrix
#          t: 3x1 translation vector
#   output: P: 3x4 camera matrix

    # TODO: move to geometry

    assert(K.shape == (3, 3))
    assert(R.shape == (3, 3))
    assert(t.shape == (3, 1))

    P = np.matmul(K, np.hstack([R, t]))

    return P
    
# --------------------------------------------------------
def project_to_3D(K, x, y, depth):
# --------------------------------------------------------

    assert(K.shape == (3, 3))
    assert(len(x) == len(y))
    assert(len(x) == len(depth))

    fx_d = K[0][0]
    fy_d = K[1][1]
    cx_d = K[0][2]
    cy_d = K[1][2]

    x = (x - cx_d) * depth / fx_d
    y = (y - cy_d) * depth / fy_d
    z = depth

    return np.array([x, y, z])

# --------------------------------------------------------
def projectToImage(P, pts_3D):
# --------------------------------------------------------
# projects 3D points into the image plane using camera matrix P
# Usage: pts_2D = projectToImage(P, pts_3D)
#   input: pts_3D: 3xn matrix
#          P:      3x4 projection matrix
#   output: pts_2D: 2xn matrix

    # project in image
    pts_2D = np.matmul(P, np.vstack([pts_3D, np.ones((1, pts_3D.shape[1]))]))

    # normalize
    pts_2D[0,:] = np.divide(pts_2D[0,:], pts_2D[2,:])
    pts_2D[1,:] = np.divide(pts_2D[1,:], pts_2D[2,:])

    # remove last row
    pts_2D = pts_2D[:-1,:]

    return pts_2D
