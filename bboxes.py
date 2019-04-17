import numpy as np
from geometry import *

class BBox2D(object):

    def __init__(self, a, b, c, d):
        self.a = a 
        self.b = b
        self.c = c
        self.d = d 

    def __str__(self):
        return 'A: {}, B: {}, C: {}, D: {}'.format(self.a, self.b, self.c, self.d)

    @property
    def A(self):
        return self.a

    @property
    def B(self):
        return self.b

    @property
    def C(self):
        return self.c

    @property
    def D(self):
        return self.d

    def is_axis_aligned(self):
        raise NotImplementedError

    def area(self):
        raise NotImplementedError

    def translate(self, pt):
        self.a += pt
        self.b += pt
        self.c += pt
        self.d += pt 

    def rotate(self, theta):
        raise NotImplementedError

    def contains(self, pts):

        # TODO: should be optimised with recursive subdivision 
        #       (e.g. split into regular grid, check only corners and repeat only with boxes on the boundary)

        # M = point in pts
        # (0 < AM.AB < AB.AB) && (0 < AM.AD < AD.AD) (. == dot product)
        # https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle/190373#190373

        def vct(p1, p2):
            x = p2[0] - p1[0]
            y = p2[1] - p1[1]

            return np.array([x, y])

        AM = vct(self.a, pts)
        AB = vct(self.a, self.b)
        AD = vct(self.a, self.d)

        # single pt
        # AMdotAB = np.dot(AM, AB)
        # ABdotAB = np.dot(AB, AB)
        # AMdotAD = np.dot(AM, AD)
        # ADdotAD = np.dot(AD, AD)

        # vectorized
        n_cols = pts.shape[1]
        AMdotAB = np.sum(np.multiply(AM, np.tile(AB, (n_cols, 1)).T), axis=0) # [matrix column, vct column] dot product
        ABdotAB = np.dot(AB, AB)
        AMdotAD = np.sum(np.multiply(AM, np.tile(AD, (n_cols, 1)).T), axis=0)
        ADdotAD = np.dot(AD, AD)

        first_part  = ((0.0 <= AMdotAB) & (AMdotAB <= ABdotAB))
        second_part = ((0.0 <= AMdotAD) & (AMdotAD <= ADdotAD))

        return (first_part & second_part)

class BBox2DAxisAligned(BBox2D):

    def __init__(self, lt, rb): # always (x, y)

        assert(lt[0] <= rb[0])
        assert(lt[1] <= rb[1])

        x0 = lt[0] 
        y0 = lt[1]
        self.width  = rb[0] - lt[0]
        self.height = rb[1] - lt[1]

        a = lt                      # top-left
        b = (x0 + self.width, y0)   # top-right
        c = (x0, y0 + self.height)  # bottom-left
        d = rb                      # bottom-right

        BBox2D.__init__(self, a, b, c, d)

    def __str__(self):
        return 'Top-left: {}, bottom-right: {} (width: {}, height: {})'.format(self.lt, self.rb, self.width, self.height)

    def is_axis_aligned(self):
        return True

    @property
    def lt(self):
        assert(self.a[0] <= corner for corner in [self.b[0], self.c[0], self.d[0]])
        assert(self.a[1] <= corner for corner in [self.b[1], self.c[1], self.d[1]])
        return self.a

    @property
    def rb(self):
        assert(self.d[0] >= corner for corner in [self.a[0], self.b[0], self.c[0]])
        assert(self.d[1] >= corner for corner in [self.b[1], self.b[1], self.c[1]])
        return self.d

    '''
    @property
    def height(self):
        return self.height

    @property
    def width(self):
        return self.width
    '''

    @classmethod
    def from_xywh(cls, x, y, width, height):
        lt = (x, y)
        rb = (x + width, y + height)
        return cls(lt, rb)




class BBox3D(object):

    # TODO: allow passing enclosing 2D bbox

    def __init__(self, w3d, h3d, l3d, x3d = 0, y3d = 0, z3d = 0, rx = 0, ry = 0, rz = 0):

        self.w3d = w3d
        self.h3d = h3d
        self.l3d = l3d
        self.x3d = x3d
        self.y3d = y3d
        self.z3d = z3d
        self.rx = rx
        self.ry = ry
        self.rz = rz

        self.corners3D = self.__update_3D_corners()

    def __str__(self):
        return '3D bbox (dimension: {}, {}, {}; location: {}, {}, {}; orientation: {}, {}, {}'.format(self.w3d, self.h3d, self.w3d, self.x3d, self.y3d, self.z3d, self.rx, self.ry, self.rz)

    def __update_3D_corners(self):

        # compute rotational matrix around yaw axis
        R = eulerAnglesToRotationMatrix([self.rx, self.ry, self.rz])

        # 3D bounding box corners
        x_corners = [self.l3d/2,  self.l3d/2, -self.l3d/2, -self.l3d/2,  self.l3d/2,  self.l3d/2, -self.l3d/2, -self.l3d/2]
        y_corners = [         0,           0,           0,           0, -self.h3d  , -self.h3d  , -self.h3d  , -self.h3d  ]
        z_corners = [self.w3d/2, -self.w3d/2, -self.w3d/2,  self.w3d/2,  self.w3d/2, -self.w3d/2, -self.w3d/2,  self.w3d/2]

        # rotate and translate 3D bounding box
        corners_3D = np.matmul(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3D[0,:] = corners_3D[0,:] + self.x3d
        corners_3D[1,:] = corners_3D[1,:] + self.y3d
        corners_3D[2,:] = corners_3D[2,:] + self.z3d

        return corners_3D

    def get_3D_orientation(self): # not pre-computed as typically used only for debugging / plotting

        # compute rotational matrix around yaw axis
        R = eulerAnglesToRotationMatrix([self.rx, self.ry, self.rz])

        # orientation in object coordinate system
        orientation_3D = np.array([[0.0, self.l3d],
                                   [0.0, 0.0],
                                   [0.0, 0.0]])

        # rotate and translate in camera coordinate system, project in image
        orientation_3D      = np.matmul(R, orientation_3D)
        orientation_3D[0,:] = orientation_3D[0,:] + self.x3d
        orientation_3D[1,:] = orientation_3D[1,:] + self.y3d
        orientation_3D[2,:] = orientation_3D[2,:] + self.z3d

        return orientation_3D

    
    def is_valid(self):
        if self.w3d <= 0.0 or self.h3d <= 0.0 or self.l3d <= 0.0:
            return False

        return True

    @property
    def get_3D_corners(self):
        return self.corners3D


    def get_scaled_bbox3D(self, factor = (1.1, 1.1, 1.1)):
        
        return BBox3D(self.w3d * factor[0], self.h3d * factor[1], self.l3d * factor[2], 
                      self.x3d, self.y3d, self.z3d, 
                      self.rx, self.ry, self.rz)

    def __points_in_XZ_plane(self, pts_3D):

        # bottom part of 3D bbox
        # vectorized point in rectangle (~ 60x faster than above)
        A = np.array([self.corners3D[0, 0], self.corners3D[2, 0]])
        B = np.array([self.corners3D[0, 1], self.corners3D[2, 1]])
        C = np.array([self.corners3D[0, 2], self.corners3D[2, 2]])
        D = np.array([self.corners3D[0, 3], self.corners3D[2, 3]])

        XZ_bbox = BBox2D(A, B, C, D)

        return XZ_bbox.contains(pts_3D)

    def contains(self, pts_3D):

        # use only points with depth in min/max area
        min_box, max_box = self.get_enclosing_box3D()

        valid_x = (pts_3D[0, :] >= min_box[0]) & (pts_3D[0, :] <= max_box[0])
        valid_y = (pts_3D[1, :] >= min_box[1]) & (pts_3D[1, :] <= max_box[1])
        valid_z = (pts_3D[2, :] >= min_box[2]) & (pts_3D[2, :] <= max_box[2])
        valid_idxs = valid_x & valid_y & valid_z

        # test points in X-Z plane
        test_pts = np.array([pts_3D[0, valid_idxs], pts_3D[2, valid_idxs]])

        results = self.__points_in_XZ_plane(test_pts)

        # map back to the original array (which includes all points)
        in_bbox = np.full(pts_3D.shape[1], False)
        in_bbox[valid_idxs] = results

        return in_bbox

    '''
    def __str__(self):
        return 'A: {}, B: {}, C: {}, D: {}'.format(self.a, self.b, self.c, self.d)

    def is_axis_aligned(self):
        raise NotImplementedError

    def area(self):
        raise NotImplementedError

    def translate(self, pt):
        raise NotImplementedError

    def rotate(self, theta):
        raise NotImplementedError

    def move(self, pt, theta):
        raise NotImplementedError

    def scale(self, scale):
        raise NotImplementedError

    def get_min_bbox2D(self, P):
        raise NotImplementedError

    def contains(pts):
        raise NotImplementedError
    '''

    @property
    def pose(self):

        return (self.x3d, self.y3d, self.z3d), (self.rx, self.ry, self.rz)

    def get_enclosing_box3D(self):

        min_x = min(self.corners3D[0, :])
        max_x = max(self.corners3D[0, :])

        min_y = min(self.corners3D[1, :])
        max_y = max(self.corners3D[1, :])

        min_z = min(self.corners3D[2, :])
        max_z = max(self.corners3D[2, :])

        return (min_x, min_y, min_z), (max_x, max_y, max_z)

    
    def get_minimal_enclosing_bbox2D(self, P):

        corners_2D = projectToImage(P, self.get_3D_corners)

        lt = (int(min(corners_2D[0, :])), int(min(corners_2D[1, :])))
        rb = (int(max(corners_2D[0, :])), int(max(corners_2D[1, :])))

        return BBox2DAxisAligned(lt, rb)


    # @property
    # def A(self):
    #     return self.a

    # @property
    # def B(self):
    #     return self.b

    # @property
    # def C(self):
    #     return self.c

    # @property
    # def D(self):
    #     return self.d

    # @property
    # def E(self):
    #     return self.e

    # @property
    # def F(self):
    #     return self.f

    # @property
    # def G(self):
    #     return self.g

    # @property
    # def H(self):
    #     return self.h

    # @property
    # def min_x(self):
    #     return self.h

    # @property
    # def max_x(self):
    #     return self.h

    # @property
    # def min_y(self):
    #     return self.h

    # @property
    # def min_y(self):
    #     return self.h

    # @property
    # def min_z(self):
    #     return self.h

    # @property
    # def max_z(self):
    #     return self.h

