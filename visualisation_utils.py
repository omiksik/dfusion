import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import vkitti_data_utils as data_utils
from geometry import *
from open3d import *
import copy

##########################################################
# image utils
##########################################################
# --------------------------------------------------------
def plot_image(img, caption='Image'):
# --------------------------------------------------------

    fig = plt.figure(figsize=(16, 8))
    plt.title(caption)
    plt.imshow(img)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

# --------------------------------------------------------
def plot_rgbd_image(rgbd_image, rgb_caption='Image', depth_caption='Depth image'):
# --------------------------------------------------------

    fig = plt.figure()
    
    plt.subplot(1, 2, 1)
    plt.title(rgb_caption)
    plt.imshow(rgbd_image.color)

    plt.subplot(1, 2, 2)
    plt.title(depth_caption)
    plt.imshow(rgbd_image.depth)
    # plt.show()
    
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)


##########################################################
# 2D bbox utils
##########################################################
# --------------------------------------------------------
def draw_2Dbbox(img, bbox2d, colour = (0, 255, 0), line_width = 3):
# --------------------------------------------------------
    
    cv2.rectangle(img, bbox2d.lt, bbox2d.rb, colour, line_width)
    return img

# --------------------------------------------------------
def draw_2Dbboxes(img, bboxes, colour = (0, 255, 0), line_width = 3):
# --------------------------------------------------------

    for index, bbox in bboxes.iterrows(): 

        bbox2d = data_utils.extract2Dbbox(bbox)
        draw_2Dbbox(img, bbox2d, colour, line_width)

    return img


##########################################################
# 3D bbox utils
##########################################################

# --------------------------------------------------------
def computeBox3D(P, bbox3d):
# --------------------------------------------------------
# takes an object and a projection matrix (P) and projects the 3D
# bounding box into the image plane.

    # TODO: move to 3DBbox

    # index for 3D bounding box faces
    face_idx = np.array([[0,1,5,4],   # front face
                         [1,2,6,5],   # left face
                         [2,3,7,6],   # back face
                         [3,0,4,7]])  # right face

    corners_3D = bbox3d.get_3D_corners

    # only draw 3D bounding box for objects in front of the camera
    if np.any(corners_3D[2,:] < 0.1):
      return [[], []]

    # project the 3D bounding box into the image plane
    corners_2D = projectToImage(P, corners_3D)

    return [corners_2D, face_idx]

# --------------------------------------------------------
def computeOrientation3D(P, bbox3d):
# --------------------------------------------------------
# takes an object and a projection matrix (P) and projects the 3D
# object orientation vector into the image plane.

    orientation_3D = bbox3d.get_3D_orientation()

    # vector behind image plane?
    if np.any(orientation_3D[2,:] < 0.1):
      return []

    # project orientation into the image plane
    orientation_2D = projectToImage(P, orientation_3D);

    return orientation_2D

# --------------------------------------------------------
def draw_3Dbbox(img, box3d, face_idx, colour = (0, 255, 0), line_width = 2):
# --------------------------------------------------------
# box3d: 2x8 matrix of 3D coords projected into camera coords

    assert(box3d.shape == (2, 8))

    for f in range (0, 4):
        
        x_ = np.hstack([box3d[0, face_idx[f, :]], box3d[0,face_idx[f, 1]]])
        y_ = np.hstack([box3d[1, face_idx[f, :]], box3d[1,face_idx[f, 1]]])

        for pt_idx in range (1, 5):

            assert(len(x_) == 5)
            assert(len(y_) == 5)

            a = (int(x_[pt_idx-1]), int(y_[pt_idx-1]))
            b = (int(x_[pt_idx]),   int(y_[pt_idx]))

            cv2.line(img, a, b, colour, line_width)

    return img

# --------------------------------------------------------
def draw_3Dbox_orientation(img, orientation, colour_in = (0, 0, 0), 
    colour_out = (255, 255, 255), line_width = 2):
# --------------------------------------------------------

    assert(orientation.shape == (2, 2))

    a = (int(orientation[0][0]), int(orientation[1][0]))
    b = (int(orientation[0][1]), int(orientation[1][1]))

    cv2.line(img, a, b, colour_out, 2 * line_width + 1)
    cv2.line(img, a, b, colour_in, line_width)

    return img

# --------------------------------------------------------
def draw_3Dbboxes(img, P, bboxes, colour = (0, 255, 0), line_width = 3, draw_orientation = True):
# --------------------------------------------------------
# P: 3x4 camera matrix
# bboxes: bboxes in 3d coords

    for index, bbox in bboxes.iterrows(): 

        bbox3d = data_utils.extract3Dbbox(bbox)

        if not bbox3d.is_valid():
            print('Skipping 3D bbox - negative values, prob unused')
            continue

        box3d_in_camera_coords, face_idx = computeBox3D(P, bbox3d)

        img = draw_3Dbbox(img, box3d_in_camera_coords, face_idx, colour, line_width)

        # draw orientation vector?
        if draw_orientation:
            orientation = computeOrientation3D(P, bbox3d)
            img = draw_3Dbox_orientation(img, orientation)

            # bbox2d = bbox3d.get_minimal_enclosing_bbox2D(P)
            # img = draw_2Dbbox(img, bbox2d, colour = (0, 255, 0), line_width = 3)

    return img


##########################################################
# registration utils
##########################################################
'''
def draw_registration_result(source, target, transformation = None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])
'''

def draw_geometries_with_extrinsics(geometries, O3Dintrinsics, camera_pose, caption = 'visualisation'):
    vis = Visualizer()
    vis.create_window(window_name = caption, 
                      width = O3Dintrinsics.width, 
                      height = O3Dintrinsics.height)

    # optionally load visualiser options

    # add background
    for geometry in geometries:
        vis.add_geometry(geometry)

    # place camera
    traj = PinholeCameraTrajectory()
    traj.intrinsic = O3Dintrinsics
    traj.extrinsic = Matrix4dVector([camera_pose])

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(traj.intrinsic, traj.extrinsic[0])

    # run visualiser            
    vis.run()

def draw_registration_result_correspondences(source, target, O3Dintrinsics = None, 
                                             correspondence_set = None, transformation = None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    if correspondence_set is not None:
        idx = np.asarray(correspondence_set)
        np.asarray(source_temp.colors)[idx[:, 0], :] = [0, 1, 0]
        np.asarray(target_temp.colors)[idx[:, 1], :] = [0, 1, 0]

    if transformation is not None:
        source_temp.transform(transformation)

    if O3Dintrinsics is not None:
        draw_geometries_with_extrinsics([source_temp, target_temp], 
            O3Dintrinsics, np.linalg.inv(transformation), caption = 'registration result')
    else:
        draw_geometries([source_temp, target_temp])
