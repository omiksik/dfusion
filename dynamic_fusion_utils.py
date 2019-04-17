from open3d import *
import numpy as np
from bboxes import *
from geometry import *
import vkitti_data_utils as data_utils
import math

import time

# --------------------------------------------------------
def get_padded_bbox2D(bbox2d, height, width, pad_x = 0, pad_y = 0):
# --------------------------------------------------------

    l = max(0, bbox2d.lt[0] - pad_x)
    t = max(0, bbox2d.lt[1] - pad_y)
    r = min(width,  bbox2d.rb[0] + pad_x)
    b = min(height, bbox2d.rb[1] + pad_y)

    return BBox2DAxisAligned((l, t), (r, b))


# --------------------------------------------------------
def get_pixel_coords_of_3Dbbox(K, rgb, depth, bbox3d, depth_scale, bbox_factor = (1.1, 1.1, 1.1)):
# --------------------------------------------------------
    
    # TODO: point in 3D bbox tests only X-Z plane (prob ok for cars)
    # TODO: we should use bottom 4 corners, do robust plane fitting (wth RANSAC) and measure point-to-plane distance to remove 'road'

    # TODO: just for time being
    assert(np.asarray(depth).dtype == np.uint16)

    # let's get extended bboxes and make sure their within image dims
    P = formCameraMatrix(K, R = np.identity(3), t = np.zeros((3, 1)))
    scaled_bbox3D = bbox3d.get_scaled_bbox3D(bbox_factor)
    bbox2d = scaled_bbox3D.get_minimal_enclosing_bbox2D(P)
    bbox2d = get_padded_bbox2D(bbox2d, np.asarray(depth).shape[0], np.asarray(depth).shape[1], 0, 0) # let's assume the bbox is uncertain

    # generate x-y coordinates 
    m = np.mgrid[bbox2d.lt[1]:bbox2d.rb[1], bbox2d.lt[0]:bbox2d.rb[0]]

    u = m[1].flatten()
    v = m[0].flatten()
    d = np.asarray(depth)[v, u].flatten() / depth_scale

    # project to 3D
    pts_in_3D = project_to_3D(K, u, v, d)

    # get indices of points inside BBox
    in_box = scaled_bbox3D.contains(pts_in_3D)

    # uv, ie image coordinates
    u_in = u[in_box == True]
    v_in = v[in_box == True]

    return u_in, v_in

# --------------------------------------------------------
def extract_3Dbbox_area(K, rgb, depth, bbox3d, depth_scale, bbox_factor = (1.1, 1.1, 1.1)):
# --------------------------------------------------------

    # check which pixel points are in 3D bbox
    u_in, v_in = get_pixel_coords_of_3Dbbox(K, rgb, depth, bbox3d, depth_scale, bbox_factor)

    rgb_np = np.asarray(rgb)
    depth_np = np.asarray(depth)

    rgb_modified = Image((np.zeros_like(rgb_np)).astype(np.uint8))
    depth_modified = Image((np.zeros_like(depth_np)).astype(np.uint16))

    np.asarray(rgb_modified)[v_in, u_in, :] = rgb_np[v_in, u_in, :]
    np.asarray(depth_modified)[v_in, u_in] = depth_np[v_in, u_in]

    return rgb_modified, depth_modified, v_in, u_in


# --------------------------------------------------------
def invalidate_3Dbbox_area(K, rgb, depth, bbox3d, depth_scale, bbox_factor = (1.1, 1.1, 1.1)):
# --------------------------------------------------------
    
    # check which pixel points are in 3D bbox
    u_in, v_in = get_pixel_coords_of_3Dbbox(K, rgb, depth, bbox3d, depth_scale, bbox_factor)

    np.asarray(rgb)[v_in, u_in, :] = 0
    np.asarray(depth)[v_in, u_in] = 0

    return rgb, depth

# --------------------------------------------------------
def invalidate_2Dbbox_area(rgb, depth, bbox2d, pad_x = 0, pad_y = 0):
# --------------------------------------------------------

    # convert to numpy
    rgb_np = np.asarray(rgb)
    depth_np = np.asarray(depth)

    # TODO: just for time being
    assert(rgb_np.dtype == np.uint8)
    assert(depth_np.dtype == np.uint16)

    assert(rgb_np.shape[2] == 3)
    assert(rgb_np.shape[0:2] == depth_np.shape)

    # let's get extended bboxes and make sure their within image dims
    bbox2d = get_padded_bbox2D(bbox2d, rgb_np.shape[0], rgb_np.shape[1], pad_x, pad_y)

    # invalidate
    np.asarray(rgb)[bbox2d.lt[1]:bbox2d.rb[1], bbox2d.lt[0]:bbox2d.rb[0], :] = 0
    np.asarray(depth)[bbox2d.lt[1]:bbox2d.rb[1], bbox2d.lt[0]:bbox2d.rb[0]] = 0

    return rgb, depth

# --------------------------------------------------------
def invalidate_all_objects(K, rgb, depth, objects_to_remove, depth_scale, depth_trunc = math.inf,
                           invalidate_2D_bbox = False): #True): 
# --------------------------------------------------------

    # for each object
    for index, bbox in objects_to_remove.iterrows(): 

        # # invalidate
        if invalidate_2D_bbox:
            bbox2d = data_utils.extract2Dbbox(bbox)
            rgb, depth = invalidate_2Dbbox_area(rgb, depth, bbox2d, 0, 0) # 10, 10)

        # get tracklet id
        tracklet_id = data_utils.extract_tracklet_idx(bbox) 
        if tracklet_id < 0:
            # this bouding box was not tracked
            continue


        # get 3D bbox 
        bbox3d = data_utils.extract3Dbbox(bbox)

        # is at least any 3D point < depth_trunc? (otherwise won't be fused anyway)
        # TODO: print some status like skipping or something to improve debugging
        if bbox3d.get_enclosing_box3D()[0][2] > depth_trunc:
            continue

        # TODO: we might need to be more careful here - we're leaving out some shadow
        rgb, depth = invalidate_3Dbbox_area(K, rgb, depth, bbox3d, depth_scale, (1.2, 1.2, 1.1))

    return rgb, depth

# --------------------------------------------------------
def get_frame_objects(all_mot, frame_idx, labels = ['Car']):
# --------------------------------------------------------
    # TODO: this should go to vkitti_data_utils

    # get all objects within current frame
    frame_objects = all_mot.loc[all_mot['frame'] == frame_idx]

    # only objects we're after
    if labels is not None:
        frame_objects = frame_objects[frame_objects['label'].isin(labels)]

    return frame_objects


# --------------------------------------------------------
def get_cropped_bbox_and_pose(K, rgb, depth, object_info, depth_scale, depth_trunc = math.inf):
# --------------------------------------------------------

    skip_object = True

    # get 3D bbox 
    bbox3d = data_utils.extract3Dbbox(object_info)

    # get pose
    translation, rotation = bbox3d.pose
    object_pose = compose_Rt_from_euler(*translation, *rotation)

    # is at least any 3D point < depth_trunc? (otherwise won't be fused anyway)
    # TODO: print some status like skipping or something to improve debugging
    if bbox3d.get_enclosing_box3D()[0][2] > depth_trunc:
        return skip_object, None, None, object_pose

    # invalidate stuff
    # bbox2d = data_utils.extract2Dbbox(object_info)
    # rgb_crop, depth_crop = extract_bbox_area(rgb, depth, bbox2d, 0, 0)

    rgb_crop, depth_crop, v_in, u_in = extract_3Dbbox_area(K, rgb, depth, bbox3d, depth_scale, (1.2, 1.2, 1.1))
    skip_object = False

    # is at least 1 point valid?
    if len(u_in) == 0 or len(v_in) == 0:
        skip_object = True

    # if more distant side of bbox is beyond depth_trunc, let's be careful
    if bbox3d.get_enclosing_box3D()[1][2] > depth_trunc:
        if (np.amin(np.asarray(depth)[v_in, u_in]) / depth_scale) > depth_trunc:
            skip_object = True

    return skip_object, rgb_crop, depth_crop, object_pose

