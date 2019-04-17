from open3d import *
import numpy as np
import pandas as pd # reading of vkitti camera poses gt
import os

import visualisation_utils as viz
import dynamic_fusion_utils as df_utils
from dynamic_map import *
from visualiser import *
import vkitti_data_utils as data_utils


class StaticFusion(object):

    def __init__(self, dataset_params, O3Dintrinsics, visualisation_params = None, 
        verbose = False, visualise_debug = False, read_mot = False):
        
        self.dataset_params = dataset_params
        self.visualisation_params = visualisation_params

        self.visualise_debug  = visualise_debug
        self.verbose          = verbose

        self.fusion           = None
        self.O3Dintrinsics    = O3Dintrinsics

        self.all_rgb_files    = None
        self.all_camera_poses = None
        self.all_mot          = None
        self.n_files = 0

        # hardcoded for time-being (for some reason, open3d visualiser does not support kitti intrinsics)
        # dataset_base_dir  = os.path.join('data', 'kitti_full', '0001')
        intrinsics_file_viz = os.path.join(self.dataset_params.base_dir, 'calib', 'naive_viz.json') 
        self.on_the_fly_intrinsics = read_pinhole_camera_intrinsic(intrinsics_file_viz)

        self.initialise(read_mot)


    def initialise(self, read_mot = False):
        
        # initialise volume
        self.fusion = DynamicMap(bg_voxel_length = self.dataset_params.bg_voxel_length, 
                                 bg_sdf_trunc    = self.dataset_params.bg_sdf_trunc, 
                                 bg_color_type   = self.dataset_params.bg_color_type,
                                 fg_voxel_length = self.dataset_params.fg_voxel_length,  
                                 fg_sdf_trunc    = self.dataset_params.fg_sdf_trunc, 
                                 fg_color_type   = self.dataset_params.fg_color_type)

        # get all image data
        self.all_rgb_files = os.listdir(self.dataset_params.rgb_dir)
        self.n_files = len(self.all_rgb_files)
        if self.verbose:
            print('Dataset contains {} files'.format(self.n_files)) 

        # get camera poses
        self.all_camera_poses = pd.read_csv(self.dataset_params.extrinsics_file, sep=" ", index_col=False)
        assert(self.n_files == len(self.all_camera_poses.index)),\
             'Number of files: {}, number of camera poses: {}'.format(self.n_files, len(self.all_camera_poses.index))

        # get motgt data if we're running fusion with dynamically moving objects
        if read_mot:
            self.all_mot = pd.read_csv(self.dataset_params.mot_file, sep=" ", index_col=False)


    def process_objects(self, color, depth, frame_idx):
        
        # static fusion doesn't care about objects
        return color, depth


    def process_frame(self, frame_idx):

        print('Processing image {}'.format(self.all_rgb_files[frame_idx]))

        # read inputs 
        color = read_image(os.path.join(self.dataset_params.rgb_dir,   self.all_rgb_files[frame_idx]))
        depth = read_image(os.path.join(self.dataset_params.depth_dir, self.all_rgb_files[frame_idx]))

        # compute or read camera pose
        camera_pose = np.reshape(self.all_camera_poses.values[frame_idx][1:,], (4, 4))
        if self.dataset_params.invert_extrinsics: # do we have world to origin or origin to world?
            camera_pose = np.linalg.inv(camera_pose)
        # if self.verbose:
        #     print(camera_pose)

        # process objects
        color, depth = self.process_objects(color, depth, frame_idx)
            
        # convert static parts into RGBD
        rgbd = create_rgbd_image_from_color_and_depth(color, depth,
                depth_trunc = self.dataset_params.depth_trunc, 
                convert_rgb_to_intensity = False, 
                depth_scale = self.dataset_params.depth_scale)
        # viz.plot_rgbd_image(rgbd)

        # integrate into bg volume
        self.fusion.integrate_background(rgbd, self.O3Dintrinsics, camera_pose, frame_idx)

        if self.visualisation_params is not None and self.visualisation_params.on_the_fly_visualisation:
            # IOFVisualiser().visualise_frame(self.fusion, self.O3Dintrinsics, frame_idx, self.visualisation_params)
            IOFVisualiser().visualise_frame(self.fusion, self.on_the_fly_intrinsics, frame_idx, self.visualisation_params)


    def reconstruct_scene(self, start = 0, limit_frames = -1):

        assert(start >= 0)
        stop = min(self.n_files, start + limit_frames) if limit_frames >= 0 else self.n_files
        
        for frame_idx in range(start, stop):
            self.process_frame(frame_idx)

        return self.fusion




class DynamicObjectsFusion(StaticFusion):

    def __init__(self, dataset_params, O3Dintrinsics, visualisation_params = None, invalidate_2D_bbox = False, verbose = False, visualise_debug = False):
        StaticFusion.__init__(self, dataset_params, O3Dintrinsics, visualisation_params, verbose, visualise_debug, True)

        self.invalidate_2D_bbox = invalidate_2D_bbox

    def update_pose_without_fusion_if_exists(self, object_pose, tracklet_id, frame_idx, msg = ''):
        print('Object {} will not be fused {}'.format(tracklet_id, msg)) # TODO: we should first check whether object exists
        if self.fusion.object_exists(tracklet_id):
            succes = self.fusion.update_pose_without_integration(object_pose, tracklet_id, frame_idx)
            if not succes:
                print('Object {} does not exists'.format(tracklet_id))

    def check_and_refine(self, object_rgbd, object_pose, tracklet_id):
        return object_pose, True, None

    def process_objects(self, color, depth, frame_idx):

        # get all frame objects
        all_frame_objects = df_utils.get_frame_objects(self.all_mot, frame_idx, labels = ['Car', 'DontCare'])

        # object to fuse and remove from bg volume
        objects_to_reconstruct = all_frame_objects[all_frame_objects['label'].isin(['Car'])]
        objects_to_remove = all_frame_objects[all_frame_objects['label'].isin(['Car', 'DontCare'])]

        # for each object, get tracklet id, bbox and pose
        for index, object_info in objects_to_reconstruct.iterrows(): 

            # get tracklet id
            tracklet_id = data_utils.extract_tracklet_idx(object_info) 
            if tracklet_id < 0:
                # this bouding box was not tracked
                continue

            #################################
            # bbox3d = data_utils.extract3Dbbox(object_info)
            # max_dst = bbox3d.get_enclosing_box3D()[1][2]
            # min_dst = bbox3d.get_enclosing_box3D()[0][2]

            # print('Bbox dst {}, {}'.format(min_dst, max_dst))
            # print(bbox3d)
            # viz.plot_image(depth)
            #################################

            skip_object, object_rgb, object_depth, object_pose = df_utils.get_cropped_bbox_and_pose(self.O3Dintrinsics.intrinsic_matrix, color, depth, 
                object_info, self.dataset_params.depth_scale, self.dataset_params.depth_trunc)

            if self.verbose:
                print('Fusing object ID: {}'.format(tracklet_id))

            # no point in 3D bbox has depth < depth_trunc
            if skip_object:
                self.update_pose_without_fusion_if_exists(object_pose, tracklet_id, frame_idx, '(no points with depth in [0, trunc])')
                continue 

            # convert into rgbd
            object_rgbd = create_rgbd_image_from_color_and_depth(object_rgb, object_depth,
                    depth_trunc = self.dataset_params.depth_trunc, 
                    convert_rgb_to_intensity = False, 
                    depth_scale = self.dataset_params.depth_scale)

            # we can check and or refine pose here
            object_pose, integrate_object, evaluation = self.check_and_refine(object_rgbd, object_pose, tracklet_id)

            # integrate
            if integrate_object:
                self.fusion.integrate_object(object_rgbd, self.O3Dintrinsics, object_pose, tracklet_id, frame_idx)
            else:
                self.update_pose_without_fusion_if_exists(object_pose, tracklet_id, frame_idx)

        # filter out moving objects from static bg 
        # TODO: here we can filter out all know objects, or only those we reconstruct as dynamic ones
        # TODO: objects which are being reconstructed should produce binary mask or indices that will do here instead of repeating in3Dbox test
        color, depth = df_utils.invalidate_all_objects(self.O3Dintrinsics.intrinsic_matrix, color, depth,
                        objects_to_remove, self.dataset_params.depth_scale, self.dataset_params.depth_trunc,
                        invalidate_2D_bbox = self.invalidate_2D_bbox)

        return color, depth





