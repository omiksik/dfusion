from open3d import *
import numpy as np
import pandas as pd # reading of vkitti camera poses gt
import matplotlib.pyplot as plt
import argparse
import os
import time

import datasets 
import visualisation_utils as viz
from visualiser import *
from dynamic_objects_fusion import *

# import cv2

# --------------------------------------------------------
def print_intrinsics(intrinsics):
# --------------------------------------------------------
    print('Using camera intrinsics {}\n{}'.format(intrinsics, 
        intrinsics.intrinsic_matrix))

    print('Width: {}, Height: {}\n'.format(intrinsics.width,
        intrinsics.height))
 

# --------------------------------------------------------
def print_image_stats(image, caption='image'):
# --------------------------------------------------------

    print('{}: {}, {}, , min: {}, max: {}\n'.format(caption, image, np.asarray(image).dtype, 
        np.amin(np.asarray(image)), np.amax(np.asarray(image))))


# --------------------------------------------------------
def add_bool_arg(parser, name, default=False, description=''):
# --------------------------------------------------------
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help='Enable: ' + description)
    group.add_argument('--no-' + name, dest=name, action='store_false', help='Disable [default]: ' + description)
    parser.set_defaults(**{name:default})

# --------------------------------------------------------
def plot_object_data(dataset_params, pinhole_camera_intrinsic, opts):
# --------------------------------------------------------

    # get all image data
    all_rgb_files = os.listdir(dataset_params.rgb_dir)
    n_files = len(all_rgb_files)
    print('Dataset contains {} files'.format(n_files)) 
    
    # get motgt data
    all_mot = pd.read_csv(dataset_params.mot_file, sep=" ", index_col=False)
    # assert(n_files == len(all_mot.index))

    # get camera matrix for projections
    K = pinhole_camera_intrinsic.intrinsic_matrix
    P = viz.formCameraMatrix(K, R = np.identity(3), t = np.zeros((3, 1)))

    for i in range(max(0, opts.start), min(n_files, opts.start + opts.limit_frames)):
        print('Processing image {}'.format(all_rgb_files[i]))

        # read image (in each loop)
        color = read_image(os.path.join(dataset_params.rgb_dir, all_rgb_files[i]))

        # => numpy
        color_np = np.asarray(color)

        # get objects
        current_objects = all_mot.loc[all_mot['frame'] == i]
        current_cars = current_objects.loc[current_objects['label'] == 'Car']
        current_other = current_objects.loc[current_objects['label'] != 'Car']
        # print((current_objects))
        # print((current_cars))

        if opts.show_2Dbboxes:
            color_np = viz.draw_2Dbboxes(color_np, current_cars, colour = (0, 255, 0), line_width = 3)
            color_np = viz.draw_2Dbboxes(color_np, current_other, colour = (0, 0, 255), line_width = 3)

        if opts.show_3Dbboxes:
            color_np = viz.draw_3Dbboxes(color_np, P, current_cars, colour = (255, 0, 0), line_width = 2)

        viz.plot_image(color_np)

# --------------------------------------------------------
def plot_object_data_as_trajectories(dataset_params, pinhole_camera_intrinsic, opts):
# --------------------------------------------------------

    # create output dir (iff we wanna store the data...)
    # if not os.path.exists(opts.out_dir):
    #     print('Creating output dir: {}'.format(opts.out_dir))
    #     os.makedirs(opts.out_dir)

    # get all image data
    all_rgb_files = os.listdir(dataset_params.rgb_dir)
    n_files = len(all_rgb_files)
    print('Dataset contains {} files'.format(n_files)) 
    
    # get motgt data
    all_mot = pd.read_csv(dataset_params.mot_file, sep=" ", index_col=False)
    # assert(n_files == len(all_mot.index))

    # get camera matrix for projections
    K = pinhole_camera_intrinsic.intrinsic_matrix
    P = viz.formCameraMatrix(K, R = np.identity(3), t = np.zeros((3, 1)))

    for i in range(max(0, opts.start), min(n_files, opts.start + opts.limit_frames)):
        print('Processing image {}'.format(all_rgb_files[i]))

        # read image (in each loop)
        color = read_image(os.path.join(dataset_params.rgb_dir, all_rgb_files[i]))

        # => numpy
        color_np = np.asarray(color)

        # get objects
        current_objects = all_mot.loc[all_mot['frame'] == i]
        current_cars = current_objects.loc[current_objects['label'] == 'Car']
        current_other = current_objects.loc[current_objects['label'] != 'Car']
        # print((current_objects))
        # print((current_cars))

        if opts.show_2Dbboxes:

            for index, bbox in current_cars.iterrows(): 

                bbox2d = data_utils.extract2Dbbox(bbox)
                tracklet_id = data_utils.extract_tracklet_idx(bbox)
                
                color_np = viz.draw_2Dbbox(color_np, bbox2d, colour = list(np.asarray(IOFVisualiser().get_colour(tracklet_id)) * 255), line_width = 3)
                # color_np = viz.draw_2Dbbox(color_np, bbox2d, colour = get_colour(tracklet_id), line_width = 3)
                # color_np = viz.draw_2Dbbox(color_np, bbox2d, colour = (0, 255, 0), line_width = 3)
            
            # color_np = viz.draw_2Dbboxes(color_np, current_cars, colour = (0, 255, 0), line_width = 3)
            color_np = viz.draw_2Dbboxes(color_np, current_other, colour = (0, 0, 255), line_width = 3)

        if opts.show_3Dbboxes:

            for index, bbox in current_cars.iterrows(): 

                bbox3d = data_utils.extract3Dbbox(bbox)
                tracklet_id = data_utils.extract_tracklet_idx(bbox)

                if not bbox3d.is_valid():
                    print('Skipping 3D bbox - negative values, prob unused')
                    continue

                box3d_in_camera_coords, face_idx = viz.computeBox3D(P, bbox3d)

                #print()

                color_np = viz.draw_3Dbbox(color_np, box3d_in_camera_coords, face_idx, colour = list(np.asarray(IOFVisualiser().get_colour(tracklet_id)) * 255), line_width = 2)
                # color_np = viz.draw_3Dbbox(color_np, box3d_in_camera_coords, face_idx, colour = get_colour(tracklet_id), line_width = 2)

                # draw orientation vector?
                #if draw_orientation:
                orientation = viz.computeOrientation3D(P, bbox3d)
                if len(orientation) == 0:
                    continue

                color_np = viz.draw_3Dbox_orientation(color_np, orientation)

            # we prob do not need this anymore as bboxes are colour-coded now...
            # color_np = viz.draw_3Dbboxes(color_np, P, current_cars, colour = (255, 0, 0), line_width = 2)

        viz.plot_image(color_np)

        # use if we wanna save to some file
        # color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(opts.out_dir, all_rgb_files[i]), color_np)


# --------------------------------------------------------
if __name__ == "__main__":
# --------------------------------------------------------

    parser = argparse.ArgumentParser()

    add_bool_arg(parser, 'r', default=False, description='Set to true to reconstruct the scene')    
    parser.add_argument('--d', type=int, default=-1, help='Set to true to use fusion with dynamically moving objects'\
                                                              +' (0 = naive dynamic fusion)')
    parser.add_argument('--dataset', type=str, default='vkitti_coarse', help='Dataset (default: "vkitti_coarse")')
    parser.add_argument('--l', type=str, default=None, help='Load scene to mesh at "some dir" path (default: None)')

    parser.add_argument('--start', type=int, default=0, help='First frame in processed sequence (default: 0)')
    parser.add_argument('--limit_frames',  type=int, default=1000000000, help='Limits number of processed frames (default: 1000000)')
    
    add_bool_arg(parser, 'm', default=True, description='Set to true to use mesh, false for PCD')
    add_bool_arg(parser, 'o', default=False, description='Set to true to save on-the-fly visualisation')
    add_bool_arg(parser, 'f', default=False, description='Set to true to show fly-through')
    add_bool_arg(parser, 'c', default=False, description='Set to true to colorize instances')

    parser.add_argument('--out_dir', type=str, default='', help='Saves data to "directory/" path (default: None)')
    add_bool_arg(parser, 'save_volumes', default=False, description='Set to true to save volumes')
    parser.add_argument('--render_options', type=str, default='renderoptions.json', help='Path to dir with renderoptions.json (allows setting of light, normals, etc)')
    # parser.add_argument('--save_mesh', type=str, default=None, help='Saves volume to mesh at "some_path.ply" path (default: None)')
    # parser.add_argument('--save_pcd', type=str, default=None, help='Saves volume to pcd at "some_path.pcd" path (default: None)')

    add_bool_arg(parser, 'show_2Dbboxes', default=False, description='Set to true to show debug visualisations of bboxes')
    add_bool_arg(parser, 'show_3Dbboxes', default=False, description='Set to true to show debug visualisations of bboxes')
    add_bool_arg(parser, 'v', default=False, description='Set to true for verbose')
    
    opts = parser.parse_args()

    # --------------------------------------------------------
    # get dataset
    # --------------------------------------------------------
    dataset_params = datasets.getDatasetParams(dataset = opts.dataset)
    datasets.printDatasetParams(dataset_params)    

    # --------------------------------------------------------
    # read camera matrix
    # --------------------------------------------------------
    camera_intrinsic = read_pinhole_camera_intrinsic(dataset_params.intrinsics_file)
    if(opts.v):
        print_intrinsics(camera_intrinsic)

    # --------------------------------------------------------
    # visualisation params
    # --------------------------------------------------------
    online_vis_params = VisualisationParams(on_the_fly_visualisation = opts.o,
                                            use_mesh       = opts.m, 
                                            save_rgb       = True,
                                            save_depth     = False,
                                            save_bg_volume = False,
                                            save_fg_volumes= False,
                                            paint_objects  = opts.c,
                                            output_dir     = os.path.join(opts.out_dir, 'on_the_fly'),
                                            render_options = opts.render_options)

    flythough_vis_params = VisualisationParams(on_the_fly_visualisation = False,
                                               use_mesh       = opts.m, 
                                               save_rgb       = True,
                                               save_depth     = False,
                                               save_bg_volume = False,
                                               save_fg_volumes= False,
                                               paint_objects  = opts.c,
                                               output_dir     = os.path.join(opts.out_dir, 'flythrough'),
                                               render_options = opts.render_options)


    # --------------------------------------------------------
    # reconstruction
    # --------------------------------------------------------
    if opts.r:
        # which method do we want?
        if opts.d == 0:
            print('Running Dynamic Fusion')
            fusion = DynamicObjectsFusion(dataset_params, camera_intrinsic, 
                                          visualisation_params = online_vis_params, 
                                          invalidate_2D_bbox = True,
                                          verbose = opts.v)
        else:
            print('Running standard fusion')
            fusion = StaticFusion(dataset_params, camera_intrinsic, 
                                  visualisation_params = online_vis_params, 
                                  verbose = opts.v)

        # reconstruct dataset
        start = time.time()
        reconstructed_scene = fusion.reconstruct_scene(opts.start, opts.limit_frames)
        end = time.time()
        print('Reconstruction took: {} sec.)'.format(end - start))

        if len(reconstructed_scene.skipped_objects) > 0:
            print('[WARNING] These objects were not reconstructed: {}'.format(reconstructed_scene.skipped_objects) 
                  + ' (depth of all pts > depth_trunc)')

        # save volume to mesh or pcd?
        if opts.save_volumes:
            reconstructed_scene.save(opts.out_dir, use_mesh = opts.m)   

        # show the whole scene? (ie not on-the-fly) 
        if opts.f: 
            # IOFVisualiser().visualise(reconstructed_scene, camera_intrinsic, flythough_vis_params)

            # hardcoded for time-being (for some reason, open3d visualiser does not support kitti intrinsics)
            # dataset_base_dir  = os.path.join('data', 'kitti_full', '0001')
            intrinsics_file_viz = os.path.join(dataset_params.base_dir, 'calib', 'naive_viz.json') 
            camera_intrinsic_viz = read_pinhole_camera_intrinsic(intrinsics_file_viz)

            IOFVisualiser().visualise(reconstructed_scene, camera_intrinsic_viz, flythough_vis_params)

    # --------------------------------------------------------
    # debug
    # --------------------------------------------------------
    '''
    # may not work now - double check before using
    if opts.l is not None:
        loaded_scene = IndependentObjectsFusion.load(opts.l)         
        IOFVisualiser().visualise(loaded_scene, camera_intrinsic, use_mesh = True)
    '''

    if opts.show_2Dbboxes or opts.show_3Dbboxes:
        # plot_object_data(dataset_params, camera_intrinsic, opts)
        plot_object_data_as_trajectories(dataset_params, camera_intrinsic, opts)




