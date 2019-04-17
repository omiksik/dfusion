from open3d import *
from collections import namedtuple
import os

# --------------------------------------------------------
def getDatasetParams(dataset = 'vkitti', force_base_dir = None):
# --------------------------------------------------------
    DatasetParams = namedtuple('DatasetParams', 'dataset '
                                                'base_dir '
                                                'rgb_dir '
                                                'depth_dir '
                                                'extrinsics_file '
                                                'intrinsics_file '
                                                'invert_extrinsics '
                                                'mot_file '
                                                'depth_trunc '
                                                'depth_scale '
                                                'bg_voxel_length '
                                                'bg_sdf_trunc '
                                                'bg_color_type '
                                                'fg_voxel_length '
                                                'fg_sdf_trunc '
                                                'fg_color_type')

  
    if dataset == 'vkitti_coarse':

        dataset_base_dir  = os.path.join('data', 'vkitti_dataset')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'vkitti_coarse',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'rgb'),
                                depth_dir           = os.path.join(dataset_base_dir, 'depth'), 
                                extrinsics_file     = os.path.join(dataset_base_dir, 'extrinsics', '0001_clone.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'vkitti_camera.json'),
                                invert_extrinsics   = False,
                                mot_file            = os.path.join(dataset_base_dir, 'mot', '0001_clone.txt'),
                                depth_trunc         = 45,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 32.0 / 512.0, # 32.0m / 512.0 = 62.5 mm  
                                bg_sdf_trunc        = 0.9, 
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 16.0 / 512.0,    
                                fg_sdf_trunc        = 0.2,  
                                fg_color_type       = TSDFVolumeColorType.RGB8)

    elif dataset == 'vkitti':

        dataset_base_dir  = os.path.join('data', 'vkitti_dataset')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'vkitti',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'rgb'),
                                depth_dir           = os.path.join(dataset_base_dir, 'depth'), 
                                extrinsics_file     = os.path.join(dataset_base_dir, 'extrinsics', '0001_clone.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'vkitti_camera.json'),
                                invert_extrinsics   = False,
                                mot_file            = os.path.join(dataset_base_dir, 'mot', '0001_clone.txt'),
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                bg_sdf_trunc        = 0.04, 
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.2, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)

    elif dataset == 'kitti_0003':

        dataset_base_dir  = os.path.join('data', 'kitti_dataset', '0003')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_0003',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'input_images'),
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked'),  
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0003.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', '0003.json'),  
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0003_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 24.0 / 512.0,   
                                bg_sdf_trunc        = 0.4,  
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)

    elif dataset == 'kitti_full_0001':

        dataset_base_dir  = os.path.join('data', 'kitti_full', '0001')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_full_0001',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'image_2'), 
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked'),  
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0001.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', 'naive.json'),  
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0001_2d_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 24.0 / 512.0,   
                                bg_sdf_trunc        = 0.4,  
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)

    elif dataset == 'kitti_full_0002':

        dataset_base_dir  = os.path.join('data', 'kitti_full', '0002')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_full_0002',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'image_2'), 
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked'),  
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0002.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', 'naive.json'),  
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0002_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 24.0 / 512.0,   
                                bg_sdf_trunc        = 0.4,  
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)


    elif dataset == 'kitti_full_0003':

        dataset_base_dir  = os.path.join('data', 'kitti_full', '0003')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_full_0003',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'image_2'), 
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked'),  
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0003.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', 'naive.json'),  
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0003_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 24.0 / 512.0,   
                                bg_sdf_trunc        = 0.4,  
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)

    elif dataset == 'kitti_full_0008':

        dataset_base_dir  = os.path.join('data', 'kitti_full', '0008')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_full_0008',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'image_2'), 
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked'),  
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0008.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', 'naive.json'),  
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0008_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 24.0 / 512.0,  
                                bg_sdf_trunc        = 0.4, 
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)

    ##############################################################################
    # C O A R S E
    ##############################################################################

    elif dataset == 'kitti_0003_coarse':

        dataset_base_dir  = os.path.join('data', 'kitti_dataset', 'tracking', '0003')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_0003_coarse',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'input_images'),
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked'), 
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0003.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', '0003.json'), 
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0003_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 96.0 / 512.0, 
                                bg_sdf_trunc        = 0.4,  
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)

    elif dataset == 'kitti_full_0001_coarse':

        dataset_base_dir  = os.path.join('data', 'kitti_full', '0001')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_full_0001_coarse',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'image_2'), 
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked'), 
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0001.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', 'naive.json'), 
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0001_2d_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 96.0 / 512.0,  
                                bg_sdf_trunc        = 0.4,  
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)

    elif dataset == 'kitti_full_0002_coarse':

        dataset_base_dir  = os.path.join('data', 'kitti_full', '0002')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_full_0002_coarse',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'image_2'), 
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked'),  
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0002.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', 'naive.json'), 
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0002_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 96.0 / 512.0, 
                                bg_sdf_trunc        = 0.4,  
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)


    elif dataset == 'kitti_full_0003_coarse':

        dataset_base_dir  = os.path.join('data', 'kitti_full', '0003')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_full_0003_coarse',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'image_2'), 
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked'),  
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0003.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', 'naive.json'), 
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0003_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 96.0 / 512.0,  
                                bg_sdf_trunc        = 0.4,  
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)

    elif dataset == 'kitti_full_0008_coarse':

        dataset_base_dir  = os.path.join('data', 'kitti_full', '0008')
        if force_base_dir is not None:
            dataset_base_dir = force_base_dir

        params  = DatasetParams(dataset             = 'kitti_full_0008',
                                base_dir            = dataset_base_dir,
                                rgb_dir             = os.path.join(dataset_base_dir, 'image_2'), 
                                depth_dir           = os.path.join(dataset_base_dir, 'psm_depth_57_masked_erode'),  
                                extrinsics_file     = os.path.join(dataset_base_dir, 'camera_pose', 'libviso_odometry_0008.txt'),
                                intrinsics_file     = os.path.join(dataset_base_dir, 'calib', 'naive.json'),  
                                invert_extrinsics   = True,
                                mot_file            = os.path.join(dataset_base_dir, 'tracking_mdp', '0008_2d_3d_tracker.txt'), 
                                depth_trunc         = 40,  
                                depth_scale         = 100, # do NOT touch this !!! 
                                bg_voxel_length     = 96.0 / 512.0, 
                                bg_sdf_trunc        = 0.4, 
                                bg_color_type       = TSDFVolumeColorType.RGB8,
                                fg_voxel_length     = 8.0 / 512.0, # single voxel size: 8.0m / 512.0 = 15.625 mm  
                                fg_sdf_trunc        = 0.4, 
                                fg_color_type       = TSDFVolumeColorType.RGB8)


    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    return params

# --------------------------------------------------------
def printDatasetParams(params):
# --------------------------------------------------------

    print('Using dataset: {}'.format(params.dataset))
    print('\t base_dir: {}'.format(params.base_dir))
    print('\t rgb_dir: {}'.format(params.rgb_dir))
    print('\t depth_dir: {}'.format(params.depth_dir))
    print('\t extrinsics_file: {}'.format(params.extrinsics_file))
    print('\t intrinsics_file: {}'.format(params.intrinsics_file))
    print('\t invert_extrinsics: {}'.format(params.invert_extrinsics))
    print('\t mot_file: {}'.format(params.mot_file))
    print('\t depth_trunc: {}'.format(params.depth_trunc))
    print('\t depth_scale: {}'.format(params.depth_scale))
    print('\t bg_voxel_length: {}'.format(params.bg_voxel_length))
    print('\t bg_sdf_trunc: {}'.format(params.bg_sdf_trunc))
    print('\t bg_color_type: {}'.format(params.bg_color_type))
    print('\t fg_voxel_length: {}'.format(params.fg_voxel_length))
    print('\t fg_sdf_trunc: {}'.format(params.fg_sdf_trunc))
    print('\t fg_color_type: {}\n'.format(params.fg_color_type))




