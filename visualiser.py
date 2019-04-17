from open3d import *
import numpy as np
from geometry import *
import os
from collections import namedtuple


VisualisationParams = namedtuple('VisualisationParams',
                                 'on_the_fly_visualisation ' 
                                 'use_mesh '
                                 'save_rgb '
                                 'save_depth '
                                 'save_bg_volume '
                                 'save_fg_volumes '
                                 'paint_objects ' 
                                 'output_dir ' 
                                 'render_options')

class IOFVisualiser(object):

    def __init__(self):
        pass

    @staticmethod
    def __extract_mesh(volume, transform = None, uniform_color = None):

        print("Extracting a triangle mesh from the volume.")
        mesh = volume.extract_triangle_mesh()
        mesh.purge()
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        if uniform_color is not None:
            mesh.paint_uniform_color(uniform_color)
        
        if transform is not None:
            mesh.transform(transform)
        
        return mesh

    @staticmethod
    def __extract_pointcloud(volume, transform = None, downsample_voxel_size = None,
                             uniform_color = None):

        print("Extracting a point cloud from the volume.")
        #pcd = volume.extract_voxel_point_cloud()
        pcd = volume.extract_point_cloud()

        # optional downsample
        if downsample_voxel_size is not None:
            pcd = voxel_down_sample(pcd, voxel_size = downsample_voxel_size)

        if uniform_color is not None:
            pcd.paint_uniform_color(uniform_color)

        # optional transform    
        if transform is not None:
            pcd.transform(transform)
        
        return pcd

    @staticmethod
    def get_colour(value):
        colors = [[ 0.90196078,  0.09803922,  0.29411765],
                  [ 0.23529412,  0.70588235,  0.29411765],
                  [ 1.        ,  0.88235294,  0.09803922],
                  [ 0.2627451 ,  0.38823529,  0.84705882],
                  [ 0.96078431,  0.50980392,  0.19215686],
                  [ 0.56862745,  0.11764706,  0.70588235],
                  [ 0.2745098 ,  0.94117647,  0.94117647],
                  [ 0.94117647,  0.19607843,  0.90196078],
                  [ 0.7372549 ,  0.96470588,  0.04705882],
                  [ 0.98039216,  0.74509804,  0.74509804],
                  [ 0.        ,  0.50196078,  0.50196078],
                  [ 0.90196078,  0.74509804,  1.        ],
                  [ 0.60392157,  0.38823529,  0.14117647],
                  [ 1.        ,  0.98039216,  0.78431373],
                  [ 0.50196078,  0.        ,  0.        ],
                  [ 0.66666667,  1.        ,  0.76470588],
                  [ 0.50196078,  0.50196078,  0.        ],
                  [ 1.        ,  0.84705882,  0.69411765],
                  [ 0.        ,  0.        ,  0.45882353],
                  [ 0.50196078,  0.50196078,  0.50196078]]

        return colors[value % (len(colors))]


    @staticmethod
    def visualise(fusion, O3Dintrinsics, params, caption = ''): 
        
        preload_all_objects = True # do NOT change this - not implemented yet

        # place where we move objects that should be removed from the map
        garbage_pose = np.eye(4)
        garbage_pose[0:3, 3] = [-1000, -1000, -1000] 

        render_options = None
        if params is not None:
            use_mesh = params.use_mesh
            save_rgb = params.save_rgb
            save_depth = params.save_depth
            paint_objects = params.paint_objects
            output_dir = params.output_dir
            render_options = params.render_options


        extract_geometry = __class__.__extract_mesh if use_mesh else __class__.__extract_pointcloud

        if (save_rgb or save_depth) and output_dir is not '' and not os.path.exists(output_dir):
            print('Creating output dir: {}'.format(output_dir))
            os.makedirs(output_dir)

        print("Plotting " + caption)

        # get map
        dynamic_map = fusion.get_map_state
        frames = sorted(dynamic_map)
        if(len(dynamic_map) < 1):
            # TODO print something
            return

        # load background
        bg_volume = fusion.get_stationary_volume().volume

        if fusion.is_read_only is not True:
            bg_volume = extract_geometry(bg_volume)

        # list of objects
        dynamic_objects_pcd = {}    # tracklet_id -> pcd, last_pose_in_world_frame
        dynamic_objects_poses = {}
        dynamic_objects_in_map = set()

        if(preload_all_objects):
            all_objects = fusion.get_all_objects()

            # preload pointclouds
            for object_idx, obj in all_objects.items():
                pcd_object = obj.volume
                if fusion.is_read_only is not True:
                    if paint_objects is not True:
                        pcd_object = extract_geometry(pcd_object)
                    else: 
                        pcd_object = extract_geometry(pcd_object, uniform_color = __class__.get_colour(object_idx)) 

                dynamic_objects_pcd[object_idx] = pcd_object
                dynamic_objects_poses[object_idx] = np.eye(4)

        # visualisation
        __class__.visualise.frame_index = frames[0]
        __class__.visualise.vis = Visualizer()

        glb = __class__.visualise

        def move_forward(vis):

            glb = __class__.visualise

            def move_object(dynamic_objects_pcd, dynamic_objects_poses, camera_pose, object_pose, object_idx):
                # move object
                object_pose_in_world_t_1 = dynamic_objects_poses[object_idx]
                object_pose_in_world_t = np.matmul(np.linalg.inv(camera_pose), object_pose)
                object_pose_in_world = np.matmul(object_pose_in_world_t, np.linalg.inv(object_pose_in_world_t_1))

                # TODO: we should deal only with objects that are within few frustum
                # TODO: deciding whether object's moving or not should take also rotation into account
                t_object_pose_in_world_t_1 = object_pose_in_world_t_1[0:3, 3]
                t_object_pose_in_world_t = object_pose_in_world_t[0:3, 3]
                distance = np.linalg.norm(t_object_pose_in_world_t_1 - t_object_pose_in_world_t)
                if distance < 1e-4:
                    print('id: {}, distance: {}'.format(object_idx, distance))
                    print('object idx: {} is not moving => skip\n'.format(object_idx))
                    # continue
                    return False

                dynamic_objects_pcd[object_idx].transform(object_pose_in_world)
                dynamic_objects_poses[object_idx] = object_pose_in_world_t # object_pose_in_world

                # render_all = True
                return True

            if glb.frame_index < (frames[0] + len(frames)):

                print('Rendering frame: {}'.format(glb.frame_index))

                # retrieve camera in this frame
                camera_pose = fusion.get_stationary_pose(glb.frame_index) 

                # get objects for this frame
                frame_objects = dynamic_map[glb.frame_index]

                # update objects
                render_all = False
                for object_idx in frame_objects:

                    # add missing objects (add_geometry)
                    if object_idx not in dynamic_objects_in_map:
                        # print('Adding object id: {}'.format(object_idx))
                        vis.add_geometry(dynamic_objects_pcd[object_idx])
                        dynamic_objects_in_map.add(object_idx)

                    # get camera poses for objects
                    object_pose = fusion.get_pose_of_object(object_idx, glb.frame_index)

                    r = move_object(dynamic_objects_pcd, dynamic_objects_poses, camera_pose, object_pose, object_idx)
                    render_all = render_all or r

                # let's move objects for which tracklet has ended to some garbage location
                # ideally, this should do geometry.delete or recreate the map here
                for object_idx in dynamic_objects_in_map:
                    last_frame = fusion.get_object(object_idx).get_id_of_last_frame()

                    if glb.frame_index > last_frame:
                        r = move_object(dynamic_objects_pcd, dynamic_objects_poses, camera_pose, garbage_pose, object_idx)
                        render_all = render_all or r
                        #print('Object idx: {} should be removed because last frame idx: {}'.format(object_idx, last_frame))


                # place camera
                traj = PinholeCameraTrajectory()
                traj.intrinsic = O3Dintrinsics
                traj.extrinsic = Matrix4dVector([camera_pose])

                ctr = vis.get_view_control()
                ctr.convert_from_pinhole_camera_parameters(traj.intrinsic, traj.extrinsic[0])

                # TODO: ideally, this should update only moving geometries...
                if render_all:
                    __class__.visualise.vis.update_geometry()
                    __class__.visualise.vis.poll_events()
                    __class__.visualise.vis.update_renderer()

                # optionally save rgb or depth
                if save_rgb:
                    __class__.visualise.vis.capture_screen_image(os.path.join(output_dir, "reconstruction_rgb_%06d.png" % glb.frame_index))

                if save_depth:
                    __class__.visualise.vis.capture_depth_image(os.path.join(output_dir, "reconstruction_depth_%06d.png" % glb.frame_index))

                glb.frame_index = glb.frame_index + 1
            else:
                __class__.visualise.vis.register_animation_callback(None)


        vis = __class__.visualise.vis 
        vis.create_window(window_name = caption, 
                          width = O3Dintrinsics.width, 
                          height = O3Dintrinsics.height)
        vis.add_geometry(bg_volume)

        if render_options is not None:
            vis.get_render_option().load_from_json(render_options)

        traj = PinholeCameraTrajectory()
        traj.intrinsic = O3Dintrinsics
        # print(frames[0])
        # print([fusion.get_stationary_pose(frames[0])])
        traj.extrinsic = Matrix4dVector([fusion.get_stationary_pose(frames[0])])
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(traj.intrinsic, traj.extrinsic[0])

        # vis.get_render_option().load_from_json("../../TestData/renderoption.json")
        vis.register_animation_callback(move_forward)
        vis.run()
        vis.destroy_window()


    @staticmethod
    def visualise_frame(fusion, O3Dintrinsics, frame_idx, params, caption = ''):
        
        render_options = None
        if params is not None:
            use_mesh = params.use_mesh
            save_rgb = params.save_rgb
            save_depth = params.save_depth
            save_bg_volume = params.save_bg_volume
            save_fg_volumes = params.save_fg_volumes
            paint_objects = params.paint_objects
            output_dir = params.output_dir
            render_options = params.render_options
            # TODO add blocking / non-blocking

        # well, there's prob no point of going through plotting if it is in non-blocking mode but...
        # if (not save_rgb) and (not save_depth) and (not save_bg_volume) and (not save_fg_volumes):
        #     return

        extract_geometry = __class__.__extract_mesh if use_mesh else __class__.__extract_pointcloud

        print("Plotting " + caption)

        # get map
        dynamic_map = fusion.get_map_state
        if(len(dynamic_map) < 1):
            # TODO print something
            return

        # load background
        bg_volume = fusion.get_stationary_volume().volume
        bg_volume = extract_geometry(bg_volume)

        # list of objects
        dynamic_objects_pcd = {}    # tracklet_id -> pcd, last_pose_in_world_frame
        dynamic_objects_poses = {}
        dynamic_objects_in_map = set()

        # all_objects = fusion.get_all_objects()
        all_objects = fusion.get_all_objects_in_frame(frame_idx)

        # preload pointclouds
        for object_idx, obj in all_objects.items():
            pcd_object = obj.volume
            if paint_objects is not True:
                pcd_object = extract_geometry(pcd_object)
            else: 
                pcd_object = extract_geometry(pcd_object, uniform_color = __class__.get_colour(object_idx)) 


            dynamic_objects_pcd[object_idx] = pcd_object
            dynamic_objects_poses[object_idx] = np.eye(4)


        vis = Visualizer()
        vis.create_window(window_name = caption, 
                          width = O3Dintrinsics.width, 
                          height = O3Dintrinsics.height)

        # optionally load visualiser options

        # add background
        vis.add_geometry(bg_volume)
        # vis.add_geometry(dynamic_objects_pcd[0])
        # dynamic_objects_in_map.add(0)

        # retrieve camera in this frame
        camera_pose = fusion.get_stationary_pose(frame_idx) 

        # get objects for this frame
        frame_objects = dynamic_map[frame_idx]

        # update objects
        for object_idx in frame_objects:

            # add missing objects (add_geometry)
            if object_idx not in dynamic_objects_in_map:
                vis.add_geometry(dynamic_objects_pcd[object_idx])
                dynamic_objects_in_map.add(object_idx)

            # get camera poses for objects
            object_pose = fusion.get_pose_of_object(object_idx, frame_idx)

            # move object
            object_pose_in_world_t_1 = dynamic_objects_poses[object_idx]
            object_pose_in_world_t = np.matmul(np.linalg.inv(camera_pose), object_pose)
            object_pose_in_world = np.matmul(object_pose_in_world_t, np.linalg.inv(object_pose_in_world_t_1))

            # TODO: we should deal only with objects that are within few frustum
            # TODO: deciding whether object's moving or not should take also rotation into account
            t_object_pose_in_world_t_1 = object_pose_in_world_t_1[0:3, 3]
            t_object_pose_in_world_t = object_pose_in_world_t[0:3, 3]
            distance = np.linalg.norm(t_object_pose_in_world_t_1 - t_object_pose_in_world_t)
            if distance < 1e-4:
                print('id: {}, distance: {}'.format(object_idx, distance))
                print('object idx: {} is not moving => skip\n'.format(object_idx))
                continue

            dynamic_objects_pcd[object_idx].transform(object_pose_in_world)
            dynamic_objects_poses[object_idx] = object_pose_in_world_t # object_pose_in_world

        # place camera
        traj = PinholeCameraTrajectory()
        traj.intrinsic = O3Dintrinsics
        traj.extrinsic = Matrix4dVector([camera_pose])

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(traj.intrinsic, traj.extrinsic[0])

        if render_options is not None:
            vis.get_render_option().load_from_json(render_options)

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        # vis.run()

        # optionally save rgb or depth
        if save_rgb:
            if output_dir is not '' and not os.path.exists(output_dir):
                print('Creating output dir: {}'.format(output_dir))
                os.makedirs(output_dir)

            vis.capture_screen_image(os.path.join(output_dir, "reconstruction_rgb_%06d.png" % frame_idx))

        if save_depth:
            if output_dir is not '' and not os.path.exists(output_dir):
                print('Creating output dir: {}'.format(output_dir))
                os.makedirs(output_dir)

            vis.capture_depth_image(os.path.join(output_dir, "reconstruction_depth_%06d.png" % frame_idx))


        # TODO: should be moved to dynamic map, with option to not to extract the volumes again...
        if save_fg_volumes:
            if output_dir is not '' and not os.path.exists(output_dir):
                print('Creating output dir: {}'.format(output_dir))
                os.makedirs(output_dir)

            # here we wanna iterate over all objects and dump them to file
            for object_idx in frame_objects:
                volume = dynamic_objects_pcd[object_idx]

                basename = 'volume_{:04d}_at_{:06d}'.format(object_idx, frame_idx)
                if use_mesh:
                    filename = basename + '.ply'
                    path = os.path.join(output_dir, filename)
                    write_triangle_mesh(path, volume)

                else:
                    filename = basename + '.pcd'
                    path = os.path.join(output_dir, filename)
                    write_point_cloud(path, volume)

        if save_bg_volume:
            if output_dir is not '' and not os.path.exists(output_dir):
                print('Creating output dir: {}'.format(output_dir))
                os.makedirs(output_dir)

            # deal with bg volume
            basename = 'background_at_{:06d}'.format(frame_idx)
            if use_mesh:
                filename = basename + '.ply'
                path = os.path.join(output_dir, filename)
                write_triangle_mesh(path, bg_volume)

            else:
                filename = basename + '.pcd'
                path = os.path.join(output_dir, filename)
                write_point_cloud(path, bg_volume)


        vis.destroy_window()


