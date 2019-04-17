from open3d import *
import numpy as np
from geometry import *

import time
import json
import os

# might be worth implementing independing of bboxes, ie we could use also semantics, etc as instatiations

class Open3DUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def extract_mesh(volume, transform = None):

        print("Extracting a triangle mesh from the volume.")
        mesh = volume.extract_triangle_mesh()
        mesh.purge()
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        
        if transform is not None:
            mesh.transform(transform)
        
        return mesh

    @staticmethod
    def save_volume2mesh(path, volume, transform = None):

        mesh = Open3DUtils.extract_mesh(volume, transform)
        write_triangle_mesh(path, mesh)


    @staticmethod
    def extract_pcd(volume, transform = None, downsample_voxel_size = None):

        print("Extracting a point cloud from the volume.")
        pcd = volume.extract_voxel_point_cloud()
        
        # optional downsample
        if downsample_voxel_size is not None:
            pcd = voxel_down_sample(pcd, voxel_size = downsample_voxel_size)

        # optional transform    
        if transform is not None:
            pcd.transform(transform)
        
        return pcd

    @staticmethod
    def save_volume2pcd(path, volume, transform = None, downsample_voxel_size = None):

        pcd = Open3DUtils.extract_pcd(volume, transform, downsample_voxel_size)
        write_point_cloud(path, pcd)


    @staticmethod
    def TSDF_colortype2str(color_type):
        
        if color_type == TSDFVolumeColorType.RGB8:
            return 'RGB8'
        elif color_type == TSDFVolumeColorType.Gray32:
            return 'Gray32'
        else:
            return 'Unknown' 

    def str2TSDF_colortype(color_type_str):
        
        if color_type_str == 'RGB8':
            return TSDFVolumeColorType.RGB8
        elif color_type_str == 'Gray32':
            return TSDFVolumeColorType.Gray32
        else:
            return 'Unknown' 


class DynamicFusionEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj,'toJson'):
            return obj.toJson()
        elif isinstance(obj, set):
            return list(obj) 
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


class IndependentVolume(object):

    def __init__(self, voxel_length, sdf_trunc, color_type):

        self.voxel_length = voxel_length
        self.sdf_trunc    = sdf_trunc
        self.color_type   = color_type

        self.volume = ScalableTSDFVolume(self.voxel_length, self.sdf_trunc, self.color_type)
        self.poses = {} 

        # self.last_object_pose = np.eye(4)

    def integrate(self, rgbd, O3Dintrinsics, Rt, frame_idx):
        self.volume.integrate(rgbd, O3Dintrinsics, Rt)
        self.poses[frame_idx] = Rt
        # TODO: we should save intrinsics for (multi-camera) replay as well

    def update_pose_without_integration(self, Rt, frame_idx):
        self.poses[frame_idx] = Rt
        # TODO: we should save intrinsics for (multi-camera) replay as well


    def get_volume(self):
        return self.volume

    def get_pose_at_frame(self, frame_idx): # we might rather wanna transform volume to this pose

        if frame_idx not in self.poses:
            return [], False
        else:
            return self.poses[frame_idx], True

    def get_id_of_last_frame(self):

        if len(self.poses) == 0:
            return -1
        else:
            return max(self.poses.keys())

    def toJson(self):

        return dict(voxel_length=self.voxel_length, 
            sdf_trunc=self.sdf_trunc, 
            color_type=Open3DUtils.TSDF_colortype2str(self.color_type),
            poses=self.poses)

    def save_volume(self, dir, basename = 'volume', use_mesh = True):

        # save mesh
        if use_mesh:
            filename = basename + '.ply'
            path = os.path.join(dir, filename)
            Open3DUtils().save_volume2mesh(path, self.get_volume())
        else:
            filename = basename + '.pcd'
            path = os.path.join(dir, filename)
            Open3DUtils().save_volume2pcd(path, self.get_volume())

        # save poses
        filename = basename + '.json'
        path = os.path.join(dir, filename)
        with open(path, 'w') as out_file:
            json.dump(self, out_file, cls=DynamicFusionEncoder, indent=4, separators=(',', ': '))

            # print(self.poses[35])
            
    def load_volume(self, dir, basename, use_mesh = True):

        if use_mesh:
            filename = basename + '.ply'
            path = os.path.join(dir, filename)
            self.volume = read_triangle_mesh(path)
        else:
            raise('Load volume supports mesh only now')
            filename = basename + '.pcd'
            path = os.path.join(dir, filename)
            self.volume = load_pointcloud(path)

        filename = basename + '.json'
        path = os.path.join(dir, filename)
        # print('\nLoading map state from {}'.format(filename))

        with open(path, 'r') as in_file:
            s = json.load(in_file)
            self.poses        = s['poses']
            self.voxel_length = s['voxel_length']
            self.sdf_trunc    = s['sdf_trunc']
            self.color_type   = Open3DUtils.str2TSDF_colortype(s['color_type'])

            self.poses        = dict((int(k), np.array(v)) for k,v in self.poses.items())


class DynamicMap(object):

    def __init__(self, bg_voxel_length, bg_sdf_trunc, bg_color_type,
                       fg_voxel_length = None, fg_sdf_trunc = None, fg_color_type = None):
    
        self.bg_voxel_length = bg_voxel_length
        self.bg_sdf_trunc    = bg_sdf_trunc
        self.bg_color_type   = bg_color_type

        self.fg_voxel_length = fg_voxel_length if fg_voxel_length is not None else bg_voxel_length
        self.fg_sdf_trunc    = fg_sdf_trunc    if fg_sdf_trunc    is not None else bg_sdf_trunc
        self.fg_color_type   = fg_color_type   if fg_color_type   is not None else bg_color_type

        self.background = IndependentVolume(self.bg_voxel_length, self.bg_sdf_trunc, self.bg_color_type)
        self.objects = {}
        self.unreconstructed_objects = set() 
        self.map_state = {} # frame_idx, list of objects_ids present objects within frame

        self.read_only = False

        # this should be done => TODO: prob we should store first and last frame ID to improve UX
        # TODO: allow extraction of first and last frame

        # self.save_incrementally = False


    def __register_frame(self, frame_idx):
        if frame_idx not in self.map_state: # if haven't fused any frame before, this object is un-reconstructed
            self.map_state[frame_idx] = set()

    def __add_object_to_frame(self, object_idx, frame_idx):
        self.__register_frame(frame_idx)            
        self.map_state[frame_idx].add(object_idx)

    def object_exists(self, object_idx):
        return (object_idx in self.objects)

    def integrate_object(self, object_rgbd, O3Dintrinsics, object_pose, object_idx, frame_idx):

        # TODO: might be better to catch this sooner and assume here all's good
        # do we have any valid depth point? if not, no point of introducing a new volume / fusing
        if np.all(np.asarray(object_rgbd.depth) == 0):
            # if object_idx not in self.objects: # if haven't fused any frame before, this object is un-reconstructed
            if not self.object_exists(object_idx): # if haven't fused any frame before, this object is un-reconstructed
                self.unreconstructed_objects.add(object_idx)
            return 

        # do we have a volume for this tracklet?
        # if object_idx not in self.objects:
        if not self.object_exists(object_idx):
            self.objects[object_idx] = IndependentVolume(self.fg_voxel_length, self.fg_sdf_trunc, self.fg_color_type) # can be different for each object

        # integrate
        self.objects[object_idx].integrate(object_rgbd, O3Dintrinsics, object_pose, frame_idx)

        # make sure we won't report this object as un-reconstructed
        if object_idx in self.unreconstructed_objects:
            self.unreconstructed_objects.remove(object_idx)

        # we should prob update the map state here...
        self.__add_object_to_frame(object_idx, frame_idx)

    def update_pose_without_integration(self, object_pose, object_idx, frame_idx):
         # do we have a volume for this tracklet?
        if object_idx not in self.objects:
            return False

        # integrate
        self.objects[object_idx].update_pose_without_integration(object_pose, frame_idx)

        # we should prob update the map state here...
        self.__add_object_to_frame(object_idx, frame_idx)

        return True


    def integrate_background(self, rgbd, O3Dintrinsics, camera_pose, frame_idx):
        
        self.background.integrate(rgbd, O3Dintrinsics, camera_pose, frame_idx)
        self.__register_frame(frame_idx)

    @property
    def is_read_only(self):
        return self.read_only

    # map
    @property
    def get_map_state(self):
        return self.map_state

    # background
    def get_stationary_volume(self):
        return self.background

    # def get_stationary_volume(self):
    #     return self.background.volume

    def get_stationary_pose(self, frame_idx):
        return self.background.get_pose_at_frame(frame_idx)[0]

    # objects
    def get_all_objects(self):
        return self.objects

    def get_all_objects_in_frame(self, frame_idx):

        if frame_idx not in self.map_state: 
            return {}

        objects_in_frame = {}
        for object_idx in self.map_state[frame_idx]:
            objects_in_frame[object_idx] = self.objects[object_idx]

        return objects_in_frame

    def get_pose_of_object(self, object_idx, frame_idx):
        # TODO: check object_idx exists
        return self.objects[object_idx].get_pose_at_frame(frame_idx)[0]

    def get_object(self, object_idx):
        return self.objects[object_idx] if object_idx in self.objects else None

    @property
    def skipped_objects(self):
        return self.unreconstructed_objects

    # TODO
    def object_idxs_in_frame(self, frame_idx):

        if frame_idx not in self.map_state: 
            return []

        return self.map_state[frame_idx]

    def retrieve_map_at_frame(self, frame_idx):

        # TODO: here we assume frame_idx exists

        bg_pose   = self.background.get_pose_at_frame(frame_idx)[0]
        bg_volume = self.background.volume
        
        if frame_idx not in self.map_state: 
            return (bg_volume, bg_pose), []

        objects_in_frame = {}
        for object_idx in self.map_state[frame_idx]:

            pose   = self.objects[object_idx].get_pose_at_frame(frame_idx)[0]
            volume = self.objects[object_idx].volume
            
            objects_in_frame[object_idx] = (volume, pose)

        return (bg_volume, bg_pose), objects_in_frame



    def toJson(self):

        return dict(bg_voxel_length=self.bg_voxel_length, 
            bg_sdf_trunc=self.bg_sdf_trunc, 
            bg_color_type=Open3DUtils.TSDF_colortype2str(self.bg_color_type),
            fg_voxel_length=self.fg_voxel_length,
            fg_sdf_trunc=self.fg_sdf_trunc,
            fg_color_type=Open3DUtils.TSDF_colortype2str(self.fg_color_type),
            unreconstructed_objects=self.unreconstructed_objects,
            map_state=self.map_state)


    def save(self, dir, use_mesh = True):

        if not os.path.exists(dir):
            print('Creating output dir: {}'.format(dir))
            os.makedirs(dir)

        print('\nSaving reconstruction to {}'.format(dir))

        # save all objects
        for object_idx, obj in self.objects.items():
            basename = 'volume_{:04d}'.format(object_idx)
            print('\nSaving independent object {} as {}'.format(object_idx, basename))
            
            obj.save_volume(dir, basename, use_mesh)

        # save background
        basename = 'background'
        print('\nSaving static part as {}'.format(basename))
        self.background.save_volume(dir, basename, use_mesh)

        # save map state, and other stuff
        filename = 'dynamic_map.json'
        path = os.path.join(dir, filename)
        print('\nSaving map state as {}'.format(filename))
        with open(path, 'w') as out_file:
            json.dump(self, out_file, cls=DynamicFusionEncoder, indent=4, separators=(',', ': '))


    def restore_from_json(self, dir, json_obj):

        use_mesh = True

        # load bg volume
        print('Loading bg volume from background')
        self.background = IndependentVolume(self.bg_voxel_length, self.bg_sdf_trunc, self.bg_color_type) 
        self.background.load_volume(dir, 'background')

        # set to read-only
        self.read_only = True

        # restore map-state
        self.unreconstructed_objects = json_obj['unreconstructed_objects']
        self.map_state = json_obj['map_state']

        self.map_state        = dict((int(k), np.array(v)) for k,v in self.map_state.items())


        # load all objects
        loaded = set()
        for frame_idx, object_idxs in self.map_state.items():
            for object_idx in object_idxs:

                if object_idx not in self.objects:
                    basename = 'volume_{:04d}'.format(int(object_idx))
                    print('Loading object {} from {}'.format(int(object_idx), basename))

                    self.objects[object_idx] = IndependentVolume(self.fg_voxel_length, self.fg_sdf_trunc, self.fg_color_type) 
                    self.objects[object_idx].load_volume(dir, basename)


    @staticmethod
    def load(dir):

        filename = 'dynamic_map.json'
        path = os.path.join(dir, filename)
        print('\nLoading map state from {}'.format(filename))

        with open(path, 'r') as in_file:
            s = json.load(in_file)

            new_obj = IndependentObjectsFusion(s['bg_voxel_length'], s['bg_sdf_trunc'], Open3DUtils.str2TSDF_colortype(s['bg_color_type']),
                s['fg_voxel_length'], s['fg_sdf_trunc'], Open3DUtils.str2TSDF_colortype(s['fg_color_type']))
            new_obj.restore_from_json(dir, s)
            return new_obj

        return []

'''
class IndependentObjectsFusionFromBBoxes3D(object):

    def __init__(self):
        pass
'''




