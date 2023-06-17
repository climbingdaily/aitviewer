'''
Filename: quick_view.py
Created Date: Sunday, April 2nd 2023, 10:58:55 pm
Author: climbingdaily

Copyright (c) 2023 Yudi Dai
'''

import os
import pickle as pkl
import argparse
import numpy as np
import open3d as o3d

from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer

def fix_points_num(points: np.array, num_points: int=512):

    points = points[~np.isnan(points).any(axis=-1)]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc = pc.voxel_down_sample(voxel_size=0.05)
    ratio = int(len(pc.points) / num_points + 0.05)
    if ratio > 1:
        pc = pc.uniform_down_sample(ratio)

    points = np.asarray(pc.points)
    origin_num_points = points.shape[0]

    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    else:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res

def load_pkl(filename):
    with open(filename, 'rb') as f:
        dets = pkl.load(f)
    return dets

def get_poses(humans, person='second_person', pose_attr='opt_pose', trans_attr='opt_trans'):

    for attr in [pose_attr, 'opt_pose', 'pose', 'nothing']:
        if attr in humans[person]:
            pose = humans[person][attr]
            pose_attr = attr
            break
        if attr == 'nothing':
            print(f"'{trans_attr}' is not in {person}")
            return None

    for attr in [trans_attr, 'opt_trans', 'trans', 'mocap_trans', 'nothing']:
        if attr in humans[person]:
            trans = humans[person][attr]
            trans_attr = attr
            break
        if attr == 'nothing':
            print(f"'{trans_attr}' is not in {person}")
            return None
    
    if 'beta' not in humans[person]:
        humans[person]['beta'] = [0] * 10
    if 'gender' not in humans[person]:
        humans[person]['gender'] = 'neutral'

    gender = humans[person]['gender']
    betas  = humans[person]['beta']

    return  {"body_pose"    : pose[:, 3: 24 * 3].copy(), 
             "global_orient": pose[:, :3].copy(), 
             "smpl_betas"   : np.array(betas), 
             "trans"        : trans, 
             "gender"       : gender,
             "trans_attr"   : trans_attr,
             "pose_attr"    : pose_attr}

def load_sloper4d_data(pkl_results, 
                       person='second_person', 
                       pose_attr='opt_pose',
                       trans_attr='opt_trans',
                       rgb=None,
                       is_global=True):
                       
    rgb = [58, 147, 189] if rgb is None else rgb
    results = get_poses(pkl_results, person, pose_attr, trans_attr)
        
    if results is not None:
        trans_offset = np.zeros_like(results['trans'])
        if not is_global:
            trans_offset[:, :2] = results['trans'][:, :2]
        smpl_layer     = SMPLLayer(model_type='smpl', gender=results["gender"], device=C.device)
        sloper4d_smpl  = SMPLSequence(poses_body=results['body_pose'],
                            smpl_layer = smpl_layer,
                            poses_root = results['global_orient'],
                            trans      = results['trans'] - trans_offset,
                            betas      = results['smpl_betas'],
                            color      = (rgb[0]/255, rgb[1] / 255, rgb[2] / 255, 1.0),
                            name       = f"{person}_{results['pose_attr']}-annot",
                            z_up       = True)
        return {f"{person}_{results['pose_attr']}": sloper4d_smpl, "trans_offset": trans_offset}
    else:
        return {}

def load_point_cloud(pkl_results, person='second_person', points_num = 1024, trans=None):
    if 'point_clouds' not in pkl_results[person]:
        return None
    
    point_clouds = [np.array([0, 0, 0])] * len(pkl_results['frame_num'])

    for i, pf in enumerate(pkl_results[person]['point_frame']):
        point_clouds[pkl_results['frame_num'].index(pf)] = pkl_results[person]['point_clouds'][i]

    pp = np.array([fix_points_num(pts, points_num) for pts in point_clouds])
    ptc_sloper4d = PointClouds(points = pp - trans[:, None, :] if trans is not None else pp, 
                               color  = (58/255, 147/255, 189/255, 0.5), 
                               z_up   = True)
    return ptc_sloper4d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D viewer')
    parser.add_argument('-P', '--pkl_file', type=str, 
                        help='PKL file path')
    parser.add_argument('-N', '--person', type=str, default='second_person',
                        help='The person that to visualize. default to "second_person". ')
    parser.add_argument('--pose', type=str, default='opt_pose',
                        help='The pose that will be loaded from PKL file"')
    parser.add_argument('--is_global', type=bool, default=False,
                        help='whether to show the global translation of the person')
    parser.add_argument('-S', '--scene_path', type=str, default='',
                        help='The scene mesh path')

    v = Viewer()
    args = parser.parse_args()
    
    pkl_results = load_pkl(args.pkl_file)

    geometry_dict = {}

    geometry_dict.update(load_sloper4d_data(pkl_results, 
                                     person    = "first_person",
                                     pose_attr ='opt_pose',
                                     is_global = args.is_global))
    
    geometry_dict.update(load_sloper4d_data(pkl_results, 
                                     person    = args.person,
                                     pose_attr = args.pose,
                                     rgb       = [228, 100, 100],
                                     is_global = args.is_global))
    
    if args.scene_path is not None and os.path.exists(args.scene_path):
        scene_mesh = o3d.io.read_triangle_mesh(args.scene_path)
        scene_mesh_seq = Meshes(
                    np.asarray(scene_mesh.vertices)[None, ...],
                    np.asarray(scene_mesh.triangles),
                    is_selectable = False,
                    gui_affine    = False,
                    color         = (160 / 255, 160 / 255, 160 / 255, 1.0),
                    name          = "Scene",
                    rotation      = aa2rot_numpy(np.array([-1, 0, 0]) * np.pi/2))
        
        geometry_dict.update({"scene": scene_mesh_seq})
    
    point_cloud = load_point_cloud(pkl_results, trans = geometry_dict['trans_offset'])

    if point_cloud is not None:
        geometry_dict.update({"points": point_cloud})

    for _, geometry in geometry_dict.items():
        try:
            v.scene.add(geometry)
        except:
            pass

    v.run()
