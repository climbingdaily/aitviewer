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
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.utils.so3 import aa2rot_numpy

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
             "global_trans" : trans, 
             "gender"       : gender,
             "trans_attr"   : trans_attr,
             "pose_attr"    : pose_attr}

def load_sloper4d_data(pkl_results, 
                       person='second_person', 
                       pose_attr='opt_pose',
                       trans_attr='opt_trans',
                       rgb=None):
                       
    rgb = [58, 147, 189] if rgb is None else rgb
    results = get_poses(pkl_results, person, pose_attr, trans_attr)

    if results is not None:
        smpl_layer     = SMPLLayer(model_type='smpl', gender=results["gender"], device=C.device)
        sloper4d_smpl  = SMPLSequence(poses_body=results['body_pose'],
                            smpl_layer = smpl_layer,
                            poses_root = results['global_orient'],
                            trans      = results['global_trans'],
                            betas      = results['smpl_betas'],
                            color      = (rgb[0]/255, rgb[1] / 255, rgb[2] / 255, 1.0),
                            name       = f"{person}_{results['pose_attr']}-after_annot",
                            #  rotation   = aa2rot_numpy(np.array([-1, 0, 0]) * np.pi/2)
                            z_up       = True)
        return sloper4d_smpl
    else:
        return None

def load_point_cloud(pkl_results, person='second_person', points_num = 1024):
    if 'point_clouds' not in pkl_results[person]:
        return None
    
    point_clouds = [np.array([0, 0, 0])] * len(pkl_results['frame_num'])

    for i, pf in enumerate(pkl_results[person]['point_frame']):
        point_clouds[pkl_results['frame_num'].index(pf)] = pkl_results[person]['point_clouds'][i]

    pp = np.array([fix_points_num(pts, points_num) for pts in point_clouds])
    ptc_sloper4d = PointClouds(points=pp, 
                            # position=np.array([1.0, 0.0, 0.0]), 
                            color=(149/255, 85/255, 149/255, 0.5), 
                            z_up=True)
    return ptc_sloper4d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将手动姿势添加到序列数据中。')
    parser.add_argument('-P', '--pkl_file', type=str, 
                        default='D:\\Yudi Dai\\Documents\\Downloads\\humans_param.pkl',
                        help='包含序列数据的PKL文件的路径。')
    parser.add_argument('-N', '--person', type=str, default='second_person',
                        help='指定要更新手动姿势的人物。默认为"second_person"。')
    parser.add_argument('-S', '--scene_path', type=str, 
                        default="C:\\Users\\DAI\\Desktop\\sloper4d\\scene002_6871frames.ply",
                        help='指定一个序列数据变量的名称。如果不提供,则从PKL文件中加载。')

    v = Viewer()
    args = parser.parse_args()
    
    pkl_results = load_pkl(args.pkl_file)

    geometry_list = []

    geometry_list.append(load_sloper4d_data(pkl_results, 
                                     person = "first_person",
                                     pose_attr='opt_pose'))
    
    geometry_list.append(load_sloper4d_data(pkl_results, 
                                     person = args.person,
                                     pose_attr='opt_pose',
                                     rgb    = [228, 100, 100]))
    
    if args.scene_path is not None and os.path.exists(args.scene_path):
        scene_mesh = o3d.io.read_triangle_mesh(args.scene_path)
        scene_mesh_seq = Meshes(
                    np.asarray(scene_mesh.vertices)[None, ...],
                    np.asarray(scene_mesh.triangles),
                    is_selectable=False,
                    gui_affine=False,
                    color=(160 / 255, 160 / 255, 160 / 255, 1.0),
                    name="Scene",
                    rotation   = aa2rot_numpy(np.array([-1, 0, 0]) * np.pi/2))
        geometry_list.append(scene_mesh_seq)
    
    point_cloud = load_point_cloud(pkl_results)
    if point_cloud is not None:
        geometry_list.append(point_cloud)

    for geometry in geometry_list:
        v.scene.add(geometry)

    v.run()
