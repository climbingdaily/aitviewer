import numpy as np
import pickle as pkl
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

def get_poses(humans, person='second_person'):

    if 'opt_pose' in humans[person]:
        pose  = humans[person]['opt_pose']
    elif 'pose' in humans[person]:
        pose  = humans[person]['pose']
    else:
        return None

    if 'opt_trans' in humans[person]:
        trans = humans[person]['opt_trans']
    elif 'mocap_trans' in humans[person]:
        trans = humans[person]['mocap_trans']
    else:
        return None

    if 'beta' not in humans[person]:
        humans[person]['beta'] = [0] * 10
    if 'gender' not in humans[person]:
        humans[person]['gender'] = 'neutral'

    gender    = humans[person]['gender']
    betas     = humans[person]['beta']

    return  {"body_pose": pose[:, 3: 24 * 3].copy(), 
             "global_orient": pose[:, :3].copy(), 
             "smpl_betas": np.array(betas), 
             "global_trans": trans, 
             "gender": gender}

def load_sloper4d_data(pkl_results, name='SLOPER4D', person='second_person', rgb=[58, 147, 189]):
    results = get_poses(pkl_results, person)
    if results is None:
        return  None
    smpl_layer = SMPLLayer(model_type='smpl', gender=results["gender"], device=C.device)
    sloper4d_smpl  = SMPLSequence(poses_body=results['body_pose'],
                         smpl_layer = smpl_layer,
                         poses_root = results['global_orient'],
                         trans      = results['global_trans'],
                         betas      = results['smpl_betas'],
                         color      = (rgb[0]/255, rgb[1] / 255, rgb[2] / 255, 1.0),
                         name       = name,
                         rotation   = aa2rot_numpy(np.array([-1, 0, 0]) * np.pi/2))
    return sloper4d_smpl

def load_point_cloud(pkl_results, person='second_person', points_num = 2048):
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
    # smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
    # poses = np.zeros([10000, smpl_layer.bm.NUM_BODY_JOINTS * 3])
    # smpl_seq = SMPLSequence(poses, smpl_layer)
    v = Viewer()

    # pkl_path    = "C:\\Users\\DAI\\Desktop\\hsc4d\\2023-03-26T22_50_40__test.pkl"
    # scene_path  = "C:\\Users\\DAI\\Desktop\\hsc4d\\0312_mingpei_ym_01_6535frames.ply"

    # pkl_path    = "C:\\Users\\DAI\\Desktop\\sloper4d\\2023-03-09T12_22_59_all_term_test.pkl"
    # scene_path  = "C:\\Users\\DAI\\Desktop\\hsc4d\\0417_003_perfect_3551frames.ply"

    
    pkl_path    = "C:\\Users\\DAI\\Desktop\\sloper4d\\2023-03-30T11_13_02_seq002_test.pkl"
    scene_path  = "C:\\Users\\DAI\\Desktop\\sloper4d\\scene002_6871frames.ply"

    pkl_results = load_pkl(pkl_path)
    first_smpl  = load_sloper4d_data(pkl_results, 
                                     name   = 'first_person', 
                                     person = "first_person")
    second_smpl = load_sloper4d_data(pkl_results, 
                                     name   = 'second_person', 
                                     person = "second_person",
                                     rgb    = [228, 100, 100])
    scene_mesh = o3d.io.read_triangle_mesh(scene_path)
    scene_mesh_seq = Meshes(
                np.asarray(scene_mesh.vertices)[None, ...],
                np.asarray(scene_mesh.triangles),
                is_selectable=False,
                gui_affine=False,
                color=(160 / 255, 160 / 255, 160 / 255, 1.0),
                name="Scene",
                rotation   = aa2rot_numpy(np.array([-1, 0, 0]) * np.pi/2))
    
    point_cloud = load_point_cloud(pkl_results)

    if first_smpl is not None:
        v.scene.add(first_smpl)
    if second_smpl is not None:
        v.scene.add(second_smpl)
    if point_cloud is not None:
        v.scene.add(point_cloud)

    v.scene.add(scene_mesh_seq)
    v.run()