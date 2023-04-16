import os
import pickle
import argparse
import numpy as np

def add_manual_pose_to_seq(pkl_file, pose_file, person='second_person', seq=None):
    # 从 PKL 文件中加载序列数据
    with open(pkl_file, 'rb') as f:
        seq = pickle.load(f) if seq is None else seq
    
    # 从 NPZ 文件中提取 'pose' 字段
    data        = np.load(pose_file)
    poses_body  = data['poses_body']   # (n, 69)
    orientation = data['poses_root']   # (n, 3)
    pose        = np.concatenate((orientation, poses_body), axis=1)

    if person in seq:
        # 检查 'opt_pose' 字段的长度是否与 'pose' 字段相等
        if len(seq[person]['pose']) == pose.shape[0]:
            # 如果相等,则将 'pose' 字段添加到 'seq[person]' 中
            seq[person]['manual_pose'] = pose
            
            # 在原始 PKL 文件名中添加 "_manual" 后缀
            new_pkl_file = os.path.splitext(pkl_file)[0] + '_manual.pkl'
            
            # 将序列数据保存到新的 PKL 文件中
            with open(new_pkl_file, 'wb') as f:
                pickle.dump(seq, f)

            print(f'手动姿势已添加到 {person} 中, 并保存到新的PKL文件中：{new_pkl_file}')
            
        else:
            print('手动姿势未添加,因为opt_pose和pose长度不匹配。')
    else:
        print('手动姿势未添加,因为未找到指定人物。')
    
    # return seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将手动姿势添加到序列数据中。')
    parser.add_argument('-P', '--pkl_file', type=str,
                        help='包含序列数据的PKL文件的路径。')
    parser.add_argument('-O', '--pose_file', type=str, default='export\SMPL\manual_pose.npz',
                        help='包含姿势数据的NPZ文件的路径。')
    parser.add_argument('-N', '--person', type=str, default='second_person',
                        help='指定要更新手动姿势的人物。默认为"second_person"。')
    parser.add_argument('-S', '--seq', type=str, default=None,
                        help='指定一个序列数据变量的名称。如果不提供,则从PKL文件中加载。')
    args = parser.parse_args()

    add_manual_pose_to_seq(args.pkl_file, args.pose_file, args.person, seq=args.seq)
