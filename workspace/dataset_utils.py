import pickle as pkl
import os, sys
import numpy as np
import bvh_converter as bvh
import pandas as pd
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
from pymo.viz_tools import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')

import shutil
import random


sys.path.insert(0,"/home/yuchen/projects/mmaction2/")

# ''' 
# 3d data format
# [{}, {}, ..., {}]
{
    'frame_dir': '',
    'label': 0,
    'keypoint': ['1', 'frame_num', 'keypoints_num', 3],
    'total_frames': 120,
}

# stgcn use 17 keypoints 
def read_pkl(filename):
    file = open(filename, 'rb')
    context = pkl.load(file)
    return context

# process dataset
def process_pkl(pkl_dirs):
    pkl_list = os.listdir(pkl_dirs)
    pkl_all = []
    for pkl_name in pkl_list:
        if pkl_name[8:15] == "action2" or pkl_name[8:13] == "stand":
            file = open(os.path.join(pkl_dirs, pkl_name), 'rb')
            context = pkl.load(file)

            if pkl_name[8:13] == "stand":
                context['label'] = 1
            if pkl_name[8:15] == "action2":
                context['label'] = 0          
            pkl_all.append(context)

    with open(r'myhrc/mydataset/val__w_2_6.pkl', 'wb') as f:
        pkl.dump(pkl_all, f)


def process_pkl_2(pkl_dirs):
    pkl_all = []
    i = 0
    pkl_list = os.listdir(pkl_dirs)
    for pkl_name in pkl_list:
        file = open(os.path.join(pkl_dirs, pkl_name), 'rb')
        context = pkl.load(file)
        # create action format
        action = {}
        action['frame_dir'] = 'undefined'
        action['label'] = i
        action['img_shape'] = (1920, 1080)
        action['original_shape'] = (1920, 1080)
        action['total_frames'] = len(context[:120])
        action['keypoint'] = np.array([frame[0] for frame in context[:120]])[np.newaxis,:]
        action['keypoint_score'] = np.ones_like(action['keypoint'])[...,0]
        pkl_all.append(action)    
        i += 1   

    with open(r'myhrc/mydataset/chico_3d_skeleton/train.pkl', 'wb') as f:
        pkl.dump(pkl_all, f)


def process_pkl_3(filename):
    context = read_pkl(filename)
    anno = context['annotations']
    xsub_train = set(context['split']['xsub_train'])
    xsub_val = set(context['split']['xsub_val'])

    # store train & val files
    train_set = []
    val_set = []

    for each in anno:
        if (each['frame_dir'] in xsub_train):
            train_set.append(each)
        if (each['frame_dir'] in xsub_val):
            val_set.append(each)

    with open(r'data/posec3d/ntu60_xsub_train_3d.pkl', 'wb') as f:
        pkl.dump(train_set, f)

    with open(r'data/posec3d/ntu60_xsub_val_3d.pkl', 'wb') as f:
        pkl.dump(val_set, f)


def data2pkl3d(data_3d, label):
    action = {}
    action['frame_dir'] = 'undefined'
    action['label'] = label
    action['keypoint'] = data_3d[np.newaxis,...]
    action['total_frames'] = len(data_3d)
    return action

def csv2pkl():

    actions = ['Assemble system', 'Consult sheets', 'No action', 'Picking in front', 'Picking left', 'Put down component', 
                'Put down measuring rod', 'Put down screwdriver', 'Put down subsystem', 'Take component', 'Take measuring rod',
                'Take screwdriver', 'Take subsystem', 'Turn sheets']

    # 17 joints * 3 coordinates, 顺序是这个数据集定义的关节顺序
    joints = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 
                'Spine', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']

    # 加载所有待处理的csv文件
    dst_pkl = []
    # src_dir = "/home/yuchen/projects/dataset/Segmented/pkl/train"
    src_dir = "/home/yuchen/projects/dataset/Segmented/pkl/val"

    for action in actions[:3]:
        path = os.path.join(src_dir, action)
        csv_list = os.listdir(path)
        
        for each in csv_list:
            csv_path = os.path.join(path, each)
            reader = pd.read_csv(csv_path)

            head = reader.columns.values
            title = [head[id] for id in range(len(head)) if head[id][:-2] in set(joints)]
            title_ids = np.array([id for id in range(len(head)) if head[id][:-2] in set(joints)])

            sort = []
            for i in range(len(joints)):
                jx = joints[i] + '.X'
                jy = joints[i] + '.Y'
                jz = joints[i] + '.Z'
                ix = title.index(jx)
                iy = title.index(jy)
                iz = title.index(jz)
                sort.extend([ix, iy, iz])

            title_ids = title_ids[sort]  
            data_3d = reader.iloc[:, title_ids].values
            data_3d = data_3d.reshape(-1,17,3)[::8,...]
            data_3d = data2pkl3d(data_3d, label = actions.index(action))
            dst_pkl.append(data_3d)
    
    random.shuffle(dst_pkl)
    with open('/home/yuchen/projects/dataset/Segmented/pkl/inhard_val.pkl', 'wb') as f:
        pkl.dump(dst_pkl, f)
    

def csv2pkl_single(filename):

    # 17 joints * 3 coordinates, 顺序是这个数据集定义的关节顺序
    joints = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 
                'Spine', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']

    reader = pd.read_csv(filename)
    head = reader.columns.values
    title = [head[id] for id in range(len(head)) if head[id][:-2] in set(joints)]
    title_ids = np.array([id for id in range(len(head)) if head[id][:-2] in set(joints)])

    sort = []
    for i in range(len(joints)):
        jx = joints[i] + '.X'
        jy = joints[i] + '.Y'
        jz = joints[i] + '.Z'
        ix = title.index(jx)
        iy = title.index(jy)
        iz = title.index(jz)
        sort.extend([ix, iy, iz])

    title_ids = title_ids[sort]  
    data_3d = reader.iloc[1:, title_ids].values
    data_3d = data_3d.reshape(-1,17,3)

    data2pkl3d(data_3d)


def bvh2np(filename):
    parser = BVHParser()
    data_3d = parser.parse(filename)
    
    data_pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('abdolute_translation_deltas')),
        ('const', ConstantsRemover()),
        ('np', Numpyfier()),
        ('down', DownSampler(2)),
        ('stdscale', ListStandardScaler())
    ])

    piped_data = data_pipe.fit_transform([data_3d])

    mp = MocapParameterizer('position')
    positions = mp.fit_transform([data_3d])

    nb_play_mocap(positions[0], 'pos', 
              scale=2, camera_z=800, frame_time=1/120, 
              base_url='pymo/mocapplayer/playBuffer.html')


def split(action_dir):

    S_train=["P01_R01", "P01_R03", "P03_R01", "P03_R03", "P03_R04", "P04_R02", "P05_R03", "P05_R04", 
                "P06_R01", "P07_R01", "P07_R02", "P08_R02", "P08_R04", "P09_R01", "P09_R03", "P10_R01", 
                "P10_R02", "P10_R03", "P11_R02", "P12_R01", "P12_R02", "P13_R02", "P14_R01", "P15_R01", 
                "P15_R02", "P16_R02"]
    S_val=["P01_R02", "P02_R01", "P02_R02", "P04_R01", "P05_R01", "P05_R02", 
            "P08_R01", "P08_R03", "P09_R02", "P11_R01", "P14_R02", "P16_R01"]

    src_dir = "/home/yuchen/projects/dataset/Segmented/pkl"
    dst_dir = "/home/yuchen/projects/dataset/Segmented/pkl"

    csv_train = []
    csv_val = []

    csv_list = os.listdir(os.path.join(src_dir, action_dir))
    for each in csv_list:
        if each[:7] in S_train:
            csv_train.append(each)
        if each[:7] in S_val:
            csv_val.append(each)

    if not os.path.exists(os.path.join(dst_dir, 'train', action_dir)):
        os.mkdir(os.path.join(dst_dir, "train", action_dir))
    if not os.path.exists(os.path.join(dst_dir, 'val', action_dir)):
        os.mkdir(os.path.join(dst_dir, 'val', action_dir))


    for each in csv_train:
        shutil.copy(os.path.join(src_dir, action_dir, each), os.path.join(dst_dir, 'train', action_dir, each))
    for each in csv_val:
        shutil.copy(os.path.join(dst_dir, action_dir, each), os.path.join(dst_dir, 'val', action_dir, each))


if __name__ == "__main__":

    # pkl_dirs = "myhrc/mydataset/train_60-80_new/annotation"
    # pkl_dirs = "myhrc/mydataset/validation_60-80_new/annotation"
    # pkl_dirs = "myhrc/mydataset/chico_3d_skeleton/S00"
    # 
    # pkl_file = 'data/posec3d/ntu120_xsub_train.pkl'
    # pkl_file = 'myhrc/mydataset/chico_3d_skeleton/S00/hammer.pkl'
    # pkl_file = 'myhrc/mydataset/train__action_1_5.pkl'
    # pkl_file = 'myhrc/mydataset/chico_3d_skeleton/train.pkl'
    # pkl_file = 'data/posec3d/ntu60_3danno.pkl'
    pkl_file = 'myhrc/mydataset/inhard_val.pkl'

    read_pkl(pkl_file)
    # process_pkl_2(pkl_dirs)
    # process_pkl(pkl_dirs)
    # process_pkl_3(pkl_file)

    # filename = "/home/yuchen/projects/dataset/Segmented/pkl/Assemble system/P01_R01_0013.84_0018.88_worldpos.csv"
    # filename = "/home/yuchen/projects/dataset/Segmented/SkletonSegmented/Assemble system/P01_R01_0013.84_0018.88.bvh"
    csv2pkl()
    # bvh2np(filename)

    # split("No action")