import pickle as pkl
import os, sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')

# from myhrc.dataset_utils import read_pkl

def read_pkl(fname):
    with open(fname, 'rb') as f:
        data = pkl.load(f)
    return data
    

def vis_3d_pose(pkl_file):
    inhard = [
        [1, 8],
        [1, 5],
        [1, 2],
        [5, 6],
        [6, 7],
        [2, 3],
        [3, 4],
        [8, 9],
        # [10, 14],
        [14, 15],
        [15, 16],
        [16, 17],
        [10, 11],
        [11, 12],
        [12, 13]
    ]
    inhard = np.array(inhard) - 1

    chico = [
        [0,1], 
        [1,2], 
        [2,3], 
        [0,4], 
        [4,5], 
        [5,6], 
        [1,9], 
        [4,12], 
        [8,7], 
        [8,9], 
        [8,12], 
        [9,10], 
        [10,11],
        [12,13], 
        [13,14]
    ]

    def update(index):
        print(index)
        # ax.lines = []
        # ax.collections = []
        ax.cla()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(-100, 100)
        ax.set_xlim(-100, 100)
        ax.set_zlim(-100, 100)
        
        for skeleton in inhard:
            endpoint1 = skeleton[0]
            endpoint2 = skeleton[1]
            endpoint_x = [data_3d[index, 0, endpoint1, 0], data_3d[index, 0, endpoint2, 0]]
            endpoint_y = [data_3d[index, 0, endpoint1, 1], data_3d[index, 0, endpoint2, 1]]
            endpoint_z = [data_3d[index, 0, endpoint1, 2], data_3d[index, 0, endpoint2, 2]]
            # print(endpoint_x, endpoint_y, endpoint_z)
            ax.plot(endpoint_x, endpoint_z, endpoint_y, c='r')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # load data
    data_3d = read_pkl(pkl_file)
    data_3d = data_3d[10]['keypoint'][0]
    data_3d = np.array(data_3d, dtype='float32')

    frame_num = data_3d.shape[0]
    data_3d = data_3d.reshape(frame_num, -1, 17, 3)
    mean = np.mean(data_3d, axis=2)
    mean = mean[:, :, np.newaxis, :]

    data_3d = data_3d - mean
    val = np.mean(data_3d, axis=2)

    anim = FuncAnimation(fig, update, frames=frame_num, interval=40, repeat=False, cache_frame_data=True)    
    # anim.save("workspace/demo/vis_3d_pose.mp4")

    plt.show()
    
    
def vis_2d_pose(fname):
    fig = plt.figure()
    ax = fig.add_subplot()
    context = read_pkl(fname)
    data_2d = context['split']['xsub_val'][10]
    
    frame_num = len(data_2d)
    


if __name__ == '__main__':
    # pkl_file = "myhrc/mydataset/chico_3d_skeleton/train.pkl"
    fname = "myhrc/mydataset/inhard_train.pkl"
    # vis_3d_pose(fname)
    vis_2d_pose(fname)