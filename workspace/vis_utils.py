import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import os.path as osp
import cv2
import mmcv

def vis_pose_2d(context):
    
    # human pose
    coco = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
        (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
        (1, 0), (3, 1), (2, 0), (4, 2)
    ]
    
    # (1, frames, 17, 2)
    pose_2d = context['annotations'][0]['keypoint']
    pose_2d[..., 1] = 800-pose_2d[..., 1]
    
    # matplot
    plt.figure()
    sns.set_style('white')
    # 设置(x,y)等比例
    # ax = plt.gca()
    # ax.set_aspect(1)
    plt.axis('equal')
    # plt.xlim((0, 1400))
    # plt.ylim((0, 600))
    
    # plot pose
    for item in range(len(pose_2d[0])):
        plt.cla()
        plt.title(f'frame_{item}')
        for skl in coco:
            kp_x = (pose_2d[0, item, skl[0], 0], pose_2d[0, item, skl[1], 0])
            kp_y = (pose_2d[0, item, skl[0], 1], pose_2d[0, item, skl[1], 1])
            plt.plot(kp_x, kp_y)
            plt.savefig("workspace/demo/temp/frame_{}.png".format(item))


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('workspace/demo/tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))
        # 这个size是用来crop input_video
        frame = mmcv.imcrop(frame, np.array([140, 40, 440+140, 440+40]))
        # 这个size用于crop rviz_video
        # frame = mmcv.imcrop(frame, np.array([390, 80, 385+390, 385+80]))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


if __name__ == '__main__':
     
    # fname = 'workspace/data/nuaa5.pkl'
    # with open(fname, 'rb') as f:
    #     context = pk.load(f)
        
    # vis_pose_2d(context)
    
    # vname = 'workspace/demo/rviz_1.mp4'
    vname = 'workspace/demo/input_result.mp4'
    frame_extraction(vname, 480)
    