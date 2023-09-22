# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
import argparse
import cv2
import mmcv
import numpy as np
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer

import time
import copy
import pickle
from operator import itemgetter
from demo.utils import slide_window, get_pred_uncertainty

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (228, 28, 33)  # BGR, white
THICKNESS = 2
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')

    """
    data_generate/datasets_action_6/videos__/assembly_person1_0004_test.mp4
    data_generate/datasets_action_6/videos__/assembly_person1_0010_test.mp4
    data_generate/datasets_action_6/videos__/assembly_person1_0016_test.mp4
    data_generate/datasets_action_6/videos__/left_person1_0001_test.mp4
    data_generate/datasets_action_6/videos__/left_person1_0007_test.mp4
    data_generate/datasets_action_6/videos__/left_person1_0013_test.mp4
    data_generate/datasets_action_6/videos__/polish_person1_0003_test.mp4
    data_generate/datasets_action_6/videos__/polish_person1_0009_test.mp4
    data_generate/datasets_action_6/videos__/polish_person1_0015_test.mp4
    data_generate/datasets_action_6/videos__/putdown_person1_0005_test.mp4
    data_generate/datasets_action_6/videos__/putdown_person1_0011_test.mp4
    data_generate/datasets_action_6/videos__/putdown_person1_0017_test.mp4
    data_generate/datasets_action_6/videos__/right_person1_0002_test.mp4
    data_generate/datasets_action_6/videos__/right_person1_0008_test.mp4
    data_generate/datasets_action_6/videos__/right_person1_0014_test.mp4
    data_generate/datasets_action_6/videos__/stand_person1_0000_test.mp4
    data_generate/datasets_action_6/videos__/stand_person1_0006_test.mp4
    data_generate/datasets_action_6/videos__/stand_person1_0012_test.mp4
    """

    parser.add_argument('--video', default='demo/videos/long_view2.mp4', help='video file/url')
    
    parser.add_argument('--out_filename', default='demo/results/long_view2_result.mp4', help='output filename')
    parser.add_argument(
        '--config',
        # default='configs/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j_custom.py',
        # default='configs/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j.py',
        # default='configs/stgcn/stgcn_custom_hrnet/j.py',
        # default='configs/stgcn/stgcn_custom_hrnet/j_6.py',
        default='configs/stgcn/stgcn_custom_hrnet/j_5_custom2_1conf.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        # default='checkpoints/j.pth',
        # default='work_dirs/stgcn/stgcn_pyskl_nuaa6/j_4/epoch_16.pth',
        # default='work_dirs/stgcn/stgcn_pyskl_factory/jj/epoch_16.pth',
        # default='work_dirs/stgcn_custom/epoch_30.pth',
        default='work_dirs/stgcn_custom_6_custom2_1/epoch_20.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        # default='tools/data/label_map/nturgbd_120.txt',
        default='data_generate/label_map/custom_6.txt',
        # default='workspace/label_map/nuaa6.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
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

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


def main():
    args = parse_args()

    frame_paths, original_frames = frame_extraction(args.video,
                                                    args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    
    GCN_flag = 'GCN' in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        GCN_nperson = format_op['num_person']

    model_idx = [2, 3, 4, 5]
    model_list = []
    for idx in model_idx:
        device = f"cuda:{idx}"
        checkpoint = f"work_dirs/stgcn_custom_5_custom2_1conf_{idx}/epoch_30.pth"
        model = init_recognizer(config, checkpoint, device)
        model_list.append(model)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Get Human detection results
    det_results = detection_inference(args, frame_paths)
    torch.cuda.empty_cache()

    pose_results = pose_inference(args, frame_paths, det_results)
    torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    # 使用GCN model
    if GCN_flag:
        # We will keep at most `GCN_nperson` persons per frame.
        tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
        keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
        
        new_kp = [0, 5, 6, 7, 8, 9, 10, 11, 12]
        lshoulder = keypoint[..., 5, :]
        rshoulder = keypoint[..., 6, :]
        lhip = keypoint[..., 11, :]
        rhip = keypoint[..., 12, :]
        neck = 1/2 * (lshoulder + rshoulder)
        midhip = 1/2 * (lhip + rhip)     
        keypoint = keypoint[..., new_kp, :]
        keypoint = np.concatenate(
            (keypoint, neck[..., np.newaxis, :], midhip[..., np.newaxis, :]),
            axis=2
        )
        
        lshoulder_score = keypoint_score[..., 5]
        rshoulder_score = keypoint_score[..., 6]
        lhip_score = keypoint_score[..., 11]
        rhip_score = keypoint_score[..., 12]
        neck_score = 1/2 * (lshoulder_score + rshoulder_score)
        midhip_score = 1/2 * (lhip_score + rhip_score)
        
        keypoint_score = keypoint_score[..., new_kp]
        keypoint_score = np.concatenate(
            (keypoint_score, neck_score[..., np.newaxis], midhip_score[..., np.newaxis]),
            axis=2
        )
        # 将骨骼点的score置为1
        keypoint_score = np.ones_like(keypoint_score)
        
        start_frame = 20
        fake_list = slide_window(fake_anno, keypoint, keypoint_score, start_frame)

    else:
        num_person = max([len(x) for x in pose_results])
        # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
        num_keypoint = 17
        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                  dtype=np.float16)
        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

    # 实时推理
    action_list = []
    label_list = []
    uncertainties = []
    predictions = []
    for idx, item in enumerate(fake_list):
        
        # result = inference_recognizer(model, data)
        # action = label_map[result[0][0]]
        # label_list.append(action)
           
        results = []
        for model in model_list:
            data = copy.deepcopy(item)
            result = inference_recognizer(model, data)
            result = sorted(result, key=lambda x: x[0])
            result = [item[1] for item in result]
            results.append(result)
            
        results = np.array(results)
        pred, uncertainty = get_pred_uncertainty(results)

        action = label_map[np.argmax(pred)]
        action_list.append(action)
        label_list.append(f"{action}: {round(uncertainty, 5)}")
        
        predictions.append(np.max(pred))
        uncertainties.append(uncertainty)
        
    # 预测的实时的不确定度写入pkl文件
    info = {}
    action_list[0:0] = [action_list[0]] * (num_frame - len(action_list))
    predictions[0:0] = [predictions[0]] * (num_frame - len(predictions))
    uncertainties[0:0] = [uncertainties[0]] * (num_frame - len(uncertainties))
    info['action'] = action_list
    info['predictions'] = predictions
    info['uncertainties'] = uncertainties
    with open('demo/results/long_view2_uncertainty.pkl', 'wb') as f:
        pickle.dump(info, f)
         
    # 前xx帧的识别结果为第一次检测的类别
    label_list[0:0] = [label_list[0]] * (num_frame - len(label_list))
    
    # 骨骼及动作标签可视化
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i], dataset='CustomDataset')
        # vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]
    for item in range(num_frame):
        
        # 更新label的颜色  
        label = label_list[item]
        uncer = float(label[label.index(':')+1:])
        if uncer <= 0.2:
            FONTCOLOR = (228, 28, 33)
        else:
            FONTCOLOR = (33, 28, 228)
            
        # 添加是第几帧
        frame = f"Frame: {item}"
        label_list[item] = label + frame
                
        cv2.putText(vis_frames[item], label_list[item], (20, 20), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=10)
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)

if __name__ == '__main__':
    main()
