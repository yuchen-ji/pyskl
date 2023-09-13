import os
import os.path as osp

import cv2
import numpy as np
import mmcv
from pyskl.apis import inference_recognizer, init_recognizer
from demo.utils import slide_window
import copy
import shutil


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 1.5
FONTCOLOR = (228, 28, 33)  # BGR, white
THICKNESS = 2
LINETYPE = 1


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



if __name__ == '__main__':


    fname = 'data_generate/data_from_mmhuman/inference_result.npz'
    vname = 'demo/long-horizon_10fps.mp4'
    # cname = 'configs/stgcn/stgcn_custom_hrnet/j_6_custom2.py'
    cname = 'configs/stgcn/stgcn_custom_hrnet/j_6_custom2_1conf.py'
    # checkpoint = 'work_dirs/stgcn_custom_6_custom2_1/epoch_20.pth'
    checkpoint = 'work_dirs/stgcn_custom_6_custom2_1conf/epoch_20.pth'
    lname = 'data_generate/label_map/custom_6.txt'
    oname = 'demo/demo_mmhuman.mp4'
    device = 'cuda:0'


    # 加载mmhuman推理的2d关键点
    context = np.load(fname, allow_pickle=True)
    keypoint = context['body_pose_2d'][np.newaxis,...]
    keypoint_score = context['body_pose_score'][np.newaxis,...]
    context.close()

    # 加载原始视频
    frame_paths, original_frames = frame_extraction(vname, 1080)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # 定义骨骼数据格式
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)


    # 读取配置文件
    config = mmcv.Config.fromfile(cname)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    GCN_flag = 'GCN' in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        GCN_nperson = format_op['num_person']

    # 加载标签映射文件
    label_map = [x.strip() for x in open(lname).readlines()]

    # 初始化行为识别模型
    model = init_recognizer(config, checkpoint, device)
    
    # 重组骨骼结构
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

    keypoint_score = np.ones_like(keypoint_score)

    start_frame = 20
    fake_list = slide_window(fake_anno, keypoint, keypoint_score, start_frame)


    label_list = []
    for idx, item in enumerate(fake_list):
        data = copy.deepcopy(item)
        results = inference_recognizer(model, data)
        action_label = label_map[results[0][0]]
        label_list.append(f"{action_label}: {results[0][1]}")
        # print(f"{action_label}: {results[0][1]}")
   
    label_list[0:0] = [label_list[0] for i in range(num_frame - len(label_list))]
    
    vis_frames = [
        mmcv.imread(frame_paths[i])
        for i in range(num_frame)
    ]

    for item in range(num_frame):
        cv2.putText(vis_frames[item], label_list[item], (40, 40), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=10)
    vid.write_videofile(oname, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)