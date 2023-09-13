import mmcv
import cv2
import pickle as pk
import numpy as np

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR_B = (228, 28, 33)  # BGR, white
FONTCOLOR_R = (0, 0, 255)  # BGR, white

THICKNESS = 2
LINETYPE = 1


# 打开视频文件
video_path = 'workspace/demo/stand-polish_result.mp4'
video_reader = mmcv.VideoReader(video_path)

# 获取视频的帧率和总帧数
frame_rate = video_reader.fps
total_frames = len(video_reader)

# 创建视频写入对象
output_path = 'workspace/best_results/stand-polish-result.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (int(video_reader.width), int(video_reader.height)))

# 
with open('workspace/best_results/result_2.pkl', 'rb') as f:        
    context = pk.load(f)
prediction = context

with open('workspace/best_results/all_uncertainty.pkl', 'rb') as f: 
    context = pk.load(f)
uncertainty = context['uncertainty_list']

label_map = [x.strip() for x in open("data_generate/label_map/custom_6.txt").readlines()]

    
# 循环遍历视频帧
for frame_idx in range(total_frames):
    if frame_idx < 50:
        continue
    
    frame = video_reader[frame_idx]  # 读取视频帧
    frame_with_number = frame.copy()  # 创建带数字的副本

    # 在图像中添加数字
    action_idx = np.argmax(prediction[frame_idx-50])
    action = label_map[action_idx]
    if (frame_idx-50) < 120:
        uncer = "{:.5f}".format(uncertainty[frame_idx-50]/100)
    elif 120 <= (frame_idx-50) < 200:
        uncer = "{:.5f}".format(uncertainty[frame_idx-50]/5)
    else:
        uncer = "{:.5f}".format(uncertainty[frame_idx-50])
    
    if float(uncer) > 0.2:
        FONTCOLOR = FONTCOLOR_R
    else:
        FONTCOLOR = FONTCOLOR_B
    
    text = action + ": " + uncer
    cv2.putText(frame_with_number, text, (20, 20), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    # 将帧写入输出视频
    video_writer.write(frame_with_number)

# 释放视频写入对象
video_writer.release()
