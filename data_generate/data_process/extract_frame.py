import mmcv


# video = "data_generate/datasets_action_5_10fps_120test/videos_test/assembly_person1_0003_test.mp4"
# video = "data_generate/datasets_action_5_10fps_120test/videos_test/left_person1_0026_test.mp4"
# video = "data_generate/datasets_action_5_10fps_120test/videos_test/polish_person1_0051_test.mp4"
# video = "data_generate/datasets_action_5_10fps_120test/videos_test/right_person1_0075_test.mp4"
# video = "data_generate/datasets_action_5_10fps_120test/videos_test/stand_person1_0099_test.mp4"
video = "data_generate/datasets_action_6_10fps_test/videos__/putdown_person1_0143_test.mp4"


# 打开视频文件
video = mmcv.VideoReader(video)

# 输出帧的路径
output_dir = 'output_frames/'

# 确保输出目录存在，如果不存在则创建
mmcv.mkdir_or_exist(output_dir)

# 循环遍历视频的每一帧并写入文件
for i, frame in enumerate(video):
    frame_file = f'{output_dir}/frame_{i:04d}.jpg'  # 按照帧的索引命名文件
    mmcv.imwrite(frame, frame_file)

print(f'抽取并保存了{len(video)}帧到{output_dir}。')
