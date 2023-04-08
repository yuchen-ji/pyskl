#!/bin/bash

# 指定要删除文件的文件夹路径
FILE_FOLDER="/workspaces/pyskl/workspace/demo/videos"

# 遍历文件夹中的所有视频文件
for video_file in "$VIDEO_FOLDER"/*.mp4; do

  # 获取视频文件名（不包括扩展名）
  filename=$(basename "$video_file" .mp4)

  # 使用 FFmpeg 将帧率设置为 15 帧每秒
  ffmpeg -i "$video_file" -r 15 "$VIDEO_FOLDER/${filename}_15fps.mp4"

  rm "$video_file"
  echo "Deleted file: $video_file"

done