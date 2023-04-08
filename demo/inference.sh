#!/bin/bash

# 指定视频文件夹路径
VIDEO_FOLDER=$1
OUT_FOLDER=$2

# 遍历文件夹中的所有视频文件
for video_file in "$VIDEO_FOLDER"/*.mp4; do

  # 获取视频文件名（不包括扩展名）
  filename=$(basename "$video_file" .mp4)

  # 使用 Python 执行脚本并修改参数
  python demo/demo_skeleton_coco_skl.py --video "$video_file" --out_filename "$OUT_FOLDER/${filename}_result.mp4"

done
