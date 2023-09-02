#!/bin/bash

src_folder="/workspaces/pyskl/data_generate/datasets_action_6/videos"
dst_folder="/workspaces/pyskl/data_generate/datasets_action_6/videos_"

mkdir -p "$dst_folder"

for video_file in "$src_folder"/*; do
    if [[ -f "$video_file" ]]; then
        filename=$(basename "$video_file")
        filename_no_extension="${filename%.*}"
        ffmpeg -i "$video_file" -c:v libx264 -c:a aac -strict experimental "$dst_folder/$filename_no_extension.mp4"
    fi
done
