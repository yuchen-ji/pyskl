#!/bin/bash

src_folder="/workspaces/pyskl/data_generate/videos_"
dst_folder="/workspaces/pyskl/data_generate/videos__"
target_framerate=15

mkdir -p "$dst_folder"

for video_file in "$src_folder"/*; do
    if [ -f "$video_file" ]; then
        dst_file="$dst_folder/$(basename "$video_file")"
        ffmpeg -i "$video_file" -vf "fps=$target_framerate" "$dst_file"
    fi
done