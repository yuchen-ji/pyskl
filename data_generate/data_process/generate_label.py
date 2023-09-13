import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subprocess
import json
import re


def get_label_num(fname):
    if "stand" in fname:
        return 0
    if "left" in fname:
        return 1
    if "right" in fname:
        return 2
    if "polish" in fname:
        return 3
    if "assembly" in fname:
        return 4
    if "putdown" in fname:
        return 5
    

def get_video_frame_count(video_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-an',
        '-vcodec', 'copy',
        '-f', 'null',
        '-'
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stderr

    frame_count_match = re.search(r"frame=\s*(\d+)", output)
    if frame_count_match:
        frame_count = int(frame_count_match.group(1))
        return frame_count
    else:
        return None
    
    
if __name__ == '__main__':
    
    train_label = []
    test_label = []
    dir_name = "/workspaces/pyskl/data_generate/datasets_action_5_10fps_120test/videos_all"
    train_fname = "/workspaces/pyskl/data_generate/datasets_action_5_10fps_120test/train.json"
    test_fname = "/workspaces/pyskl/data_generate/datasets_action_5_10fps_120test/test.json"
    
    fnames = os.listdir(dir_name)
    for fname in fnames:
        info = {}
        info['vid_name'] = fname[:-4]
        info['label'] = get_label_num(fname)
        info['start_frame'] = 0
        info['end_frame'] = get_video_frame_count(os.path.join(dir_name, fname))
        
        if 'train' in fname:
            train_label.append(info)
        if 'test' in fname:
            test_label.append(info)
            
    with open(train_fname, 'w') as f:
        json.dump(train_label, f)
        f.write('\n')
        
        
    with open(test_fname, 'w') as f:
        json.dump(test_label, f)
        f.write('\n')
    