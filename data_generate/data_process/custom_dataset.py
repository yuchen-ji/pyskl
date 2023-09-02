import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmcv import load, dump
from pyskl.smp import *


# Step 0：拍摄所需的视频，重命名视频文件名词，并生成符合规范标签文件
"""
TODO:
视频名称格式: {action}_{person}_{view}_0001.mp4
标签名称(包括测试集和训练集):
{
"vid_name": "{action}_{person}_{view}_0001",
"label": 0, 每一个action对应一个label, 从 0 开始 1, 2, 3, ...
"start_frame": 0,   动作的起始和终止帧, 因为我们一个video对应一个single action, 所以start=0, end=len(video)
"end_frame": 100
}
NOTE: 
拍摄视频的时候, 使用fps=30的手机拍摄, 之后调整成15帧的, 15/30帧都要测试一下
相机拍摄视角：(0, 60), delta=3, (0, 30), delta=6
单个动作的样本总数为: 20*5=100
"""


# # Step 1：导入标签文件和视频文件，
# train = load('data_generate/datasets_action_6/train.json')
# test = load('data_generate/datasets_action_6/test.json')
# tmpl = 'data_generate/datasets_action_6/videos__/{}.mp4'

# lines = [(tmpl + ' {}').format(x['vid_name'], x['label']) for x in train + test]
# mwlines(lines, 'data_generate/datasets_action_6/custom.list')


# Step 2: 运行关键点检测程序, 生成关键点label
"""
这里自己通过终端nohup运行2d关键点检测的程序
custom_2d_skeleton.py 依次需要接收3个参数: {GPU_NUM} --video-list {mwlines写入的文件位置} --out {输出标签的文件位置}
"""
# bash tools/dist_run.sh tools/data/custom_2d_skeleton.py 6 --video-list data_generate/datasets_action_6/custom.list --out data_generate/datasets_action_6/custom_annos.pkl

# Step 3: 合并训练集和测试集为一个pickle文件
train = load('data_generate/datasets_action_6/train.json')
test = load('data_generate/datasets_action_6/test.json')
annotations = load('data_generate/datasets_action_6/custom_annos.pkl')
split = dict()
split['train'] = [x['vid_name'] for x in train]
split['test'] = [x['vid_name'] for x in test]
dump(dict(split=split, annotations=annotations), 'data_generate/datasets_action_6/custom_hrnet.pkl')  # 最终的pickle文件，用于训练行为识别模型