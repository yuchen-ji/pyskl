import torch
from mmcv import load, dump
from pyskl.smp import *
from pyskl.models import build_model
from time import time
from tqdm import tqdm

layout = 'coco'
graph = dict(layout=layout, mode='spatial')
graph_rdm = dict(layout=layout, mode='random', num_filter=8, init_off=.04, init_std=.02)
graph_bin = dict(layout=layout, mode='binary_adj')
aagcn_cfg = dict(type='AAGCN', graph_cfg=graph)
ctrgcn_cfg = dict(type='CTRGCN', graph_cfg=graph)
dgstgcn_cfg = dict(type='DGSTGCN', gcn_ratio=0.125, gcn_ctr='T', gcn_ada='T', tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'], graph_cfg=graph_rdm)
msg3d_cfg = dict(type='MSG3D', graph_cfg=graph_bin)
stgcn_cfg = dict(type='STGCN', graph_cfg=graph)
stgcnpp_cfg = dict(type='STGCN', gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn', graph_cfg=graph)

cfg_map = dict(stgcn=stgcn_cfg)

batch = 1
warmup = 10
iters = 100
num_joints = {'nturgb+d': 25, 'coco': 17, 'openpose': 18}[layout]
num_person = 1
seq_len = 100

# Measure FPS with GPU
device = 'cuda:0'
for k, v in cfg_map.items():
    gcn = build_model(v)
    gcn = gcn.to(device)
    gcn.eval()
    start = 0
    for i in range(warmup + iters):
        if i == warmup:
            start = time()
        inp = torch.randn(batch, num_person, seq_len, num_joints, 3).to(device)
        with torch.no_grad():
            out = gcn(inp)
    end = time()
    # （推理次数 / 时间）表示1s进行多少次的预测
    fps = (batch * iters) / (end - start)
    print(f'{k} FPS: {fps}')