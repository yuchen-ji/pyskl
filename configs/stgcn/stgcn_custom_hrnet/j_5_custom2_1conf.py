model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='custom2', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=5, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data_generate/datasets_action_5_10fps_120test/custom_hrnet_11kp_1conf.pkl'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='custom2', feats=['j']),
     dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='custom2', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='custom2', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10, seed=2),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,  # 单个gpu的batch-size
    workers_per_gpu=2,  # 单个gpu分配的数据加载线程
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='test'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

# optimizer
optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 30
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/stgcn_custom_5_custom2_1conf_8/'

# nohup bash tools/dist_train.sh configs/stgcn/stgcn_custom_hrnet/j_5_custom2_1conf.py 6 --validate --test-last --test-best > nohup.log 2>&1 &
# bash tools/dist_test.sh configs/stgcn/stgcn_custom_hrnet/j_5_custom2_1conf.py work_dirs/stgcn_custom_5_custom2_1conf_1/epoch_30.pth 6 --eval top_k_accuracy --out result.pkl
