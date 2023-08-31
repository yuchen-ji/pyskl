# 制作Custom数据集

## 0、数据收集
**动作集合：[站立, 伸左手(来回)，抛光，装配，放下，右转弯腰（左手），伸手（左）]**
### Step 1
先尝试较为固定的视角 (35, 50)，delta=3 \
俯仰角 (NONE)，3组 \
简单动作：[站立, 伸手，右转弯腰] \
拍摄一个video

### Step 2
增加视角，扩充数据集 \
偏航角：(30, 75), delta=3 \
俯仰角: (NONE)，5组 \
同一位置：3次动作 \
复杂动作：[站立，伸左手，右转，抛光，装配，放下] \
总视频数：$16*5*3*6=1440$


## 1、数据处理
### 1.1、首先处理视频格式和帧率
```bash
cd data_generate
unzip videos.zip
# 先将*.MOV转成*.MP4, 写入videos_文件夹
data_generate/data_process/process_type.sh
# 将视频fps调整为15, 写入videos__文件夹
data_generate/data_process/process_fps.sh

# NOTE: 最后手动将用于分隔的黑色视频删除
```

### 1.2、将视频处理按类别重命名(`data_generate/data_process/rename_videos.py`)
重写改写map(标签,视频初始名称)
```python
"""
如下所示，因为视频是按照一定顺序拍摄的，且视频的初始名称也为顺序编号，
所以，请重写视频名称(index)和动作类别的关系
"""

def get_action(index, split):
    name = 'NULL'
    action = ['stand', 'left', 'right']
    if split == 'train':
        if index % 6 == 0 or index % 6 == 1:
            name = action[0]
        if index % 6 == 2 or index % 6 == 3:
            name = action[1]
        if index % 6 == 4 or index % 6 == 5:
            name = action[2]
    
    if split == 'test':
        if index % 3 == 0:
            name = action[0]
        if index % 3 == 1:
            name = action[1]
        if index % 3 == 2:
            name = action[2]
    return name
```
### 1.3、指定运行参数
```python
# NOTE：请务必先处理train，再处理test数据

# file_path为上一步处理后的视频的文件夹路径，默认为videos__
file_path = "data_generate/videos__"
# 指定现在处理的是训练集还是测试集
split = "train"
# 指定视频开始编号和结尾编号，train和test的视频分别指定
start_idx = 2970
end_idx = 2980
```
### 1.4、运行文件
```bash
python data_generate/data_process/rename_videos.py
```

## 2、根据处理后的视频生成标签文件(`data_generate/data_process/generate_label.py`)
### 2.1、修改代码
```python
"""
根据名称得到对应的label_index,自己指定，唯一，从index=0开始
"""

def get_label_num(fname):
    if "stand" in fname:
        return 0
    if "left" in fname:
        return 1
    if "right" in fname:
        return 2
```
### 2.2、同时创建标签map文件
在任意位置创建txt文件，按照index的顺序依次写入具体的动作名称。\
请参考: `data_generate/label_map/custom_3.txt`
### 2.3、修改参数
```python
# dir_name为你上一步重新命名后的文件的路径
dir_name = "data_generate/videos__"
# train_fname，test_fname为你需要导出的json标签文件的位置，如存在，请先删除
train_fname = "data_generate/train.json"
test_fname = "data_generate/test.json"
```

## 3、生成骨骼标签
请参考`data_generate/data_process/custom_dataset.py` \
按照：Step1，Step2，Step3，依次运行 \
请使用nohup指令运行Step2，避免远程服务器掉线
```bash
nohup bash tools/dist_run.sh tools/data/custom_2d_skeleton.py 6 --video-list data_generate/custom.list --out data_generate/custom_annos.pkl > nohup.out 2>&1 &
```
最后，生成的pkl文件会存在你指定的位置下 \


## 4、训练行为识别模型
### 编写配置文件
参考`configs/stgcn/stgcn_custom_hrnet/j.py`，创建一份你的配置文件 \
需要修改以下参数：
```python
"""
@layout：骨骼类型，目前设为coco，后续会更改
@num: 数据集中的类别个数
@ann_file：生成的用于行为识别的数据集的路径
@lr: 线性学习率，根据显卡的个数线性增加/减少，8卡~0.1
@work_dir: 指定一个新的，用于存储训练过程文件的目录
"""
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=3, in_channels=256))

ann_file = 'data_generate/custom_hrnet.pkl'

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005, nesterov=True)

work_dir = './work_dirs/stgcn_custom/'
```
### 运行训练程序
```python
bash tools/dist_train.sh configs/stgcn/stgcn_custom_hrnet/j.py 6 --validate --test-last --test-best
```
### 运行测试程序
```python
bash tools/dist_test.sh configs/stgcn/stgcn_custom_hrnet/j.py work_dirs/stgcn_custom/epoch_30.pth 6 --eval top_k_accuracy --out result.pkl
```

## 5、运行测试DEMO