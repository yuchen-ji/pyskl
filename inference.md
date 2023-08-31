## **Training & Real-time inference**
### 拉取镜像
```bash
# 基础镜像
docker pull nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
docker pull jiyuchen666/pyskl:v0.1
# 进入容器
docker run --gpus all --shm-size 16g -t -i --name mypyskl -v /path/to/your/pyskl:/pyskl jiyuchen666/pyskl:v0.1 /bin/bash
```

### 推理
```bash
# 在容器中
conda activate pyskl
cd /pyskl
# --input：指定输入的视频，--out_filename: 输出的结果，--config：模型的配置文件，--checkpoints：权重文件
python demo/demo_skeleton_coco_skl_25_window.py --video=/path/to/your/videos.mp4 --out_filename=/path/to/your/output.mp4 --config=work_dirs/stgcn/stgcn_pyskl_factory/coco/j.py --checkpoint=work_dirs/stgcn/stgcn_pyskl_factory/coco/epoch_16.pth
# 批量推理视频文件
cd /pysk
bash demo/inference.sh /path/to/your/video_dirs /path/to/write_dirs
```

### 数据处理
```sh
# 将所有数据concat
cd /pyskl
cd workspace
# 运行 process_data 中的 concat_pkl() 
python workspace/process_data.py
```

### 训练
```bash
# 在容器中
cd /pyskl
# bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# ${CONFIG_FILE} 一般采用 configs/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j.py
# [optional arguments]: --validata --test-last --test-best, 需要pkl文件中包含测试集和验证集
bash tools/dist_train.sh configs/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j.py 8 --validate --test-last --test-best
```

### label：
标签定义在`workspace/label_map/factory6.txt`中，从上至下，index从0开始
```txt
assemble
move
polish
put-down
standing
turn-right
```