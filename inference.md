## **Real-time inference**
### 拉取镜像
```bash
# 基础镜像
docker pull nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
docker pull jiyuchen666/pyskl:latest
# 进入容器
docker run --gpus all --shm-size 16g -t -i --name mypyskl -v /your/path/pyskl:/pyskl jiyuchen666/pyskl /bin/bash
```
### 进入容器后
```bash
conda activate pyskl
cd /pyskl
# --input：指定输入的视频，--out_filename: 输出的结果，--config：模型的配置文件，--checkpoints：权重文件
python demo/demo_skeleton_coco_skl.py --video=workspace/report/putdown.mp4 --out_filename=workspace/report/putdown_result.mp4 --config=work_dirs/stgcn/stgcn_pyskl_factory/coco/j.py --checkpoint=work_dirs/stgcn/stgcn_pyskl_factory/coco/epoch_16.pth
# 批量推理视频文件
cd /pysk
bash demo/inference.sh /path/to/your/videos/dirs /path/to/write/dirs
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