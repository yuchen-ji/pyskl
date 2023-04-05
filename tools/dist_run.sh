#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

SCRIPT=$1
GPUS=$2

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT $SCRIPT ${@:3}
# Any arguments from the third one are captured by ${@:3}
