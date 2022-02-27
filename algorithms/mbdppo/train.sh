#!/bin/bash

# chmod 777 train.sh
# 网络端口号，0-1000系统特殊端口，不要用
#   1000以上，65536以下

# CUDA_VISIBLE_DEVICES为你的显卡组中使用显卡编号
# nproc_per_node数量与你使用的显卡数量相同
# python3修改对自己机器python运行命令
# 其他参数无需修改
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 OMP_NUM_THREADS=2 python -m \
    torch.distributed.launch --nproc_per_node=6 --master_port=55679 \
    train_ddp.py