#!/bin/bash
echo "running"

sudo docker run -itd -e NVIDIA_VISIBLE_DEVICES=0 --gpus all --name trt --runtime=nvidia \
        --shm-size 16G --ulimit memlock=-1 --ulimit stack=67108864  \
        -v /home/2023ztl/Desktop:/work nvcr.io/nvidia/pytorch:24.06-py3 /bin/bash