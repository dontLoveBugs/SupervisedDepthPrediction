#!/bin/bash
n_gpus=$1
flag=$2
path=$3

if [ $n_gpus -gt 1 ]
  then
    echo "running [python -m torch.distributed.launch --nproc_per_node=$n_gpus train.py $flag $path]"
    python -m torch.distributed.launch --nproc_per_node=$n_gpus train.py $flag $path
  else
    echo "running [python train.py $flag $path]"
    python train.py $flag $path
 fi