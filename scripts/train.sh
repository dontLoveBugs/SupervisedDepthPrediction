#!/bin/bash
mode=$1
n_gpus=$2
flag=$3
path=$4

if [ $mode -eq 0 ]
then
  if [ $n_gpus -gt 1 ]
  then
    echo "running [python -m torch.distributed.launch --nproc_per_node=$n_gpus train.py $flag $path]"
    python -m torch.distributed.launch --nproc_per_node=$n_gpus train.py $flag $path
  else
    echo "running [python train.py $flag $path]"
    python train.py $flag $path
  fi
else
  if [ $n_gpus -gt 1 ]
  then
    echo "running [python -m torch.distributed.launch --nproc_per_node=$n_gpus train_for_submission.py $flag $path]"
    python -m torch.distributed.launch --nproc_per_node=$n_gpus train_for_submission.py $flag $path
  else
    echo "running [python train_for_submission.py $flag $path]"
    python train_for_submission.py $flag $path
  fi
fi