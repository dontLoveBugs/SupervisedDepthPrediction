# SupervisedDepthPrediction
This is a distributed training framework for supervised depth prediction based on Pytorch 1.0 (Pytorch version >= 1.2 is best). Now it provide the implementation of DORN(state of the art in KITTI depth prediction benchmark), and you can implement your model in your customed dataset with a little modification.

# Highlights
- **Distributed & Single GPU** Flexible selection between distributed training with multi gpus and a single gpu.
- **Flexible Visulization Implementation** You can implement your visulizers for network comprehensive analysis. 
- **Suport Various Optimizers and Learning-rate Policy** Provide all the optimizers and learning-rate schedulers in pytorch. And support poly lr_scheduler and warmup, which are widely used in segmentation and detection.
- **Support Grad Clip** Provide grad clip to avoid gradient exploding.
- **Mixed Precision Training** Support mixed precision training with NVIDIA apex lib.
- **Sync BN** Support Sync BN when training with multi gps.
- **Break-point Restoration** Support continue to train from a break-point.





# Installation


# Dataset


# Training


# Testing


# Results
