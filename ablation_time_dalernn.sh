#!/bin/bash

for i in 10 8 6 4 2
do
        python train.py -cfg dalernn-t-12 -nl 3 --epochs 5 -b 256 --lr 1e-4 --rank 0 --world-size 1 --gpu 1 --timesteps ${i} \
         $1 -wd 0 --in-res 160 --expname timestep_ablation_pf14_dalernn_ts12_fsize9 \
          --seed 56
done
