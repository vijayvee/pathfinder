#!/bin/bash
seeds=(10 56 1234)
for i in {1..3}
do
	python train.py -cfg dalernn-t-12 -nl 3 --epochs 5 -b 256 --lr 1e-4 --rank 0 --world-size 1 \
	       	--gpu $1 /mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_9 --timesteps 8 -wd 0 --in-res 160 \
	       	--expname pf9_dalernn_ts12_fsize9_k32 --seed ${seeds[$i-1]}
done
