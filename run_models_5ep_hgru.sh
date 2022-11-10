#!/bin/bash
seeds=(56 1234)
for i in {1..2}
do
	python train.py -cfg hgru -nl 3 --epochs 5 -b 256 --lr 1e-4 --rank 0 --world-size 1 \
	       	--gpu $1 /mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_9 -wd 0 --in-res 160 \
	       	--expname paper_pf9_hgru_ts12_fsize9_k38 --seed ${seeds[$i-1]}
done
