#!/bin/bash
seeds=(10 56 1234)
for i in {1..3}
do
	python train.py -cfg hgru -nl 3 --epochs 5 -b 256 --lr 1e-4 --rank 0 --world-size 1 \
	       	--gpu $1 /home/vveeraba/src/pathfinder_full/curv_contour_length_14 -wd 0 --in-res 160 \
	       	--expname pf9_hgru_ts12_fsize9_k38 --seed ${seeds[$i-1]}
done
