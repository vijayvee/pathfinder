#!/bin/bash
seeds=(10 56 1234)
gpus=(0 1 3)
for i in {1..3}
do
	python train.py -cfg hgru -nl 3 --epochs 50 -b 256 --lr 1e-4 --rank 0 --world-size 1 \
	       	--gpu ${gpus[$i-1]} //curv_contour_length_9 -wd 0 --in-res 160 \
	       	--expname paper_pf9_hgru_ts12_fsize9_k32 --seed ${seeds[$i-1]} &
done
echo "Training job w 3 random seeds"
