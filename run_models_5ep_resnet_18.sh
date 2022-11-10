#!/bin/bash
seeds=(10 56 1234)
for i in {1..3}
do
	python train.py -cfg resnet_thin -nl 18 --epochs 5 -b 256 --lr 1e-4 --rank 0 --world-size 1 \
	       	--gpu $1 /mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_9 -wd 0 --in-res 160 \
	       	--expname paper_pf9_resnet18_thin_ts12_fsize9 --seed ${seeds[$i-1]}
done

for i in {1..3}
do
        python train.py -cfg resnet_thin -nl 10 --epochs 5 -b 256 --lr 1e-4 --rank 0 --world-size 1 \
                --gpu $1 /mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_9 -wd 0 --in-res 160 \
                --expname paper_pf9_resnet10_thin_ts12_fsize9 --seed ${seeds[$i-1]}
done
