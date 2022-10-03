#!/bin/bash
seeds=(10 56 1234)
for i in {1..3}
do
	python train.py -cfg resnet -nl 18 --epochs 5 -b 256 --lr 1e-4 --rank 0 --world-size 1 \
	       	--gpu $1 $2 -wd 0 --in-res 160 \
	       	--expname pf14_resnet_full --seed ${seeds[$i-1]}
done
