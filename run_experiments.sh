#!/bin/bash

# Script to reproduce results

for ((i=0;i<3;i+=1))
do
	python main.py \
	--env SparsePendulumRBF-v0 \
	--policy "TD3" \
	--buffer_size 1000000 \
	--batch_size 256 \
	--max_timesteps 20000 \
	--seed $i
done
