#!/bin/bash

# Script to reproduce results

for ((i=0;i<3;i+=1))
do
	python main.py \
	--env SparsePendulumRBF-v0 \
	--policy "TD3" \
	--expl_noise 0.5 \
	--policy_noise 0.0 \
	--noise_clip 0.0\
	--tau 1.0 \
	--buffer_size 1000000 \
	--batch_size 256 \
	--start_timesteps 1000 \
	--max_timesteps 50000 \
	--seed $i

done
