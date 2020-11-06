#!/bin/bash

# Script to reproduce results

for ((i=0;i<3;i+=1))
do
	python main.py \
	--env ContinuousGridWorld-v0 \
	--policy "TD3" \
	--buffer_size 1000000 \
	--batch_size 256 \
	--max_timesteps 20000 \
	--n_steps 15 \
	--expl_noise 0.01 \
	--policy_noise 0.002 \
	--noise_clip 0.05 \
	--seed $i
done
