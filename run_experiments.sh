#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do

	python main.py \
	--env ContinuousGridWorld-v0 \
	--policy "TD3" \
	--expl_noise 0.01 \
	--policy_noise 0.0 \
	--noise_clip 0.0\
	--tau 0.005 \
	--buffer_size 1000000 \
	--batch_size 256 \
	--start_timesteps 1000 \
	--max_timesteps 20000 \
	--seed $i

	python main.py \
	--env ContinuousGridWorld-v0 \
	--policy "TD3" \
	--expl_noise 0.01 \
	--policy_noise 0.0 \
	--noise_clip 0.0\
	--tau 0.001 \
	--buffer_size 1000000 \
	--batch_size 256 \
	--start_timesteps 1000 \
	--max_timesteps 20000 \
	--seed $i

done
