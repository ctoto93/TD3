#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do
	python main.py \
	--env ContinuousGridWorld-v0 \
	--policy "TD3" \
	--expl_noise 0.01 \
	--policy_noise 0.02 \
	--noise_clip 0.05\
	--tau 0.5 \
	--buffer_size 1000000 \
	--batch_size 256 \
	--max_timesteps 5000 \
	--seed $i

	python main.py \
	--env ContinuousGridWorld-v0 \
	--policy "TD3" \
	--expl_noise 0.01 \
	--policy_noise 0.005 \
	--noise_clip 0.01\
	--tau 0.5 \
	--buffer_size 1000000 \
	--batch_size 256 \
	--max_timesteps 5000 \
	--seed $i

done
