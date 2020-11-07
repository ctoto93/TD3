#!/bin/bash

# Script to reproduce results

for ((i=0;i<1;i+=1))
do
	python main.py \
	--env SparsePendulumRBF-v0 \
	--policy "TD3" \
	--expl_noise 0.5 \
	--policy_noise 0.2 \
	--noise_clip 0.5\
	--tau 0.5 \
	--buffer_size 1000000 \
	--batch_size 256 \
	--start_timesteps 1000 \
	--max_timesteps 5000 \
	--seed $i

	python main.py \
	--env SparsePendulumRBF-v0 \
	--policy "TD3" \
	--expl_noise 0.5 \
	--policy_noise 0.1 \
	--noise_clip 0.2\
	--tau 0.5 \
	--buffer_size 1000000 \
	--batch_size 256 \
	--max_timesteps 5000 \
	--seed $i

	python main.py \
	--env SparsePendulumRBF-v0 \
	--policy "TD3" \
	--expl_noise 0.5 \
	--policy_noise 0.05 \
	--noise_clip 0.1\
	--tau 0.5 \
	--buffer_size 1000000 \
	--batch_size 256 \
	--max_timesteps 5000 \
	--seed $i

done
