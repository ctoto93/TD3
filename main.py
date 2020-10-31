import numpy as np
import pandas as pd
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
import envs

from tqdm import tqdm

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(policy, env_name, seed, n_steps, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	avg_step = 0.
	for _ in range(eval_episodes):
		done = False
		state = np.array([np.radians(180), 0])
		eval_env.state = state
		state = eval_env._get_obs().reshape(1, -1)
		if eval_env.transformer:
			state = eval_env.transformer.transform(state)
			while step in range(100):
				action = policy.select_action(np.array(state))
				state, reward, done, _ = eval_env.step(action)
				avg_reward += reward
				if done:
					break
			avg_step += step

	avg_reward /= eval_episodes
	avg_step /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward, avg_step


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	# Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--policy", default="TD3")
	# OpenAI gym environment name
	parser.add_argument("--env", default="HalfCheetah-v2")
	# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
	# How often (time steps) we evaluate
	parser.add_argument("--eval_freq", default=1000, type=int)
	# Max time steps to run environment
	parser.add_argument("--max_timesteps", default=50000, type=int)
	# 1 episode how many steps in the environment
	parser.add_argument("--n_steps", default=50, type=int)
	# Std of Gaussian exploration noise
	parser.add_argument("--expl_noise", default=0.5)
	parser.add_argument("--buffer_size", default=1e6,
						type=int)      # replay buffer size
	# Batch size for both actor and critic
	parser.add_argument("--batch_size", default=256, type=int)
	# Discount factor
	parser.add_argument("--discount", default=0.95)
	# Target network update rate
	parser.add_argument("--tau", default=0.005, type=float)
	# Noise added to target policy during critic update
	parser.add_argument("--policy_noise", default=0.2)
	# Range to clip target policy noise
	parser.add_argument("--noise_clip", default=0.5)
	# Frequency of delayed policy updates
	parser.add_argument("--policy_freq", default=2, type=int)
	# Save model and optimizer parameters
	parser.add_argument("--save_model", action="store_true")
	# Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--load_model", default="")
	args = parser.parse_args()
	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
			"action_dim": action_dim,
			"max_action": max_action,
			"discount": args.discount,
			"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(
		state_dim, action_dim, max_size=args.buffer_size)

	# Evaluate untrained policy

	evaluations = [eval_policy(
		policy, args.env, args.seed, n_steps=args.n_steps, eval_episodes=1)]
	train_rewards = []

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in tqdm(range(args.max_timesteps)):
		state, done = env.reset(), False
		episode_reward = 0

		for _ in range(args.n_steps):
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

			next_state, reward, done, _ = env.step(action)
			done_bool = 0

			# Store data in replay buffer
			replay_buffer.add(state, action, next_state, reward, done_bool)

			state = next_state
			episode_reward += reward


		train_rewards.append(episode_reward)
		policy.train(replay_buffer, args.batch_size)

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			reward, step = eval_policy(
				policy, args.env, args.seed, args.n_steps, eval_episodes=1)
			df_eval = pd.DataFrame()
			df_eval["episode"] = t + 1
			df_eval["cum_rewards"] = reward
			df_eval["step"] = step
			df_eval.to_pickle(f"./results/df_eval.pkl")
			print(f"episode {t} train rewards: {episode_reward}")
			df_train = pd.DataFrame()
			df_train["episode"] = np.arange(len(train_rewards))
			df_train["train_cum_reward"] = train_rewards
			df_train.to_pickle(f"./results/df_train.pkl")
			if args.save_model:
				policy.save(f"./models/{file_name}")
