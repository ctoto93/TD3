from gym.envs.registration import register
from envs.sparse_pendulum import REWARD_MODE_BINARY

register(
    id='SparsePendulumRBF-v0',
    entry_point='envs.sparse_pendulum:SparsePendulumRBFEnv',
    kwargs={
         'max_speed': 8.0,
         'max_torque': 2.0,
         'reward_angle_limit': 2.0,
         'reward_speed_limit': 1.0,
         'balance_counter': 5,
         'reward_mode': REWARD_MODE_BINARY,
         'cumulative': True,
         'n_rbf': None
    }

)
