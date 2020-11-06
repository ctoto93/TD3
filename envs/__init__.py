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

register(
    id='ContinuousGridWorld-v0',
    entry_point='rlboxes.envs:ContinuousGridWorldEnv',
    kwargs={
         'width': 1,
         'height': 1,
         'max_step': 0.1,
         'goal_x': 0.45,
         'goal_y': 0.45,
         'goal_side': 0.1,
         'radial_basis_fn': True,
         'x_dim': 10,
         'y_dim': 10,
         'cumulative': False
    }

)
