from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from envs.transformer import RadialBasisTransformer
from gym import spaces
import numpy as np

REWARD_MODE_BINARY = 'binary' # return 1 when pendulum within allowed speed and angle limit over balance_counter time steps
REWARD_MODE_SPARSE = 'sparse' # return continuous speed reward only within speed and range limit

class SparsePendulumRBFEnv(PendulumEnv):
    balance_counter = 0

    def __init__(self, max_speed=8.0, max_torque=2.0, reward_angle_limit=2.0,
                reward_speed_limit=1.0, balance_counter=5, reward_mode=REWARD_MODE_BINARY,
                cumulative=True, n_rbf=[5,5,17]):
        super().__init__()
        self.max_speed= max_speed
        self.max_torque= max_torque
        self.reward_speed_limit = reward_speed_limit
        self.reward_angle_limit = reward_angle_limit
        self.balance_counter = balance_counter
        self.reward_mode = reward_mode
        self.transformer = None
        self.cumulative = cumulative

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        if n_rbf:
            self.original_observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
            self.transformer = RadialBasisTransformer(self.original_observation_space, n_rbf, 'world_models/sparse_pendulum_rbf/inverse_rbf_5_5_17.h5')
            high = np.array([np.inf] * self.transformer.rbf_dim, dtype=np.float32)
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reward_binary(self, th, thdot, u):
        angle = np.degrees(angle_normalize(th))
        done = False
        reward = 0

        if not self.check_angle_speed_limit(angle, thdot):
            SparsePendulumRBFEnv.balance_counter = 0
            return reward, done

        SparsePendulumRBFEnv.balance_counter += 1

        if SparsePendulumRBFEnv.balance_counter >= self.balance_counter:
            reward = 1

        if not self.cumulative:
            done = True

        return reward, done

    def reward_sparse(self, th, thdot, u):
        angle = np.degrees(angle_normalize(th))
        done = False
        reward = 0

        if self.check_angle_speed_limit(angle, thdot):
            reward = self.max_speed - (np.absolute(thdot) / 6.0)

        return reward, done

    def check_angle(self, angle):
        return (angle >= -self.reward_angle_limit) and (angle <= self.reward_angle_limit)

    def check_speed(self, thdot):
        return (thdot >= -self.reward_speed_limit) and (thdot <= self.reward_speed_limit)

    def check_angle_speed_limit(self, angle, thdot):
        return self.check_angle(angle) and self.check_speed(thdot)

    def reset(self):
        obs = super().reset()
        if self.transformer:
            obs = self.transformer.transform(obs.reshape(1,-1))[0]
        return obs

    def step(self, u):
        th, thdot = self.state # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        done = False
        reward = 0

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering

        if self.reward_mode == REWARD_MODE_BINARY:
            reward, done = self.reward_binary(th, thdot, u)
        elif self.reward_mode == REWARD_MODE_SPARSE:
            reward, done = self.reward_sparse(th, thdot, u)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        obs = self._get_obs()
        if self.transformer:
            obs = self.transformer.transform(obs.reshape(1,-1))[0]

        return obs, reward, done, {}
