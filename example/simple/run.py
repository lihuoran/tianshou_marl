import numpy as np
from gym import spaces
from tianshou.data import VectorReplayBuffer
from tianshou.policy import BasePolicy
from torch import nn

from simulator import SimpleSimulator
from tianshou_marl.env.env import Env
from tianshou_marl.env.venv import DummyVectorEnv
from tianshou_marl.policy.ppo import PPO
from tianshou_marl.reward import BaseReward
from tianshou_marl.rollout.collector import Collector

N = 10
NUM_POLICY = 3


def get_network() -> nn.Module:
    network = nn.Sequential(
        nn.Linear(N, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
    )
    network.output_dim = 128
    return network


def get_policy() -> BasePolicy:
    return PPO(
        network=get_network(),
        obs_space=spaces.Box(0.0, 1.0, shape=(N,)),
        action_space=spaces.Discrete(N),
        lr=1e-4,
    )


class SimpleReward(BaseReward):
    def get_reward(self, state: np.ndarray, action: int, done: bool) -> float:
        return 1.0 if state[action] == 1.0 else 0.0


if __name__ == "__main__":
    venv = DummyVectorEnv(
        env_fns=[
            lambda: Env(
                simulator_fn=lambda init_state: SimpleSimulator(init_state=init_state, agent_num=3),
                reward=SimpleReward(),
                seed_iterator=[(N, 7, 11), (N, 6, 11)],
            ),
            lambda: Env(
                simulator_fn=lambda init_state: SimpleSimulator(init_state=init_state, agent_num=3),
                reward=SimpleReward(),
                seed_iterator=[(N, 3, 9), (N, 4, 9)],
            )
        ],
        agent_num=NUM_POLICY,
    )

    collector = Collector(
        venv=venv,
        policies=[get_policy() for _ in range(NUM_POLICY)],
        buffer=VectorReplayBuffer(50, 2),
    )
    collector.collect(n_step=100, no_grad=False)
