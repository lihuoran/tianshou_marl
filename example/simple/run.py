from typing import Generator

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
from tianshou_marl.training.trainer import BaseTrainer

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


def get_simulator(init_state: tuple) -> SimpleSimulator:
    return SimpleSimulator(init_state=init_state, agent_num=NUM_POLICY)


def get_venv() -> DummyVectorEnv:
    def _seed_generator() -> Generator:
        """Infinite seed generator"""
        i = 0
        while True:
            yield N, i, 10
            i = (i + 1) % N

    return DummyVectorEnv(
        env_fns=[
            lambda: Env(
                simulator_fn=get_simulator,
                reward=SimpleReward(),
                seed_iterator=_seed_generator(),
            )
            for _ in range(2)
        ],
        agent_num=NUM_POLICY,
    )


def get_collector() -> Collector:
    return Collector(
        venv=get_venv(),
        policies=policies,
        buffer=VectorReplayBuffer(total_size=10000, buffer_num=2),
    )


class SimpleReward(BaseReward):
    def get_reward(self, state: np.ndarray, action: int, done: bool) -> float:
        return 1.0 if state[action] == 1.0 else 0.0


if __name__ == "__main__":
    policies = [get_policy() for _ in range(NUM_POLICY)]

    training_settings = {
        "max_epoch": 20,
        "batch_size": 128,
        "repeat_per_collect": 8,
        "step_per_epoch": 3,
        "episode_per_collect": 1000,
        "episode_per_test": 1000,
    }
    trainer = BaseTrainer(
        policies=policies,
        train_collector=get_collector(),
        test_collector=get_collector(),
        **training_settings,
    )
    trainer.run()
