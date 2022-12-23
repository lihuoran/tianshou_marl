from __future__ import annotations

from typing import Callable, List, Optional, Type

import numpy as np

from ..exception import SeedExhausted
from .env import Env
from .worker import BaseEnvWorker, DummyEnvWorker


class BaseVectorEnv(object):
    def __init__(
        self,
        worker_cls: Type[BaseEnvWorker],
        env_fns: List[Callable[[], Env]],
        agent_num: int,
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._workers = [worker_cls(env_fn) for env_fn in env_fns]
        self._alive_flag = [True] * len(self._workers)
        self._agent_num = agent_num

    def __len__(self) -> int:
        return len(self._workers)

    @property
    def alive_env_idx(self) -> np.ndarray:
        return np.where(np.array(self._alive_flag))[0]

    def recv_all(self) -> np.ndarray:
        return np.array([self._workers[i].recv() for i in self.alive_env_idx], dtype=object)

    def reset_all(self) -> np.ndarray:
        return self.reset_workers(np.arange(len(self)))

    def reset_workers(self, worker_indexes: np.ndarray) -> np.ndarray:
        for idx in worker_indexes:
            if not self._alive_flag[idx]:
                continue
            try:
                self._workers[idx].reset()
            except SeedExhausted:
                self._alive_flag[idx] = False

        return np.array([self._workers[idx].recv() for idx in worker_indexes if self._alive_flag[idx]], dtype=object)

    def step(self, actions: np.ndarray) -> np.ndarray:
        assert actions.shape == (
            len(self.alive_env_idx),
            self._agent_num,
        ), f"Expect action shape: {(len(self), self._agent_num)}, got {actions.shape}"

        for i, (action, worker) in enumerate(zip(actions, self._workers)):
            if self._alive_flag[i]:
                worker.send(action)
        return np.array([worker.recv() for i, worker in enumerate(self._workers) if self._alive_flag[i]], dtype=object)

    def seed(self, seed: int | List[int] | None) -> None:
        if not isinstance(seed, list):
            seed = [seed] * len(self)
        assert isinstance(seed, list) and len(seed) == len(self)

        for w, s in zip(self._workers, seed):
            w.seed(s)


class DummyVectorEnv(BaseVectorEnv):
    def __init__(
        self,
        env_fns: List[Callable[[], Env]],
        agent_num: int,
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(DummyEnvWorker, env_fns, agent_num, wait_num, timeout)
