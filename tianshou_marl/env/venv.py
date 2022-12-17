from __future__ import annotations

from typing import Callable, List, Optional, Type

import numpy as np

from .env import Env
from .worker import BaseEnvWorker, DummyEnvWorker
from ..exception import SeedExhausted


class BaseVectorEnv(object):
    def __init__(
        self,
        worker_cls: Type[BaseEnvWorker],
        env_fns: List[Callable[[], Env]],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._workers = [worker_cls(env_fn) for env_fn in env_fns]

    def __len__(self) -> int:
        return len(self._workers)

    def reset_all(self) -> np.ndarray:
        ret = self.reset_workers(np.arange(len(self)))
        self._agent_num = self._workers[0].agent_num
        return ret

    def reset_workers(self, worker_indexes: np.ndarray) -> np.ndarray:
        for idx in worker_indexes:
            try:
                self._workers[idx].reset()
            except SeedExhausted:
                pass  # TODO
        return np.array([self._workers[idx].recv() for idx in worker_indexes], dtype=object)

    def step(self, actions: np.ndarray) -> np.ndarray:
        assert actions.shape == (len(self), self._agent_num), \
            f"Expect action shape: {(len(self), self._agent_num)}, got {actions.shape}"

        for action, worker in zip(actions, self._workers):
            worker.send(action)
        return np.array([worker.recv() for worker in self._workers], dtype=object)

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
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(DummyEnvWorker, env_fns, wait_num, timeout)
