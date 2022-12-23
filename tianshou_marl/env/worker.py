from __future__ import annotations

from typing import Callable

import numpy as np

from ..typing import EnvStepOutput
from .env import Env


class BaseEnvWorker(object):
    def __init__(self, env_fn: Callable[[], Env]) -> None:
        self._env_fn = env_fn
        self._env = self._env_fn()
        self._last_result: EnvStepOutput | None = None

    def seed(self, seed: int | None) -> None:
        raise NotImplementedError

    def send(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def recv(self) -> EnvStepOutput:
        assert self._last_result is not None, "You must wait `send()` to finish before calling `recv()`."

        return self._last_result

    def reset(self) -> None:
        raise NotImplementedError

    @property
    def agent_num(self) -> int:
        return self._env.agent_num


class DummyEnvWorker(BaseEnvWorker):
    def __init__(self, env_fn: Callable[[], Env]) -> None:
        super().__init__(env_fn)

    def seed(self, seed: int | None) -> None:
        pass

    def send(self, action: np.ndarray) -> None:
        self._last_result = self._env.step(action)

    def reset(self) -> None:
        self._last_result = self._env.step(None)
