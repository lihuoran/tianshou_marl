from __future__ import annotations

from typing import Tuple, cast

import numpy as np

from tianshou_marl.env.env import Simulator


class SimpleSimulator(Simulator[Tuple[int, int, int], np.ndarray, int]):
    def __init__(self, init_state: Tuple[int, int, int], agent_num: int) -> None:
        super().__init__(init_state=init_state, agent_num=agent_num)

        self.n, self.k, self.step_limit = cast(Tuple[int, int, int], self._init_state)
        self.step_count = 0

        assert 0 <= self.k < self.n

    def _get_states(self) -> np.ndarray:
        onehot = np.zeros(self.n, dtype=float)
        onehot[self.k] = 1.0
        return np.repeat(np.expand_dims(onehot, 0), self.agent_num, axis=0)

    def initialize(self) -> Tuple[np.ndarray, bool]:
        self.step_count = 0
        return self._get_states(), self.step_count >= self.step_limit

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, bool]:
        self.step_count += 1
        return self._get_states(), self.step_count >= self.step_limit


if __name__ == "__main__":
    simulator = SimpleSimulator(init_state=(10, 7, 10), agent_num=3)
