from typing import Generic

import numpy as np

from tianshou_marl.typing import ObservationType, StateType

from .base import Interpreter


class StateInterpreter(Generic[StateType, ObservationType], Interpreter):
    def __init__(self) -> None:
        pass

    def __call__(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError
