from typing import Generic

import numpy as np

from tianshou_marl.typing import ActionType, PolicyActionType

from .base import Interpreter


class ActionInterpreter(Generic[ActionType, PolicyActionType], Interpreter):
    def __init__(self) -> None:
        pass

    def __call__(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError
