from typing import Tuple, TypeVar

import numpy as np

InitialStateType = TypeVar("InitialStateType")
StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")
ObservationType = TypeVar("ObservationType")
PolicyActionType = TypeVar("PolicyActionType")

EnvStepOutput = Tuple[np.ndarray, bool, np.ndarray, dict]
