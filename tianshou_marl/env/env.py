from __future__ import annotations

from typing import Any, Callable, Generator, Generic, Iterable, Tuple

import numpy as np

from tianshou_marl.exception import SeedExhausted
from tianshou_marl.interpreter.action_interpreter import ActionInterpreter
from tianshou_marl.interpreter.state_interpreter import StateInterpreter
from tianshou_marl.reward import BaseReward
from tianshou_marl.typing import ActionType, InitialStateType, ObservationType, PolicyActionType, StateType


def _convert_to_generator(it: Generator | Iterable) -> Generator:
    if isinstance(it, Generator):
        return it
    elif isinstance(it, Iterable):
        return (e for e in it)
    else:
        raise ValueError(f"Unsupported generator type: {type(it)}")


def _create_agents(agents: list | None, agent_num: int | None) -> list:
    assert agents is not None or agent_num is not None, f"`agents` and `agent_num` should provide at least one."

    if agents is None:
        assert agent_num is not None and agent_num > 0, f"Invalid agent number {agent_num}"
        return list(range(agent_num))
    else:
        assert agent_num == len(
            agents,
        ), f"The length of agents is inconsistent with agent_num ({len(agents)} v.s. {agent_num})."
        return agents


class Env(Generic[InitialStateType, StateType, ActionType, ObservationType, PolicyActionType]):
    def __init__(
        self,
        simulator_fn: Callable[[InitialStateType], Simulator],
        seed_iterator: Generator | Iterable,
        reward: BaseReward,
        state_interpreter: StateInterpreter = None,
        action_interpreter: ActionInterpreter = None,
    ) -> None:
        self._simulator_fn = simulator_fn
        self._reward = reward
        self._state_interpreter = state_interpreter
        self._action_interpreter = action_interpreter
        self._seed_iterator = _convert_to_generator(seed_iterator)

        self._simulator: Simulator | None = None
        self._last_states: np.ndarray[StateType] | None = None
        self._last_done: bool = False

    @property
    def simulator(self) -> Simulator:
        assert self._simulator is not None, "Simulator has not been created yet."
        return self._simulator

    @property
    def agent_num(self) -> int:
        return self.simulator.agent_num

    def _reset(self) -> Tuple[np.ndarray[ObservationType], bool, np.ndarray, dict]:
        assert self._seed_iterator is not None

        try:
            while True:
                seed = next(self._seed_iterator)
                self._simulator = self._simulator_fn(seed)
                states, done = self.simulator.initialize()
                observations = self._state_interpreter(states) if self._state_interpreter is not None else states
                self._last_states = states
                self._last_done = done
                if not done:
                    break
            return observations, done, np.zeros(self.agent_num), {}
        except StopIteration:
            self._seed_iterator = None
            raise SeedExhausted

    def step(
        self,
        policy_actions: np.ndarray[PolicyActionType] | None,
    ) -> Tuple[np.ndarray[ObservationType], bool, np.ndarray, dict]:
        assert self._seed_iterator is not None, "TODO"

        if policy_actions is None:
            return self._reset()
        else:
            actions = (
                self._action_interpreter(policy_actions) if self._action_interpreter is not None else policy_actions
            )
            rewards = np.array(
                [
                    self._reward.get_reward(state, action, self._last_done)
                    for state, action in zip(self._last_states, actions)
                ],
            )
            states, done = self.simulator.step_wrapper(actions)
            observations = states if self._state_interpreter is None else self._state_interpreter(states)
            self._last_states = states
            self._last_done = done
            return observations, done, rewards, {}


class Simulator(Generic[InitialStateType, StateType, ActionType]):
    def __init__(
        self,
        init_state: InitialStateType | None = None,
        agents: list | None = None,
        agent_num: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._init_state = init_state
        self.agents = _create_agents(agents, agent_num)

    @property
    def agent_num(self) -> int:
        return len(self.agents)

    def initialize(self) -> Tuple[np.ndarray, bool]:
        raise NotImplementedError

    def step_wrapper(self, actions: np.ndarray) -> Tuple[np.ndarray, bool]:
        assert actions.shape == (self.agent_num,)
        states, done = self.step(actions)
        assert len(states.shape) == 2 and states.shape[0] == self.agent_num
        return states, done

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, bool]:
        raise NotImplementedError
