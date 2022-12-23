from tianshou_marl.typing import ActionType, StateType


class BaseReward(object):
    def get_reward(self, state: StateType, action: ActionType, done: bool) -> float:
        raise NotImplementedError
