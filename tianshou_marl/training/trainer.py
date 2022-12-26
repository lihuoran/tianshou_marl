from datetime import datetime
from typing import List, Optional

from tianshou.policy import BasePolicy

from tianshou_marl.rollout.collector import Collector


def _naive_log(msg: str) -> None:
    # TODO: replace this function with a comprehensive logger
    print(f"{datetime.now()}\t{msg}")


class BaseTrainer(object):
    def __init__(
        self,
        policies: List[BasePolicy],
        max_epoch: int,
        batch_size: int,
        step_per_epoch: int,
        repeat_per_collect: int,
        step_per_collect: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
        episode_per_test: Optional[int] = None,
        train_collector: Optional[Collector] = None,
        test_collector: Optional[Collector] = None,
    ) -> None:
        self._policies = policies

        self._train_collector = train_collector
        self._test_collector = test_collector

        self._batch_size = batch_size
        self._max_epoch = max_epoch
        self._step_per_epoch = step_per_epoch
        self._repeat_per_collect = repeat_per_collect
        self._step_per_collect = step_per_collect
        self._episode_per_collect = episode_per_collect
        self._episode_per_test = episode_per_test

    def reset(self) -> None:
        pass

    def run(self) -> None:
        for epoch in range(1, self._max_epoch + 1):
            self._train_epoch(epoch)
            if self._test_collector is not None:
                self._test_step(epoch)

    def _train_epoch(self, epoch: int) -> None:
        for policy in self._policies:
            policy.train()

        for step in range(1, self._step_per_epoch + 1):
            self._train_step(epoch, step)

    def _train_step(self, epoch: int, step: int) -> None:
        assert self._train_collector is not None

        self._train_collector.collect(
            n_step=self._step_per_collect,
            n_episode=self._episode_per_collect,
            desc=f"Training Epoch {epoch}/{self._max_epoch} step {step}/{self._step_per_epoch}"
        )
        self._policy_update_fn()

    def _test_step(self, epoch: int) -> None:
        assert self._test_collector is not None
        assert self._episode_per_test is not None

        self._test_collector.restart_env()
        self._test_collector.reset_buffers()

        for policy in self._policies:
            policy.train()

        res = self._test_collector.collect(
            n_episode=self._episode_per_test,
            desc=f"Testing {epoch}/{self._max_epoch}"
        )

        _naive_log(f"Policy rewards: {res['rews'].mean(axis=1)}, average rewards: {res['rews'].mean():.4f}")

    def _policy_update_fn(self) -> None:
        assert self._train_collector is not None

        for policy, buffer in zip(self._policies, self._train_collector.buffers):
            policy.update(
                0,
                buffer,
                batch_size=self._batch_size,
                repeat=self._repeat_per_collect,
            )
