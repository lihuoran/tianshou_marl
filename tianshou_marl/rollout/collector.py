from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tianshou.data import Batch, CachedReplayBuffer, ReplayBuffer, ReplayBufferManager, VectorReplayBuffer, to_numpy

from tianshou_marl.env.venv import BaseVectorEnv
from tianshou_marl.utils.progress import Progress


def _should_terminate(n_step: Optional[int], n_episode: Optional[int], step_count: int, episode_count: int) -> bool:
    return any(
        [
            n_step is not None and step_count >= n_step,
            n_episode is not None and episode_count >= n_episode,
        ],
    )


class Collector(object):
    def __init__(
        self,
        venv: BaseVectorEnv,
        policies: list,
        buffer: Optional[ReplayBuffer] = None,
    ) -> None:
        self._venv = venv
        self._policies = policies
        self.buffers = self._create_buffers(buffer)

        self._init_data()
        self.restart_env()

    @property
    def n_policy(self) -> int:
        return len(self._policies)

    def restart_env(self) -> None:
        self._venv.restart()
        self._collect_step_info(self._venv.reset_all())

    def reset_buffers(self, keep_statistics: bool = False) -> None:
        for buffer in self.buffers:
            buffer.reset(keep_statistics=keep_statistics)

    def _init_data(self) -> None:
        self._data = [
            Batch(
                obs={},
                obs_next={},
                done={},
                terminated={},
                truncated={},
                act={},
                rew={},
                info={},
            )
            for _ in range(self.n_policy)
        ]

    def _collect_step_info(self, step_info: np.ndarray) -> np.ndarray:
        if len(step_info) == 0:
            self._init_data()
            return np.array([])

        pool = [[] for _ in range(self.n_policy)]
        env_done = []
        for observations, done, rewards, info_dict in step_info:  # Iterate each worker
            assert observations.shape[0] == self.n_policy
            for i in range(self.n_policy):
                pool[i].append((observations[i], done, False, done, rewards[i]))
            env_done.append(done)

        batches = [Batch(**dict(zip(["obs_next", "done", "truncated", "terminated", "rew"], zip(*e)))) for e in pool]

        assert all(len(batch) == len(self._venv.alive_env_idx) for batch in batches)
        for new_batch, old_batch in zip(batches, self._data):
            old_batch.update(new_batch)

        return np.array(env_done)

    def _create_buffers(self, buffer: Optional[ReplayBuffer]) -> List[ReplayBuffer]:
        if buffer is None:
            buffer = VectorReplayBuffer(10000, len(self._venv))  # TODO: remove constant
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= len(self._venv)
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= len(self._venv)
        else:
            raise ValueError(f"Unsupported buffer type: {type(buffer)}.")

        buffers = []
        for _ in range(self.n_policy):
            buffers.append(deepcopy(buffer))
        return buffers

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        no_grad: bool = True,
        desc: str = "",
    ) -> Dict[str, Any]:
        assert n_step is not None or n_episode is not None, "Please specify at least one (either n_step or n_episode)."
        assert n_step is None or n_episode is None, "n_step and n_episode cannot be specified together."

        episode_rews = [[] for _ in range(self.n_policy)]

        progress_kwargs = {
            "total": n_step if n_step is not None else n_episode,
            "unit": "step" if n_step is not None else "episode",
            "desc": desc,
        }
        with Progress(**progress_kwargs) as progress:
            while not progress.should_terminate() and len(self._venv.alive_env_idx) > 0:
                for batch in self._data:
                    batch.obs = batch.obs_next

                if no_grad:
                    with torch.no_grad():
                        results = [policy(batch) for batch, policy in zip(self._data, self._policies)]
                else:
                    results = [policy(batch) for batch, policy in zip(self._data, self._policies)]

                for batch, result in zip(self._data, results):
                    batch.update(act=to_numpy(result.act))

                actions = np.stack([batch.act for batch in self._data]).transpose()
                env_done = self._collect_step_info(self._venv.step(actions))

                info_pool = []
                for data, buffer in zip(self._data, self.buffers):
                    # ptr, ep_rew, ep_len, ep_idx
                    info_pool.append(buffer.add(data, buffer_ids=self._venv.alive_env_idx))

                # Update progress
                if n_step is not None:
                    progress.update(len(env_done))

                if env_done.any():
                    if n_episode is not None:
                        progress.update(env_done.sum())

                    idx = np.where(env_done)[0]
                    for i in range(self.n_policy):
                        episode_rews[i].append(info_pool[i][1])

                    terminated_env_idx = self._venv.alive_env_idx[idx]
                    self._venv.reset_workers(terminated_env_idx)
                    self._collect_step_info(self._venv.recv_all())

        try:
            rews = np.array([np.concatenate(e) for e in episode_rews])
        except ValueError:
            rews = np.array(episode_rews)

        return {
            "rews": rews
        }
