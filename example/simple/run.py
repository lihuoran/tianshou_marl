from tianshou_marl.env.env import Env
from tianshou_marl.env.venv import DummyVectorEnv
from simulator import SimpleSimulator
from tianshou_marl.rollout.collector import Collector

if __name__ == "__main__":
    venv = DummyVectorEnv(
        env_fns=[
            lambda: Env(
                simulator_fn=lambda init_state: SimpleSimulator(init_state=init_state, agent_num=3),
                seed_iterator=[(10, 7, 11), (10, 6, 11)],
            ),
            lambda: Env(
                simulator_fn=lambda init_state: SimpleSimulator(init_state=init_state, agent_num=3),
                seed_iterator=[(10, 3, 9), (10, 4, 9)],
            )
        ]
    )

    collector = Collector(venv=venv)
    collector.collect()
