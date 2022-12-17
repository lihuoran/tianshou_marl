from tianshou_marl.env.venv import BaseVectorEnv


class Collector(object):
    def __init__(
        self,
        venv: BaseVectorEnv,
    ) -> None:
        self._venv = venv
        print(self._venv.reset_all())

    def collect(
        self,
    ) -> None:
        pass
