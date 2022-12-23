from typing import Any

from tqdm import tqdm


class Progress(tqdm):
    def __init__(self, total: int, **kwargs: Any) -> None:
        super().__init__(total=total, **kwargs)
        self._total = total
        self._count = 0

    def update(self, n: int = 1) -> None:
        super().update(n)
        self._count += n

    def should_terminate(self) -> bool:
        return self.remain_iters == 0

    @property
    def remain_iters(self) -> int:
        return max(0, self._total - self._count)
