import math


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.last = math.nan
        self.sum = 0
        self.avg = math.nan
        self.cnt = 0

    def reset(self) -> None:
        self.__init__()

    def update(self, value: int | float, num: int = 1) -> None:
        self.last = value
        self.cnt += num
        self.sum += value * num
        self.avg = self.sum / self.cnt


class MinMeter:
    """Computes and stores the minimum and current value."""

    def __init__(self) -> None:
        self.last = math.nan
        self.min = math.inf
        self.cnt = 0

    def reset(self) -> None:
        self.__init__()

    def update(self, value: int | float) -> None:
        self.last = value
        self.cnt += 1
        self.min = min(self.min, value)


class MaxMeter:
    """Computes and stores the maximum and current value."""

    def __init__(self) -> None:
        self.last = math.nan
        self.max = -math.inf
        self.cnt = 0

    def reset(self) -> None:
        self.__init__()

    def update(self, value: int | float) -> None:
        self.last = value
        self.cnt += 1
        self.max = max(self.max, value)
