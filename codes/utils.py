class AverageMeter:
    """Computes and stores the average value."""
    def __init__(self):
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n

    def avg(self):
        return self.sum / self.cnt
