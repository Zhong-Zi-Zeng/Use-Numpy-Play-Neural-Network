import numpy as np


class DropoutLayer:
    def __init__(self, drop_ratio):
        self.drop_ratio = drop_ratio

    def FP(self, x, is_train=False):
        self.rand = np.where(np.random.random(x.shape) > self.drop_ratio, 1, 0)

        if not is_train:
            return x * self.rand
        else:
            return x * (1 - self.drop_ratio)

    def BP(self, delta):
        return delta * self.rand

    def __str__(self):
        return 'DropoutLayer'
