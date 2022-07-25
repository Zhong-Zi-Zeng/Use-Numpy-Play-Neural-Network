import numpy as np


class FlattenLayer:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def set_weight_bias(self):
        self.output_shape = 1
        for shape in self.input_shape:
            self.output_shape *= shape

    def FP(self, x, **kwargs):
        self.input_shape = x.shape

        self.y = np.reshape(x, (self.input_shape[0], -1))
        return self.y

    def BP(self, delta):
        delta = np.reshape(delta, self.input_shape)

        return delta