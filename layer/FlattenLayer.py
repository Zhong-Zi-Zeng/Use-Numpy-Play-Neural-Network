import numpy as np



class FlattenLayer:
    def __init__(self):
        pass


    def FP(self, x, **kwargs):
        self.input_shape = x.shape

        self.y = np.reshape(x, (self.input_shape[0], -1))

        return self.y

    def BP(self, delta):
        delta = np.reshape(delta, self.input_shape)

        return delta