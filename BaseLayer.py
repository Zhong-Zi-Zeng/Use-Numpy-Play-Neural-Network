from ActivationFunction import *
import numpy as np

activation_func_table = {'relu': relu, 'sigmoid': sigmoid, 'softmax': softmax}


class BaseLayer:
    def __init__(self, input_shape, output_shape, activation='relu', use_bias=True):
        self.activation = activation_func_table [activation]
        self.use_bias = use_bias

        self.w = np.random.normal(0, 0.05, (input_shape, output_shape))
        self.b = np.random.normal(0, 0.05, (1, output_shape))

    def FP(self, x):
        self.x = x
        self.u = np.dot(self.x, self.w)

        if self.use_bias:
            self.u += self.b

        self.y = self.activation(self.u)

        return self.y

    def BP(self, delta, next_weight):
        delta = np.dot(delta, next_weight.T) * self.activation(self.u, diff=True)

        self.d_w = np.dot(self.x.T, delta)
        self.d_b = np.sum(delta, axis=0)

        return delta
