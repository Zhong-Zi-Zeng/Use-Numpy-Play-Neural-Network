from ActivationFunction import *
import numpy as np

activation_func_table = {'relu': relu, 'sigmoid': sigmoid, 'softmax': softmax, 'tanh': tanh, 'linear': linear}


class BaseLayer:
    def __init__(self, output_shape, input_shape=None, activation='relu', use_bias=True):
        self.activation = activation_func_table[activation]
        self.use_bias = use_bias

        self.input_shape = input_shape
        self.output_shape = output_shape

    def set_weight_bias(self):
        self.w = np.random.normal(0, 0.05, (self.input_shape, self.output_shape))
        self.b = np.random.normal(0, 0.05, (1, self.output_shape))

    def FP(self, x, **kwargs):
        self.x = x
        self.u = np.dot(self.x, self.w)

        if self.use_bias:
            self.u += self.b

        self.y = self.activation(self.u)

        return self.y

    def BP(self, delta):
        delta = delta * self.activation(self.u, diff=True)

        self.d_w = np.dot(self.x.T, delta)
        self.d_b = np.sum(delta, axis=0)

        d_x = np.dot(delta, self.w.T)

        return d_x
