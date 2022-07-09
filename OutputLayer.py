from LossFunction import *
from ActivationFunction import *
import numpy as np

loss_func_table = {'MSE': MSE, 'cross_entropy': cross_entropy}
activation_func_table = {'relu': relu, 'sigmoid': sigmoid, 'softmax': softmax}


class OutputLayer:
    def __init__(self, input_shape, output_shape, loss, activation='relu', use_bias=True):
        self.activation = activation_func_table[activation]
        self.loss_func = loss_func_table[loss]
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

    def BP(self, label):
        delta = self.loss_func(self.y, label, diff=True) * self.activation(self.u, label, self.y, diff=True)

        self.d_w = np.dot(self.x.T, delta)
        self.d_b = np.sum(delta, axis=0)

        return delta

    def get_loss(self, pre, label):
        return self.loss_func(pre, label)
