import numpy as np
from copy import deepcopy

# ============Stochastic Gradient Decent===========
class SGD:
    def __init__(self, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def gradient_decent(self, layer_list : list):
        for layer in layer_list:
            layer.w -= self.learning_rate * layer.d_w / self.batch_size
            layer.b -= self.learning_rate * layer.d_b / self.batch_size

# ============Momentum===========
class Momentum:
    def __init__(self, learning_rate, batch_size, alpha=0.5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha

    def gradient_decent(self, layer_list: list):
        if not hasattr(self,'last_w'):
            self.last_w = []
            self.last_b = []

            for layer in layer_list:
                self.last_w.append(np.zeros_like(layer.w))
                self.last_b.append(np.zeros_like(layer.b))

        for idx, layer in enumerate(layer_list):
            old_w = deepcopy(layer.w)
            old_b = deepcopy(layer.b)
            layer.w -= (self.learning_rate * layer.d_w + self.alpha * self.last_w[idx]) / self.batch_size
            layer.b -= (self.learning_rate * layer.d_b + self.alpha * self.last_b[idx]) / self.batch_size
            self.last_w[idx] = layer.w - old_w
            self.last_b[idx] = layer.b - old_b
