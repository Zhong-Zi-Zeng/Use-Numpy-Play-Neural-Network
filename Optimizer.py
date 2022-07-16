import numpy as np
from abc import abstractmethod, ABC
from copy import deepcopy

class Optimizer(ABC):
    def __init__(self, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    @abstractmethod
    def gradient_decent(self, layer_list: list):
        pass

# ============Stochastic Gradient Decent===========
class SGD(Optimizer):
    def __init__(self, learning_rate, batch_size):
        super().__init__(learning_rate, batch_size)

    def gradient_decent(self, layer_list: list):
        for layer in layer_list:
            if layer.__str__() != 'DropoutLayer':
                layer.w -= self.learning_rate * layer.d_w / self.batch_size
                layer.b -= self.learning_rate * layer.d_b / self.batch_size


# ============Momentum===========
class Momentum(Optimizer):
    def __init__(self, learning_rate, batch_size, alpha=0.9):
        super().__init__(learning_rate, batch_size)
        self.alpha = alpha

    def gradient_decent(self, layer_list: list):
        if not hasattr(self, 'last_w'):
            self.last_w = []
            self.last_b = []

            for layer in layer_list:
                if layer.__str__() != 'DropoutLayer':
                    self.last_w.append(np.zeros_like(layer.w))
                    self.last_b.append(np.zeros_like(layer.b))

        for idx, layer in enumerate(layer_list):
            if layer.__str__() != 'DropoutLayer':
                self.last_w[idx] = (self.alpha * self.last_w[idx] - self.learning_rate * layer.d_w) / self.batch_size
                self.last_b[idx] = (self.alpha * self.last_b[idx] - self.learning_rate * layer.d_b) / self.batch_size
                layer.w += self.last_w[idx]
                layer.b += self.last_b[idx]


# ============AdaGrad===========
class AdaGrad(Optimizer):
    def __init__(self, learning_rate, batch_size):
        super().__init__(learning_rate, batch_size)

    def gradient_decent(self, layer_list: list):
        if not hasattr(self, 'h_w'):
            self.h_w = []
            self.h_b = []
            for layer in layer_list :
                if layer.__str__() != 'DropoutLayer':
                    self.h_w.append(np.zeros_like(layer.w) + 1e-8)
                    self.h_b.append(np.zeros_like(layer.b) + 1e-8)
                else:
                    self.h_w.append(None)
                    self.h_b.append(None)

        for idx, layer in enumerate(layer_list):
            if layer.__str__() != 'DropoutLayer':
                self.h_w[idx] += layer.d_w ** 2
                layer.w -= self.learning_rate / np.sqrt(self.h_w[idx]) * layer.d_w

                self.h_b[idx] += layer.d_b ** 2
                layer.b -= self.learning_rate / np.sqrt(self.h_b[idx]) * layer.d_b
