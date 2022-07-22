from tools import im2col, col2im
from ActivationFunction import *
import numpy as np

activation_func_table = {'relu': relu, 'sigmoid': sigmoid, 'softmax': softmax, 'tanh': tanh, 'linear':linear}

class ConvolutionLayer:
    def __init__(self, channel, img_h, img_w, flt_n, flt_h, flt_w, stride, pad, activation='relu', use_bias=True):
        self.params = (channel, img_h, img_w, flt_n, flt_h, flt_w, stride, pad)

        self.output_channel = flt_n  # 輸出的圖片維度，與filter數量相同
        self.output_height = (img_h - flt_h + 2 * pad) // stride + 1
        self.output_width = (img_w - flt_w + 2 * pad) // stride + 1
        self.activation = activation_func_table[activation]
        self.use_bias = use_bias

        self.w = np.random.normal(0, 0.05, (flt_n, channel * flt_h * flt_w))
        self.b = np.random.normal(0, 0.05, (flt_n, 1))

    def FP(self, x):
        batch = x.shape[0]

        channel, img_h, img_w, flt_n, flt_h, flt_w, stride, pad = self.params

        self.metric = im2col(x, flt_h, flt_w, stride, pad)

        self.u = np.dot(self.w, self.metric)

        if self.use_bias:
            self.u += self.b

        self.u = self.u.T.reshape((batch, self.output_height, self.output_width, self.output_channel)).transpose(0, 3, 1, 2)
        self.y = self.activation(self.u)

        return self.y


    def BP(self, delta):
        batch = delta.shape[0]

        channel, img_h, img_w, flt_n, flt_h, flt_w, stride, pad = self.params

        delta = delta * self.activation(self.u, diff=True)
        delta = delta.transpose(1, 0, 2, 3).reshape(flt_n, batch * self.output_width * self.output_height)

        self.d_w = np.dot(delta, self.metric.T)
        self.d_b = np.sum(delta, axis=1, keepdims=True)


        d_x = np.dot(self.w.T, delta)
        d_x = col2im(d_x, (batch, channel ,img_h, img_w), flt_h, flt_w)

        return d_x






