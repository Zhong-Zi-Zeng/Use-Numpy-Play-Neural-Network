from tools import im2col, col2im
import numpy as np


class MaxpoolingLayer:
    def __init__(self, pool, input_shape=None):
        self.params = (pool)
        self.input_shape = input_shape
        self.output_shape = None

    def set_weight_bias(self):
        channel, img_h, img_w, = self.input_shape
        pool = self.params

        self.output_channel = channel
        self.output_height = img_h // pool if img_h % pool == 0 else img_h // pool + 1
        self.output_width = img_w // pool if img_w % pool == 0 else img_w // pool + 1
        self.output_shape = (self.output_channel, self.output_height, self.output_width)

    def FP(self, x, **kwargs):
        batch = x.shape[0]
        pool = self.params

        x = im2col(x, pool, pool, stride=pool)
        # print(x.shape)
        x = x.T.reshape((batch * self.output_height * self.output_width * self.output_channel, pool ** 2))

        self.y = np.max(x, axis=1)
        self.y = self.y.reshape((batch, self.output_height, self.output_width, self.output_channel)).transpose(0, 3, 2, 1)
        self.max_idx = np.argmax(x, axis=1)

        return self.y

    def BP(self, delta):
        batch = delta.shape[0]

        channel, img_h, img_w = self.input_shape
        pool = self.params

        delta = delta.transpose(0, 2, 3, 1)
        delta = delta.reshape(-1)
        row_value = batch * self.output_height * self.output_width * self.output_channel

        img = np.zeros((row_value, pool ** 2))
        img[np.arange(row_value), self.max_idx] = delta
        img = img.reshape((batch * self.output_height * self.output_width, channel * pool * pool)).T
        d_x = col2im(img, (batch, channel, img_h, img_w), pool, pool, stride=pool)

        return d_x




