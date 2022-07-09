import tensorflow as tf
import numpy as np
from Model import Sequential
from BaseLayer import BaseLayer
from OutputLayer import OutputLayer
import matplotlib.pyplot as plt
import copy


def normalization(x):
    x = np.float64(x)
    x /= 255.

    return x


def label_to_hotcode(label):
    dim = np.max(label) + 1
    hotcode_label = np.zeros((len(label), dim))

    for idx, y in enumerate(label):
        hotcode_label[idx][y] = 1

    return hotcode_label


# ============載入數據集============
(train_img, train_label), (test_img, test_label) = tf.keras.datasets.mnist.load_data()

# ============資料預處理============
test_x = normalization(test_img)
test_y = label_to_hotcode(test_label)

train_x = normalization(train_img)
train_y = label_to_hotcode(train_label)

test_x = np.reshape(test_x, (10000, 784))
train_x = np.reshape(train_x, (60000, 784))

# ============超參數設置============
EPOCH = 2
BATCHSIZE = 32
LR = 0.1

# ============建置模型============
model = Sequential(epoch=EPOCH, batch_size=BATCHSIZE, learning_rate=LR)
model.add(BaseLayer(784, 64, activation='sigmoid'))
model.add(BaseLayer(64, 128, activation='sigmoid'))
model.add(OutputLayer(128, 10, activation='softmax',loss='cross_entropy'))

model.fit(train_x, train_y)
model.evaluate(test_x, test_y, batch_size=16)