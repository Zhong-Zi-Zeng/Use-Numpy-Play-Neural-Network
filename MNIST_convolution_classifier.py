import tensorflow as tf
import numpy as np
from Model import Sequential
from layer import *
import matplotlib.pyplot as plt
import cv2


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

train_x = train_x.reshape(60000, 1, 28, 28)
test_x = test_x.reshape(10000, 1, 28, 28)

# ============超參數設置============
EPOCH = 10
BATCHSIZE = 32
LR = 0.1

# ============建置模型============
model = Sequential(epoch=EPOCH, batch_size=BATCHSIZE, learning_rate=LR, optimizer='SGD')
model.add(ConvolutionLayer(channel=1, img_h=28, img_w=28, flt_n=2, flt_h=3, flt_w=3))
model.add(MaxpoolingLayer(channel=2, img_h=26, img_w=26, pool=2))
model.add(FlattenLayer())
model.add(BaseLayer(338, 64, activation='relu'))
model.add(BaseLayer(64, 128, activation='relu'))
model.add(OutputLayer(128, 10, activation='softmax',loss='cross_entropy'))

model.fit(train_x, train_y)
model.evaluate(test_x, test_y, batch_size=16)

# ============測試============
for index, x in enumerate(test_x[:9]):
    x = x[np.newaxis, :]
    output = model.predict(x)
    cls = np.argmax(output)
    label = test_y[index]

    plt.subplot(331 + index)
    plt.subplots_adjust(wspace=0.3, hspace=0.8)
    plt.imshow(test_img[index])
    plt.title('Pre:%d Label:%d' % (cls, test_label[index]))

plt.show()