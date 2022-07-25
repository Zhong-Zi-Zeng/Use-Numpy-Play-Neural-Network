import numpy as np
from Model import Sequential
from layer import *
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


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
(train_img, train_label), (test_img, test_label) = cifar10.load_data()


# ============資料預處理============
test_x = normalization(test_img)
test_y = label_to_hotcode(test_label)

train_x = normalization(train_img)
train_y = label_to_hotcode(train_label)

train_x = train_x.transpose(0, 3, 2, 1)
test_x = test_x.transpose(0, 3, 2, 1)

# ============超參數設置============
EPOCH = 10
BATCHSIZE = 32
LR = 0.1

# ============建置模型============
model = Sequential(epoch=EPOCH, batch_size=BATCHSIZE, learning_rate=LR, optimizer='SGD')
model.add(ConvolutionLayer(flt_n=5, flt_h=3, flt_w=3, input_shape=(3, 32, 32)))
model.add(MaxpoolingLayer(pool=2))

model.add(ConvolutionLayer(flt_n=10, flt_h=3, flt_w=3))
model.add(MaxpoolingLayer(pool=2))

model.add(FlattenLayer())
model.add(BaseLayer(64, activation='relu'))
model.add(BaseLayer(128, activation='relu'))
model.add(OutputLayer(10, activation='softmax',loss='cross_entropy'))

model.compile()
model.fit(train_x, train_y)
model.evaluate(test_x, test_y, batch_size=16)


# ============測試============
for index, x in enumerate(test_x):
    x = x[np.newaxis, :]
    output = model.predict(x)
    cls = np.argmax(output)
    label = test_y[index]

    plt.imshow(test_img[index])
    plt.title('Pre:%d Label:%d' % (cls, test_label[index]))
    plt.show()



