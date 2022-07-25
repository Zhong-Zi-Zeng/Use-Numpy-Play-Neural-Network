import numpy as np
from sklearn.datasets import load_iris
from Model import Sequential
from layer import *

np.random.seed(10)

def label_to_hotcode(label):
    dim = np.max(label) + 1
    hotcode_label = np.zeros((len(label), dim))

    for idx, y in enumerate(label):
        hotcode_label[idx][y] = 1.0

    return hotcode_label

def normalization(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    n_x = (x - mean) / std

    return n_x

def shuffle(x, y):
    length = len(x)

    for i in range(length):
        r_1 = np.random.randint(length)
        r_2 = np.random.randint(length)

        x[r_1], x[r_2] = x[r_2], x[r_1]
        y[r_1], y[r_2] = y[r_2], y[r_1]

    return x, y

# ============超參數設置============
BATCHSIZE = 8
EPOCH = 2000
LR = 0.01

# ============訓練資料============
iris = load_iris()
data_x = np.array(iris.data)
data_y = np.array(iris.target)

data_x = normalization(data_x)
data_y = label_to_hotcode(data_y)

data_x, data_y = shuffle(data_x, data_y)

train_x, train_y, val_x, val_y = data_x[0:75], data_y[0:75], data_x[75:], data_y[75:]

# ============建置模型============
model = Sequential(epoch=EPOCH, batch_size=BATCHSIZE, learning_rate=LR, optimizer='AdaGrad')
model.add(BaseLayer(input_shape=4, output_shape=25, activation='relu'))
# model.add(DropoutLayer(0.5))
model.add(BaseLayer(25, activation='relu'))
# model.add(DropoutLayer(0.5))
model.add(OutputLayer(3, activation='softmax',loss='cross_entropy'))

model.compile()
model.fit(train_x, train_y, val_x, val_y, verbose=2)



