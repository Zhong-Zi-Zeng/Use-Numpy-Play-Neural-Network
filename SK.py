from sklearn import datasets
import numpy as np
from Model import Sequential
from BaseLayer import BaseLayer
from OutputLayer import OutputLayer
import matplotlib.pyplot as plt


def normalization(x):
    # num = x.shape[1]
    #
    #
    # for i in range(num):
    #     x[:, i] /= np.max(x[:, i])
    std = np.std(x, axis=0)
    u = np.mean(x, axis=0)
    x = (x - u) / std

    return x

def label_to_hotcode(label):
    dim = np.max(label) + 1
    hotcode_label = np.zeros((len(label), dim))

    for idx, y in enumerate(label):
        hotcode_label[idx][y] = 1

    return hotcode_label

def shuffle(x, y):
    length = len(x)

    for i in range(length):
        r_1 = np.random.randint(length)
        r_2 = np.random.randint(length)

        x[r_1], x[r_2] = x[r_2], x[r_1]
        y[r_1], y[r_2] = y[r_2], y[r_1]

    return x, y

# ============載入數據集============
data = datasets.load_wine()

data_x = data['data']
data_y = data['target']
target_name = data['target_names']

# ============資料預處理============
data_x, data_y = shuffle(data_x, data_y)
data_x = normalization(data_x)
data_y = label_to_hotcode(data_y)
train_x, train_y, test_x, test_y = data_x[:120], data_y[:120], data_x[120:], data_y[120:]


# ============超參數設置============
EPOCH = 1000
BATCHSIZE = 4
LR = 0.1

# ============建置模型============
model = Sequential(epoch=EPOCH, batch_size=BATCHSIZE, learning_rate=LR)
model.add(BaseLayer(13, 64, activation='relu'))
model.add(BaseLayer(64, 128, activation='relu'))
model.add(BaseLayer(128, 256, activation='relu'))
model.add(BaseLayer(256, 1024, activation='relu'))
model.add(OutputLayer(1024, 3, activation='softmax', loss='cross_entropy'))

model.fit(train_x, train_y)
model.evaluate(test_x, test_y, batch_size=8)

# ============測試============
# pre = []
# for index, x in enumerate(test_x):
#     x = x[np.newaxis, :]
#     output = model.predict(x)[0]
#     pre = np.argmax(output)
#     print('Pre:%d Label:%d'%(pre, np.argmax(test_y[index])))