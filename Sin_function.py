import numpy as np
from Model import Sequential
from BaseLayer import BaseLayer
from OutputLayer import OutputLayer
import matplotlib.pyplot as plt


# ============載入數據集============
train_x = np.linspace(-5, 5, 100)
train_x = np.reshape(train_x, (100, 1))
train_y = np.sin(train_x)
train_y = np.reshape(train_y, (100, 1))


test_x = np.linspace(-5, 5, 200)
test_x = np.reshape(test_x, (200, 1))
test_y = np.sin(test_x)
test_y = np.reshape(test_y, (200, 1))


# ============超參數設置============
EPOCH = 500
BATCHSIZE = 16
LR = 0.1

# ============建置模型============
model = Sequential(epoch=EPOCH, batch_size=BATCHSIZE, learning_rate=LR)
model.add(BaseLayer(1, 64, activation='sigmoid'))
model.add(OutputLayer(64, 1, activation='linear', loss='MSE'))

model.fit(train_x, train_y)
model.evaluate(test_x, test_y, batch_size=16)

# ============測試============
pre = []
for index, x in enumerate(test_x):
    x = x[np.newaxis, :]
    output = model.predict(x)[0][0]
    pre.append(output)

plt.plot(test_x, pre, label='Pre')
plt.plot(test_x, test_y, label='Label')
plt.show()
