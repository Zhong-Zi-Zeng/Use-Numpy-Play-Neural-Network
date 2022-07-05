import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 訓練資料
train_x = []
train_y = []
for i in np.linspace(0,1,40):
    y = i + np.random.random() / 7
    train_x.append(i)
    train_y.append(y)

# 建立權重
w0 = np.random.random((1, 20))
b0 = np.random.random((20))

w1 = np.random.random((20, 1))
b1 = np.random.random((1))

# 訓練

fig = plt.figure()
plt.ion()
for t in range(200):
    pre = []
    fig.clf()
    for index, x in enumerate(train_x):
        u0 = np.dot(x, w0) + b0
        y0 = sigmoid(u0)

        u1 = np.dot(y0, w1) + b1
        y1 = sigmoid(u1)
        pre.append(y1)

        loss = ((y1 - train_y[index]) ** 2) / 2
        print("loss = %.5f"%(loss))

        # 反向傳播
        d_1 = (y1 - train_y[index]) * (1 - y1) * y1
        d_w1 = np.dot(y0.T, d_1)
        d_b1 = d_1

        d_0 = d_1 * (1 - y0) * y0
        d_w0 = np.dot(x, d_0)
        d_b0 = d_0

        # 更新
        w1 = w1 - 0.4 * d_w1
        b1 = b1 - 0.4 * d_b1
        w0 = w0 - 0.4 * d_w0
        b0 = b0 - 0.4 * d_b0

    plt.scatter(train_x,train_y,30)
    plt.scatter(train_x,pre,30)
    plt.pause(0.1)

plt.ioff()
plt.show()



