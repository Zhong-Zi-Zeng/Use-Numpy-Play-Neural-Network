import numpy as np
import copy


# ============ReLU============
def relu(x, label=None, y=None, diff=False):
    if not diff:
        x[x <= 0] = 0

        return x
    else:
        x[x > 0] = 1
        x[x <= 0] = 0

        return x


# ============Sigmoid============
def sigmoid(x, label=None, y=None, diff=False):
    if not diff:
        return 1 / (1 + np.exp(-x))
    else:
        output = sigmoid(x, False)

        return output * (1 - output)


# ============Softmax============
def softmax(x, label=None, y=None, diff=False):
    if not diff:
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    else:
        result = copy.copy(y)

        for index, l in enumerate(label):
            cls = np.argmax(l)
            for i in range(label.shape[1]):
                result[index, i] *= -(y[index, cls] - 1) if i == cls else -y[index, cls]

        return result
