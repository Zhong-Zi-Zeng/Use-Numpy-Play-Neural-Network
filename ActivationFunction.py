import numpy as np
import copy


# ============linear============
def linear(x, diff=False, **kwargs):
    if not diff:
        return x
    else:
        return 1


# ============ReLU============
def relu(x, diff=False, **kwargs):
    if not diff:
        x[x <= 0] = 0

        return x
    else:
        x[x > 0] = 1
        x[x <= 0] = 0

        return x


# ============Sigmoid============
def sigmoid(x, diff=False, **kwargs):
    if not diff:
        return 1 / (1 + np.exp(-x))
    else:
        output = sigmoid(x, False)

        return output * (1 - output)


# ============tanh============
def tanh(x, diff=False, **kwargs):
    if not diff:
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    else:
        return 1 - tanh(x, diff=False) ** 2


# ============Softmax============
def softmax(x, diff=False, **kwargs):
    if not diff:
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    else:
        result = copy.copy(kwargs["y"])

        for index, l in enumerate(kwargs["label"]):
            cls = np.argmax(l)
            for i in range(kwargs["label"].shape[1]):
                result[index, i] *= -(kwargs["y"][index, cls] - 1) if i == cls else -kwargs["y"][index, cls]

        return result
