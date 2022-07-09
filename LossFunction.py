import numpy as np
import copy


# ============Mean Squared Error===========
def MSE(pre, label, diff=False):
    if not diff:
        return np.sum((pre - label) ** 2) / 2
    else:
        return pre - label


# ============Cross Entropy===========
def cross_entropy(pre, label, diff=False):
    if not diff:
        return -np.sum(label * np.log(pre + 1e-7))  # avoid divide zero error
    else:
        result = copy.copy(pre)

        for index, l in enumerate(label):
            cls = np.argmax(l)
            result[index, :] = -1 / pre[index, cls]

        return result
