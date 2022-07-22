import numpy as np
from tools import im2col, col2im
import threading

np.random.seed(0)

# FP
img = np.arange(27).reshape((1, 3, 3, 3))

# ============L1============
x_0 = im2col(img, 2, 2)  # (12, 4)
w_0 = np.random.normal(0, 0.05, (2, 12))
b_0 = np.random.normal(0, 0.05, (2, 1))

u_0 = np.dot(w_0, x_0) + b_0  # (2, 4)
u_0 = u_0.T.reshape(1, 2, 2, 2).transpose(0, 3, 1, 2)  # (1, 2, 2, 2)

# ============L2============
x_1 = im2col(u_0, 2, 2)  # (8, 1)
w_1 = np.random.normal(0, 0.05, (2, 8))
b_1 = np.random.normal(0, 0.05, (2, 1))

u_1 = np.dot(w_1, x_1) + b_1  # (2, 1)
u_1 = u_1.T.reshape(1, 1, 1, 2).transpose(0, 3, 1, 2)  # (1, 2, 1, 1)

# Flatten
output = u_1.reshape((u_1.shape[0], -1))  # (1, 2)
label = np.array([[0, 1]])

# BP
d_1 = (output - label)  # Loss: cross_entropy 、Output Activation: Softmax (1, 2)
d_1 = d_1.reshape(1, 2, 1, 1)  # (1, 2, 1, 1)

d_1 = d_1.transpose(0, 2, 3, 1).reshape(1, 2)

grad_w = np.dot(x_1, d_1).T
grad_b = np.sum(d_1, axis=0, keepdims=True).T

print(grad_w)
print(grad_b)
