# 利用Numpy實作神經網路!

**利用Numpy套件所搭建而成的簡易神經網路框架，其使用上與tensorflow相似，但在神經網路層只擁有**
```
1. Base Layer (全連接層)
2. Convolution Layer (卷積層)
3. Flatten Layer (展開層)
4. Maxpooling Layer (池化層)
5. Dropout Layer (丟棄層)
```
 
**模型方面目前只擁有**
```
1. Sequential Model (序列模型)
```

**損失函數目前只擁有**
```
1. MSE (均方誤差)
2. Cross Entropy (交叉熵)
```

**激活函式上只擁有**
```
1. linear
2. relu
3. sigmoid
4. tanh
5. softmax
```

**在使用上只需要引用**
```py
from Model import Sequential
from layer import *
```

## 利用全連接網路的手寫辨識數字範例:
打開 `MINIST_classifier.py`
```py
# ============建置模型============
model = Sequential(epoch=EPOCH, batch_size=BATCHSIZE, learning_rate=LR, optimizer='AdaGrad')
model.add(BaseLayer(input_shape=784, output_shape=64, activation='relu'))
model.add(BaseLayer(128, activation='relu'))
model.add(OutputLayer(10, activation='softmax',loss='cross_entropy'))

model.compile()
model.fit(train_x, train_y)
model.evaluate(test_x, test_y, batch_size=16)
```
