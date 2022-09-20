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
打開 `MINIST_classifier.py`並執行。主要程式碼如下：
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

訓練結果：

![Figure_1](https://user-images.githubusercontent.com/102845636/191244800-e8482baf-0144-4bd9-8dd7-d57e25cda0b6.png)
![Figure_1](https://user-images.githubusercontent.com/102845636/191244941-107cb761-2299-4b18-b253-8ff41398c6d7.png)

## 利用卷積網路結合全連接層的手寫辨識數字範例:
打開 `MINIST_convolution＿classifier.py`並執行。主要程式碼如下：
```py
# ============建置模型============
model = Sequential(epoch=EPOCH, batch_size=BATCHSIZE, learning_rate=LR, optimizer='SGD')
model.add(ConvolutionLayer(flt_n=5, flt_h=3, flt_w=3, input_shape=(1, 28, 28)))
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
```
訓練結果：

![Figure_1](https://user-images.githubusercontent.com/102845636/191246209-839e30c3-a3e4-4fdc-889b-14f7561d7f05.png)
![Figure_1](https://user-images.githubusercontent.com/102845636/191246318-401d5579-4423-4576-9137-a07675333bcb.png)

## 利用卷積網路結合全連接層應用於cifar10範例:
打開 `cifar10.py`並執行。主要程式碼如下：
```py
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
```
訓練結果：

![Figure_1](https://user-images.githubusercontent.com/102845636/191252361-5950c43d-32cd-4df0-b104-a44276386a40.png)
![圖片1](https://user-images.githubusercontent.com/102845636/191253050-efccbfc7-5162-4a5c-9362-6aaa85f66681.png)





