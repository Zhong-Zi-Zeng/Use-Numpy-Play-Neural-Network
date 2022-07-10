import numpy as np
from tqdm import tqdm, trange


class Sequential:
    def __init__(self, epoch, batch_size, learning_rate=0.1):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.layer_list = []

    def add(self, layer):
        self.layer_list.append(layer)

    def calculate_acc(self, pre, label):
        acc = np.sum(pre.argmax(axis=1) == label.argmax(axis=1))
        return acc

    def predict(self, x):
        return self.FP(x)

    def evaluate(self, x, label, batch_size):
        acc_list = []
        loss_list = []

        pbar = tqdm(range(0, x.shape[0], batch_size), ncols=120, desc="Evaluate on text data")
        for i in pbar:
            batch_x = x[i: min(i + batch_size, x.shape[0]), :]
            batch_y = label[i: min(i + batch_size, x.shape[0]), :]

            output = self.predict(batch_x)
            acc = self.calculate_acc(output, batch_y) / batch_size
            loss = self.output_layer.get_loss(output, batch_y) / batch_size

            acc_list.append(acc)
            loss_list.append(loss)

        print("test loss:{:.3}, test acc:{:.3}".format(np.mean(loss_list), np.mean(acc_list)))

    def fit(self, x, label):
        self.output_layer = self.layer_list[len(self.layer_list) - 1]

        for t in range(self.epoch):
            pbar = tqdm(range(0, x.shape[0], self.batch_size), ncols=120)

            for i in pbar:
                batch_x = x[i: min(i + self.batch_size, x.shape[0]), :]
                batch_y = label[i: min(i + self.batch_size, x.shape[0]), :]

                output = self.FP(batch_x)
                self.BP(batch_y)
                loss = self.output_layer.get_loss(output, batch_y) / self.batch_size
                acc = self.calculate_acc(output, batch_y) / self.batch_size
                pbar.set_description("epoch:{} - loss:{:.3} - acc:{:.3}".format(t, loss, acc))

    def FP(self, x):
        output = x

        for layer in self.layer_list:
            output = layer.FP(output)

        return output

    def BP(self, batch_y):
        delta = self.output_layer.BP(batch_y, self.batch_size)

        for index in reversed(range(len(self.layer_list) - 1)):
            next_weight = self.layer_list[index + 1].w
            delta = self.layer_list[index].BP(delta, next_weight)

        self.gradient_decent()

    def gradient_decent(self):
        for layer in self.layer_list:
            layer.w -= self.learning_rate * layer.d_w / self.batch_size
            layer.b -= self.learning_rate * layer.d_b / self.batch_size
