import matplotlib.pyplot as plt
import numpy as np
from Optimizer import *
from tqdm import tqdm

optimizer_table = {"SGD": SGD, 'Momentum': Momentum, 'AdaGrad' : AdaGrad}


class Sequential:
    def __init__(self, epoch, batch_size, learning_rate=0.1, optimizer='SGD'):
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optimizer_table[optimizer](learning_rate=learning_rate, batch_size=batch_size)

        self.layer_list = []

    def add(self, layer):
        self.layer_list.append(layer)

    def calculate_acc(self, pre, label):
        acc = np.sum(pre.argmax(axis=1) == label.argmax(axis=1), dtype=np.float64)

        return acc

    def predict(self, x):
        return self.FP(x)

    def evaluate(self, x, label, batch_size, print_mode=True):
        acc_list = []
        loss_list = []

        if print_mode:
            range_ = tqdm(range(0, x.shape[0], batch_size), ncols=120, desc="Evaluate on text data")
        else:
            range_ = range(0, x.shape[0], batch_size)

        for i in range_:
            batch_x = x[i: min(i + batch_size, x.shape[0]), :]
            batch_y = label[i: min(i + batch_size, x.shape[0]), :]

            output = self.predict(batch_x)
            acc = self.calculate_acc(output, batch_y) / len(batch_x)
            loss = self.output_layer.get_loss(output, batch_y) / len(batch_x)

            acc_list.append(acc)
            loss_list.append(loss)

        if print_mode:
            print("test loss:{:.3f}, test acc:{:.3f}".format(np.mean(loss_list), np.mean(acc_list)))

        return np.mean(loss_list), np.mean(acc_list)

    def fit(self, x, label, val_x=None, val_y=None, verbose=0):
        self.output_layer = self.layer_list[len(self.layer_list) - 1]

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        for t in range(self.epoch):
            pbar = tqdm(range(0, x.shape[0], self.batch_size), ncols=120)

            for i in pbar:
                batch_x = x[i: min(i + self.batch_size, x.shape[0]), :]
                batch_y = label[i: min(i + self.batch_size, x.shape[0]), :]

                output = self.FP(batch_x)
                self.BP(batch_y)
                loss = self.output_layer.get_loss(output, batch_y) / len(batch_x)
                acc = self.calculate_acc(output, batch_y) / len(batch_x)

                pbar.set_description("epoch:{} - loss:{:.4f} - acc:{:.4f}".format(t, loss, acc))

            t_l, t_a = self.evaluate(x, label, x.shape[0], print_mode=False)
            train_loss.append(t_l)
            train_acc.append(t_a)

            if val_x is not None:
                v_l, v_a = self.evaluate(val_x, val_y, val_x.shape[0], print_mode=False)
                val_loss.append(v_l)
                val_acc.append(v_a)

        self.show_figure(verbose, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)

    def show_figure(self, verbose, **kwargs):
        if verbose == 0:
            plt.plot(range(self.epoch), kwargs["train_loss"], label='Train Loss')
            plt.plot(range(self.epoch), kwargs["train_acc"], label='Train acc')

            print('Train_loss:{:.3f} , Train_acc:{:.3f}'.format(np.mean(kwargs["train_loss"]), np.mean(kwargs["train_acc"])))

        elif verbose == 1 and kwargs["val_loss"]:
            plt.plot(range(self.epoch), kwargs["train_loss"], label='Train Loss')
            plt.plot(range(self.epoch), kwargs["val_loss"], label='Val Loss')

            print('Train_loss:{:.3f} , Train_acc:{:.3f}'.format(np.mean(kwargs["train_loss"]), np.mean(kwargs["val_loss"])))

        elif verbose == 2 and kwargs["val_loss"] and kwargs["val_acc"]:
            plt.plot(range(self.epoch), kwargs["train_loss"], label='Train Loss')
            plt.plot(range(self.epoch), kwargs["train_acc"], label='Train acc')
            plt.plot(range(self.epoch), kwargs["val_loss"], label='Val Loss')
            plt.plot(range(self.epoch), kwargs["val_acc"], label='Val acc')
            print('Train_loss:{:.3f} , Train_acc:{:.3f}'.format(np.mean(kwargs["train_loss"]), np.mean(kwargs["train_acc"])))
            print('Val_loss:{:.3f} , Val_acc:{:.3f}'.format(np.mean(kwargs["val_loss"]), np.mean(kwargs["val_acc"])))

        plt.legend()
        plt.show()

    def FP(self, x):
        output = x

        for layer in self.layer_list:
            output = layer.FP(output, is_train=True)

        return output

    def BP(self, batch_y):
        delta = self.output_layer.BP(batch_y, self.batch_size)

        for index in reversed(range(len(self.layer_list) - 1)):
            delta = self.layer_list[index].BP(delta)

        self.gradient_decent()

    def gradient_decent(self):
        self.optimizer.gradient_decent(self.layer_list)
