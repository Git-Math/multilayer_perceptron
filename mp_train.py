import sys
import os
import csv
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

import utils
from utils import error

train_data_dirname = "train_data"
layers_filename = "layers.csv"
neurons_filename = "neurons.csv"
w_filename = "w"
b_filename = "b"
x_min_filename = "x_min"
x_max_filename = "x_max"

def usage():
    error('%s [dataset]' % sys.argv[0])

def scale(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def dloss_softmax(y, y_mp):
    return y_mp - y

def feedforward(wb, x, layers, neurons):
    za = []
    for i in range(layers - 1):
        prev_x = x if i == 0 else za[i - 1]["a"]
        z = np.dot(wb[i]["w"], prev_x) + wb[i]["b"]
        a = softmax(z) if i == layers - 2 else relu(z)
        za.append({ "z": z, "a": a})
    y_mp = za[layers - 2]["a"]
    return y_mp, za

def loss(y, y_mp):
    n = len(y)
    return -1 / n * (np.dot(y, np.log(y_mp + 1e-15).T) + np.dot(1 - y, np.log(1 - y_mp + 1e-15).T))

def backpropagation(x, y, y_mp, wb, za, learning_rate):
    len_wb = len(wb)
    dloss_wb = []
    for i in range(len_wb - 1, -1, -1):
        prev_x = x if i == 0 else za[i - 1]["a"]
        prev_y = dloss_softmax(y, y_mp) if i == len_wb - 1 else dloss_a
        dloss_z = prev_y if i == len_wb - 1 else prev_y * drelu(za[i]["z"])
        dloss_a = np.dot(wb[i]["w"].T, dloss_z)
        dloss_w = 1. / prev_x.shape[1] * np.dot(dloss_z, prev_x.T)
        dloss_b = 1. / prev_x.shape[1] * np.dot(dloss_z, np.ones([dloss_z.shape[1],1]))
        dloss_wb.insert(0, { "w": dloss_w, "b": dloss_b })
    for i in range(len_wb):
        wb[i]["w"] -= learning_rate * dloss_wb[i]["w"]
        wb[i]["b"] -= learning_rate * dloss_wb[i]["b"]
    return wb

def print_loss_graph(l, l_val):
    plt.plot(l, label = "Training Loss")
    plt.plot(l_val, label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Graph")
    plt.show()

def save_train_data(wb, layers, neurons, x_min, x_max):
    try:
        os.makedirs(train_data_dirname, exist_ok=True)
        with open(train_data_dirname + "/" + layers_filename, "w") as layers_file:
            layers_file.write(str(layers) + "\n")
            layers_file.close()
        with open(train_data_dirname + "/" + neurons_filename, "w") as neurons_file:
            for (i, e) in enumerate(neurons):
                neurons_file.write(str(e))
                neurons_file.write("\n") if i == len(neurons) - 1 else neurons_file.write(",")
            neurons_file.close()
        for (i, e) in enumerate(wb):
            np.save(train_data_dirname + "/" + w_filename + str(i), e["w"])
            np.save(train_data_dirname + "/" + b_filename + str(i), e["b"])
        np.save(train_data_dirname + "/" + x_min_filename, x_min)
        np.save(train_data_dirname + "/" + x_max_filename, x_max)
    except:
        error("save train data failed")

def shuffle_data(data):
    data = np.transpose(data)
    np.random.shuffle(data)
    return np.transpose(data)

def split_xy(data):
    x = data[2:, :]
    y = data[:2, :]
    return x, y

def split_train_val(x, y):
    train_size = int(x.shape[1] * 0.8)
    x_val = x[:, train_size:]
    x = x[:, :train_size]
    y_val = y[:, train_size:]
    y = y[:, :train_size]
    return x, x_val, y, y_val

def read_data(filename, feature_number):
    # checks
    if not os.path.isfile(filename):
        error('no such file: %s' % filename)

    # parser: csv to feature lists
    try:
        with open(filename, 'r') as fs:
            reader = csv.reader(fs)
            row_number = sum(1 for row in reader)
            fs.seek(0)
            data = np.empty([feature_number + 2, row_number])
            i_row = 0
            for row in reader:
                for i, field in enumerate(row):
                    if i == 1:
                        if field == "M":
                            data[0][i_row] = 1
                            data[1][i_row] = 0
                        else:
                            data[0][i_row] = 0
                            data[1][i_row] = 1
                    elif i > 1:
                        data[i][i_row] = float(field)
                i_row += 1
    except:
        error("invalid dataset")

    return data

if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
    if not os.path.isfile(sys.argv[1]):
        error('no such file: %s' % sys.argv[1])

    learning_rate = 0.04
    epoch = 10000
    feature_number = 30
    np.random.seed(0)
    data = read_data(sys.argv[1], feature_number)
    data = shuffle_data(data)
    x, y = split_xy(data)
    x_min = np.min(x, axis = 1).reshape(-1, 1)
    x_max = np.max(x, axis = 1).reshape(-1, 1)
    x = scale(x, x_min, x_max)
    x, x_val, y, y_val = split_train_val(x, y)
    l = []
    l_val = []
    layers = 4
    neurons = [feature_number]
    for i in range(layers - 2):
        neurons.append(int(feature_number * 2 / 3 + 1))
    neurons.append(2)
    # He initialization
    wb = []
    for i in range(layers - 1):
        w = np.random.randn(neurons[i + 1], neurons[i]) / np.sqrt(neurons[i])
        b = np.random.randn(neurons[i + 1], 1)
        wb.append({ "w": w, "b": b})
    val_low = sys.float_info.max
    no_val_prog = 0
    for i in range(epoch + 1):
        y_mp, za = feedforward(wb, x, layers, neurons)
        y_mp_val, za_val = feedforward(wb, x_val, layers, neurons)
        l.append(loss(y[0], y_mp[0]))
        l_val.append(loss(y_val[0], y_mp_val[0]))
        if l_val[-1] >= val_low:
            no_val_prog += 1
        else:
            no_val_prog = 0
            val_low = l_val[-1]
        if no_val_prog == 10:
            print("early stop epoch %d/%d - loss: %.4f - val_loss: %.4f" % (i, epoch, l[-1], l_val[-1]))
            break
        if i % 100 == 0:
            print("epoch %d/%d - loss: %.4f - val_loss: %.4f" % (i, epoch, l[-1], l_val[-1]))
        if i < epoch:
            wb = backpropagation(x, y, y_mp, wb, za, learning_rate)
    save_train_data(wb, layers, neurons, x_min, x_max)
    print_loss_graph(l, l_val)

