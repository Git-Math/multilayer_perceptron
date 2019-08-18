import sys
import os
import csv
import numpy as np
import random

import utils
from utils import error

def usage():
    error('%s [dataset]' % sys.argv[0])

def scale(feature_matrix):
    min_matrix = np.min(feature_matrix, axis = 1).reshape(-1, 1)
    max_matrix = np.max(feature_matrix, axis = 1).reshape(-1, 1)
    scaled_feature_matrix = (feature_matrix - min_matrix) / (max_matrix - min_matrix)
    return scaled_feature_matrix

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
    y_mp[y_mp == 0.0] += 1e-15
    y_mp[y_mp == 1.0] -= 1e-15
    return -1 / n * (np.dot(y, np.log(y_mp).T) + np.dot(1 - y, np.log(1 - y_mp).T))

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
            x = np.empty([feature_number, row_number])
            y = np.empty([2, row_number])
            i_row = 0
            for row in reader:
                for i, field in enumerate(row):
                    if i == 1:
                        if field == "M":
                            y[0][i_row] = 1
                            y[1][i_row] = 0
                        else:
                            y[0][i_row] = 0
                            y[1][i_row] = 1
                    elif i >= 1:
                        x[i - 2][i_row] = float(field)
                i_row += 1
    except:
        error("invalid dataset")

    return x, y

if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
    if not os.path.isfile(sys.argv[1]):
        error('no such file: %s' % sys.argv[1])

    learning_rate = 0.05
    epoch = 10000
    feature_number = 30
    x, y = read_data(sys.argv[1], feature_number)
    x = scale(x)
    x_val = x[:, 455:]
    x = x[:, :455]
    y_val = y[:, 455:]
    y = y[:, :455]
    layers = 4
    neurons = [feature_number, int(feature_number * 2 / 3 + 1), int(feature_number * 2 / 3 + 1), 2]
    # He initialization
    np.random.seed(0)
    wb = []
    for i in range(layers - 1):
        w = np.random.randn(neurons[i + 1], neurons[i]) / np.sqrt(neurons[i])
        b = np.random.randn(neurons[i + 1], 1)
        wb.append({ "w": w, "b": b})
    for i in range(epoch + 1):
        y_mp, za = feedforward(wb, x, layers, neurons)
        y_mp_val, za_val = feedforward(wb, x_val, layers, neurons)
        l = loss(y[0], y_mp[0])
        l_val = loss(y_val[0], y_mp_val[0])
        if i % 1000 == 0:
            print("epoch %d/%d - loss: %f - val_loss: %f" % (i, epoch, l, l_val))
        wb = backpropagation(x, y, y_mp, wb, za, learning_rate)
    count = 0
    for i in range(y.shape[1]):
        if y[0][i] == 1  and y_mp[0][i] > 0.5 or y[0][i] == 0  and y_mp[0][i] < 0.5:
            count += 1
    print(count)
    for i in range(y_val.shape[1]):
        if y_val[0][i] == 1  and y_mp_val[0][i] > 0.5 or y_val[0][i] == 0  and y_mp_val[0][i] < 0.5:
            count += 1
    print(count)
