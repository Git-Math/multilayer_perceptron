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

def dsoftmax(z):
    return z * (1. - z)

def feedforward(wb, x, layers, neurons):
    za = []
    for i in range(layers - 1):
        prev_x = x if i == 0 else za[i - 1]["a"]
        z = np.dot(wb[i]["w"], prev_x) + wb[i]["b"]
        a = softmax(z) if i == layers - 2 else relu(z)
        za.append({ "z": z, "a": a})
    y_mp = za[layers - 2]["a"][1]
    return y_mp, za

def loss(y, y_mp):
    n = y.shape[1]
    return -1 / n * (np.dot(y, np.log(y_mp).T) + np.dot(1 - y, np.log(1 - y_mp).T))

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
            y = np.empty([1, row_number])
            i_row = 0
            for row in reader:
                for i, field in enumerate(row):
                    if i == 1:
                        y[0][i_row] = 1.0 if field == "M" else 0.0
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

    feature_number = 30
    x, y = read_data(sys.argv[1], feature_number)
    x = scale(x)
    x_val = x[:, 455:]
    x = x[:, :455]
    y_val = y[:, 455:]
    y = y[:, :455]
    layers = 4
    neurons = [feature_number, int(feature_number * 2 / 3 + 1), int(feature_number * 2 / 3 + 1), 2]
    learning_rate = 0.01
    # He initialization
    np.random.seed(0)
    wb = []
    for i in range(layers - 1):
        w = np.random.randn(neurons[i + 1], neurons[i]) / np.sqrt(neurons[i])
        b = np.random.randn(neurons[i + 1], 1)
        wb.append({ "w": w, "b": b})
    y_mp, za = feedforward(wb, x, layers, neurons)
    y_mp_val, za_val = feedforward(wb, x_val, layers, neurons)
    l = loss(y, y_mp)
    l_val = loss(y_val, y_mp_val)
    print("epoch - loss: %f - val_loss: %f" % (l, l_val))
