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

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def feedforward(wb, x, layers, neurons):
    za = []
    for i in range(layers - 1):
        prev_x = x if i == 0 else za[i - 1]["a"]
        z = np.dot(wb[i]["w"], prev_x) + wb[i]["b"]
        a = softmax(z) if i == layers - 2 else relu(z)
        za.append({ "z": z, "a": a})
    y_mp = za[layers - 2]["a"]
    return y_mp, za

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
