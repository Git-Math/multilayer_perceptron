import sys
import os
import csv
import numpy as np

import utils
from utils import error
import mp_train

def usage():
    error('%s [dataset]' % sys.argv[0])

def load_train_data():
    try:
        with open(mp_train.train_data_dirname + "/" + mp_train.layers_filename, "r") as layers_file:
            layers = int(layers_file.readline())
            layers_file.close()
        with open(mp_train.train_data_dirname + "/" + mp_train.neurons_filename, "r") as neurons_file:
            line = neurons_file.readline()
            neurons = line.split(",")
            neurons = list(map(int, neurons))
            neurons_file.close()
        wb = []
        for i in range(layers - 1):
            w = np.load(mp_train.train_data_dirname + "/" + mp_train.w_filename + str(i) + ".npy")
            b = np.load(mp_train.train_data_dirname + "/" + mp_train.b_filename + str(i) + ".npy")
            wb.append({ "w": w, "b": b})
        x_min = np.load(mp_train.train_data_dirname + "/" + mp_train.x_min_filename + ".npy")
        x_max = np.load(mp_train.train_data_dirname + "/" + mp_train.x_max_filename + ".npy")
    except:
        error("load train data failed")
    return layers, neurons, wb, x_min, x_max

def save_predict(y_mp):
    try:
        with open("predict.csv", "w") as predict_file:
            for i in range(y_mp.shape[1]):
                predict_file.write("M\n" if y_mp[0][i] > 0.5 else "B\n")
            predict_file.close()
    except:
        error("save predict failed")

def mean_squared_error(y, y_mp):
    return 1 / y.shape[1] * np.sum((y_mp - y) ** 2)

def mean_absolute_error(y, y_mp):
    return 1 / y.shape[1] * np.sum(np.absolute(y_mp - y))

def negative_log_likelihood(y_mp):
    return -1 / y_mp.shape[1] * np.sum(np.log(y_mp + 1e-15))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
    if not os.path.isfile(sys.argv[1]):
        error('no such file: %s' % sys.argv[1])

    feature_number = 30
    data = mp_train.read_data(sys.argv[1], feature_number)
    x, y = mp_train.split_xy(data)
    layers, neurons, wb, x_min, x_max = load_train_data()
    x = mp_train.scale(x, x_min, x_max)
    try:
        y_mp, za = mp_train.feedforward(wb, x, layers, neurons)
        l = mp_train.loss(y[0], y_mp[0])
    except:
        error("invalid train data")
    print("binary cross entropy: %.4f" % l)
    mse = mean_squared_error(y, y_mp)
    print("mean squared error: %.4f" % mse)
    mae = mean_absolute_error(y, y_mp)
    print("mean absolute error: %.4f" % mae)
    nll = negative_log_likelihood(y_mp)
    print("negative log likelihood: %.4f" % nll)
    count = 0
    for i in range(y.shape[1]):
        if y[0][i] == 1  and y_mp[0][i] > 0.5 or y[0][i] == 0  and y_mp[0][i] <= 0.5:
            count += 1
    print("prediction accuracy: %d/%d" % (count, y.shape[1]))
    print("prediction accuracy percentage: %.2f%%" % (count / y.shape[1] * 100))
    save_predict(y_mp)
