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
    except:
        error("load train data failed")
    return layers, neurons, wb

if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
    if not os.path.isfile(sys.argv[1]):
        error('no such file: %s' % sys.argv[1])

    feature_number = 30
    x, y = mp_train.read_data(sys.argv[1], feature_number)
    x = mp_train.scale(x)
    layers, neurons, wb = load_train_data()
    try:
        y_mp, za = mp_train.feedforward(wb, x, layers, neurons)
        l = mp_train.loss(y[0], y_mp[0])
    except:
        error("invalid train data")
    print("binary cross entropy error: %f" % l)
    count = 0
    for i in range(y.shape[1]):
        if y[0][i] == 1  and y_mp[0][i] > 0.5 or y[0][i] == 0  and y_mp[0][i] < 0.5:
            count += 1
    print("prediction accuracy: %.2f%%" % round(count / y.shape[1], 2))
