import sys
import os
import csv
import numpy as np

import utils
from utils import error

feature_number = 30

def usage():
    error('%s [dataset]' % sys.argv[0])

def scale(feature_matrix):
    min_matrix = np.min(feature_matrix, axis = 0)
    max_matrix = np.max(feature_matrix, axis = 0)
    scaled_feature_matrix = (feature_matrix - min_matrix) / (max_matrix - min_matrix)
    return scaled_feature_matrix

def read_data(filename):
    # checks
    if not os.path.isfile(filename):
        error('no such file: %s' % filename)

    # parser: csv to feature lists
    try:
        with open(filename, 'r') as fs:
            reader = csv.reader(fs)
            row_number = sum(1 for row in reader)
            fs.seek(0)
            x = np.empty([row_number, feature_number])
            y = np.empty([row_number, 1])
            i_row = 0
            for row in reader:
                for i, field in enumerate(row):
                    if i == 1:
                        y[i_row][0] = 1.0 if field == "M" else 0.0
                    elif i >= 1:
                        x[i_row][i - 2] = float(field)
                i_row += 1
    except:
        error("invalid dataset")

    return x, y

if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
    if not os.path.isfile(sys.argv[1]):
        error('no such file: %s' % sys.argv[1])

    x, y = read_data(sys.argv[1])
    x = scale(x)
    print(x)
    