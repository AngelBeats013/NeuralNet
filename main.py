import os
import sys

import numpy as np
import pandas as pd

import path
import pre_process
from ANN import ANN

if len(sys.argv) < 7:
    print('At least 6 arguments required. Got: %s' % (len(sys.argv) - 1))
    print('Please provide: data set, percent of data for training, maximum iterations, learning rate, number of hidden'
          'layers and neuron number of each hidden layer')
    print('For example, arguments could be "car 80 200 0.5 2 4 2". This means using car data set, 80 percent of data'
          'for training, maximum 200 iterations, 0.5 learning rate, 2 hidden layers, (4, 2) neurons for each layer.')
    exit(1)

data_set = str(sys.argv[1]).strip()
train_percent = int(sys.argv[2])
max_iter = int(sys.argv[3])
learning_rate = float(sys.argv[4])
hidden_layer_num = int(sys.argv[5])
neuron_num = [int(x) for x in sys.argv[6:len(sys.argv)]]

if len(neuron_num) != hidden_layer_num:
    print('Not enough neuron numbers given!')
    print('You specified %s hidden layers, but only %s numbers given' % (hidden_layer_num, len(neuron_num)))
    exit(1)

print('%s%% data used for training' % train_percent)
print('Max iterations: %s' % max_iter)
print('Learning rate: %s' % learning_rate)
for i in range(hidden_layer_num):
    print('Hidden layer %s, %s neurons' % (i + 1, neuron_num[i]))

extra_test_data_path = None
if data_set == 'car':
    if not os.path.isfile(path.CAR_DATA_PROCESSED_PATH):
        pre_process.car_pre_process()
    print('Using car data set')
    data_path = path.CAR_DATA_PROCESSED_PATH
elif data_set == 'iris':
    if not os.path.isfile(path.IRIS_DATA_PROCESSED_PATH):
        pre_process.iris_pre_process()
    print('Using iris data set')
    data_path = path.IRIS_DATA_PROCESSED_PATH
elif data_set == 'adult':
    if not os.path.isfile(path.ADULT_DATA_PROCESSED_PATH):
        pre_process.adult_pre_process()
    print('Using adult data set')
    data_path = path.ADULT_DATA_PROCESSED_PATH
    extra_test_data_path = path.ADULT_TEST_DATA_PROCESSED_PATH
else:
    print('Data set not recognized!')
    exit(1)

# Separate testing data from training data and combine with extra testing data
df = pd.read_csv(data_path)
msk = np.random.rand(len(df)) < train_percent / 100.0
train_data = df[msk]
test_data = df[~msk]
if extra_test_data_path is not None:
    df_test = pd.read_csv(extra_test_data_path)
    test_data = pd.concat(test_data, df_test)
train_data.reindex()
test_data.reindex()

# Train the ANN, test and print the report
clf = ANN(max_iter, learning_rate, hidden_layer_num, neuron_num)
clf.train(train_data)
clf.test(test_data)
