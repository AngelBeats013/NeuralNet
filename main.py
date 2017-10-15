import os
import sys

import pre_process

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
hidden_layers = int(sys.argv[5])
neuron_num = [int(x) for x in sys.argv[6:len(sys.argv)]]

if len(neuron_num) != hidden_layers:
    print('Not enough neuron numbers given!')
    print('You specified %s hidden layers, but only %s numbers given' % (hidden_layers, len(neuron_num)))

print('%s%% data used for training' % train_percent)
print('Max iterations: %s' % max_iter)
print('Learning rate: %s' % learning_rate)
for i in range(hidden_layers):
    print('Hidden layer %s, %s neurons' % (i+1, neuron_num[i]))

if data_set == 'car':
    if not os.path.isfile(pre_process.CAR_DATA_PROCESSED_PATH):
        pre_process.car_pre_process()
    print('Using car data set')
    # Call ANN train function here
elif data_set == 'iris':
    if not os.path.isfile(pre_process.IRIS_DATA_PROCESSED_PATH):
        pre_process.iris_pre_process()
    print('Using iris data set')
    # Call ANN train function here
elif data_set == 'adult':
    if not os.path.isfile(pre_process.ADULT_DATA_PROCESSED_PATH):
        pre_process.adult_pre_process()
    print('Using adult data set')
    # Call ANN train function here
else:
    print('Data set not recognized!')