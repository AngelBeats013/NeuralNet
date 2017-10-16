import random

class ANN:
    def __init__(self, max_iter, learning_rate, hidden_layer_num, neuron_num):
        '''

        :param max_iter:
        :param learning_rate:
        :param hidden_layer_num:
        :param neuron_num:
        '''
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.hidden_layer_num = hidden_layer_num
        self.neuron_num = neuron_num
        self.weights_dict = dict()


    def train(self, train_data):
        '''
        Train an ANN using given data set and parameters
        :param train_data: training data in pandas DataFrame
        '''
        print('\nTraining data size: %s' % len(train_data))

        # Add input and output layer neuron_num list
        input_layer = [len(train_data.columns)-1]
        self.neuron_num = input_layer + self.neuron_num
        output_num = 1 # Only one output value is needed
        self.neuron_num.append(output_num)

        # Run back propagation algorithm
        error = 0.0
        for iter_num in range(self.max_iter):
            # Randomly generate weights
            cur_layer = 0
            next_layer = 1
            while next_layer < len(self.neuron_num):
                # Calculate weights for each layer
                for cur_neuron in range(self.neuron_num[cur_layer] + 1):  # plus 1 for bias weight
                    # Calculate weights for each neuron in current layer
                    neuron_weights = []
                    for _ in range(self.neuron_num[next_layer]):
                        neuron_weights.append(random.uniform(-1.0, 1.0))
                    # Add weights of this neuron to dict
                    if cur_layer not in self.weights_dict:
                        self.weights_dict[cur_layer] = dict()
                    self.weights_dict[cur_layer][cur_neuron] = neuron_weights
                cur_layer += 1
                next_layer += 1
            for instance in train_data.itertuples():
                instance = list(instance)
                instance.pop(0) # First value is index, which is not needed
                target = instance.pop()  # Last value is target value
                out_values = self.forward(instance)
                self.backward(instance, target, out_values)

            # Calc MSE
            error = 0.0
            for instance in train_data.itertuples():
                instance = list(instance)
                instance.pop(0)  # First value is index, which is not needed
                target = instance.pop()  # Last value is target value
                out_values = self.forward(instance)
                error += ((target - out_values[len(out_values) - 1][0]) ** 2)
            error = error / len(train_data)
            print('%s iteration, error rate: %s%%' % (iter_num, error * 100))
            if error == 0.0:
                break

        print('Training complete. Accuracy: %s%%' % ((1.0 - error) * 100))


    def test(self, test_data):
        '''
        Test this ANN against test data and print test report
        :param test_data: test data in pandas DataFrame
        '''
        print('\nTesting data size: %s' % len(test_data))
        error = 0.0
        for instance in test_data.itertuples():
            instance = list(instance)
            instance.pop(0)  # First value is index, which is not needed
            target = instance.pop()  # Last value is target value
            out_values = self.forward(instance)
            error += (target - out_values[len(out_values) - 1][0]) ** 2
        error = error / len(test_data)
        print('Test accuracy: %s%%' % ((1.0 - error) * 100))


    def forward(self, instance):
        '''
        Run forward pass for given data instance and return outputs of each node
        :param instance: data instance
        :return: output values for each node
        '''
        instance.insert(0, 1.0)  # Set bias for this net.
        out_values = dict()
        out_values[0] = dict()
        for i in range(len(instance)):
            out_values[0][i] = instance[i]

        # Forward pass
        cur_layer = 0
        next_layer = 1
        while next_layer < len(self.neuron_num):
            out_values[next_layer] = dict()
            # Calculate output value for each layer
            for j in range(self.neuron_num[next_layer]):
                # Calc output value for each neuron in next layer
                val = 0.0
                for i in range(self.neuron_num[cur_layer]):
                    val += self.weights_dict[cur_layer][i][j] * out_values[cur_layer][i]
                val = self.sigmoid(val)
                out_values[next_layer][j] = val
            cur_layer += 1
            next_layer += 1
        return out_values

    def backward(self, instance, target, out_values):
        # Calc delta for each node
        out_layer = len(self.neuron_num)-1
        delta_dict = dict()

        # First calculate output layer deltas
        delta_dict[out_layer] = dict()
        for i in range(self.neuron_num[out_layer]):
            output = out_values[out_layer][i]
            delta_dict[out_layer][i] = output * (1 - output) * (target - output)

        # Then hidden layer deltas, need to exclude input layer
        cur_layer = len(self.neuron_num)-2
        while cur_layer > 0:
            delta_dict[cur_layer] = dict()
            for i in range(self.neuron_num[cur_layer]):
                # Calc downstream deltas
                downstream = 0.0
                for j in range(self.neuron_num[cur_layer+1]):
                    downstream += delta_dict[cur_layer+1][j] * self.weights_dict[cur_layer][i][j]
                output = out_values[cur_layer][i]
                delta_dict[cur_layer][i] = output * (1 - output) * downstream
            cur_layer -= 1

        # Update weights, start from back
        cur_layer = len(self.neuron_num) - 2
        while cur_layer >= 0:
            for i in range(self.neuron_num[cur_layer]):
                for j in range(self.neuron_num[cur_layer+1]):
                    self.weights_dict[cur_layer][i][j] += self.learning_rate * delta_dict[cur_layer+1][j] * out_values[cur_layer][i]
            cur_layer -= 1

    def sigmoid(self, x):
        e = 2.718
        result=1/(e ** (-x) + 1)
        return result