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

        # Randomly generate weights
        cur_layer = 0
        next_layer = 1
        while next_layer < len(self.neuron_num):
            # Calculate weights for each layer
            for cur_neuron in range(self.neuron_num[cur_layer]+1): # plus 1 for bias weight
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

        # Run back propagation algorithm
        error = 1.0
        for _ in range(self.max_iter):
            if error == 0.0:
                break
            for instance in train_data.itertuples():
                instance = list(instance)
                instance.pop(0) # First value is index, which is not needed
                self.bp(instance)
                break
            break


        # error = 0
        # iteration = 0
        # pre_num = 10#一共几个属性，不会读。。
        # post_num = neuron_num
        # targetlist=[]#target 就是class有几种，列成list，是那种就把对应位置的数字写为1，剩下的都是0
        # weightlist=[]
        # biaslist=[]
        # output_node=10#class 有几种，不会读。。。
        #
        # train_input, test_input = percent_input(data_file_path, train_percent)#分离train和test
        # while error is not 0 or iteration is not max_iter: #training
        #
        #     if iteration==1:
        #         outlist, weightlist , biaslist= forward1(pre_num, post_num, hidden_layer_num, train_input, output_node)
        #         weightlist, biaslist=backward(learning_rate, outlist, weightlist, biaslist, hidden_layer_num)
        #     else:
        #         outlist=forward(pre_num, post_num, hidden_layer_num, data_file_path, weightlist, biaslist)
        #         weightlist,biaslist= backward(learning_rate, outlist, weightlist, biaslist, hidden_layer_num)
        #     iteration=iteration+1
        #     outputlist=outlist[hidden_layer_num]
        #     error=0
        #     i=0
        #     for output in outputlist:
        #         error=error+0.5*(targetlist[i]-output)^2
        #         i=i+1
        #     print('Total training error =  ' , error)
        #
        #
        # for data_file_path in test_input:#testing
        #     outlist = forward(pre_num, post_num, hidden_layer_num, data_file_path, weightlist, biaslist)
        #     #计算error
        #     print('Total testing error =  ', error)

    def test(self, test_data):
        '''
        Test this ANN against test data and print test report
        :param test_data: test data in pandas DataFrame
        '''
        # print('Testing data size: %s' % len(test_data))
        pass

    def bp(self, instance):
        '''
        Run back propagation on given data instance and update weights
        :param instance: data instance
        '''
        target = instance.pop()  # Last value is target value
        out_values = self.forward(instance)

        # Backward pass
        self.backward(instance, target, out_values)
        pass

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
        print(delta_dict)

        # Update weights

        pass

    def forward1(self, pre_num1, post_num1 , hid_num, input, output_node):
        '''
        给所有w随机赋值，计算net和sigmoid（）
        pre_num1: 本layer的节点数
        post_num1: 下一个layer节点数
        hid_num: hidden layer数
        input： input数据
        output_node: 最后output几个节点
        :return: outlist（所有h（）和output）, weightlist(所有w)
        '''
        pre_num = pre_num1
        post_num = post_num1
        hidden_num = hid_num
        weightlist=[]
        list1 = []
        wlist = []
        netlist=[]
        input_list = input  # input的x值
        outlist=[]#所有h（net（））
        biaslist=[]

        while hidden_num is not 0:
            print('Layer %d (hidden Layer): ' % hidden_num)
            while post_num is not 0: #每个下个layer的xi寻找w，所有的list1存到wlist
                while pre_num is not 0: #一个xi对应的wi，存入list1
                    w = random.randint(1, 5)
                    list1.append(w)
                    post_num = post_num-1
                wlist.append(list1)
                print('Neuron %d weights:' % pre_num1 - pre_num+1, list1)
                list1.clear()
                pre_num = pre_num-1
            weightlist.append(wlist)
            wlist.clear()
            bias= random.randint(1, 5)#bias
            biaslist.append(bias)
            #算出所有的h（net（））存入netlist
            netlist=(cal_w_x(wlist, input_list, bias))
            outlist.append(netlist)
            netlist.clear()
            input_list=netlist

            hidden_num = hidden_num - 1
            pre_num=post_num

        print('Layer %d (output Layer): ' % hid_num+1)
        post_num=output_node
        while post_num is not 0:
            while pre_num is not 0:
                w = random.randint(1, 5)
                list1.append(w)
                post_num = post_num - 1
                wlist.append(list1)
                print('Neuron %d weights:' % pre_num1 - pre_num + 1, list1)
                list1.clear()
                pre_num = pre_num - 1
            weightlist.append(wlist)
            wlist.clear()
            bias = random.randint(1, 5)  # bias
            biaslist.append(bias)
           # 算出所有的h（net（））存入netlist
            netlist=(cal_w_x(wlist, input_list, bias))
            outlist.append(netlist)



        return outlist, weightlist, biaslist


    def cal_w_x(self, wlist , input_list, bias):
        '''
        wlist：该layer所有的weight
        input_list: 该layer所有的x
        算出下个layer某个点的net经过sigmoid()处理后的节点output
        :return: 一个layer的sigmoid()处理后的节点output
        '''
        result=[]
        for list in wlist:
            netsum = 0
            i=0

            for w in list:
                netsum=netsum+w*input_list[i]+bias
                i=i+1
            result.append(sigmoid(netsum))
        return result

    def sigmoid(self, x):
        e = 2.718
        result=1/(e ** (-x) + 1)
        return result