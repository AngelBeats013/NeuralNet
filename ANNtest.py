import random


def ANN_Training():
    '''
    每次iteration输入一个instance
    input: input_dataset
    percent: training_percent
    max: maximum_iterations
    hid_num: number of hidden layers
    neuron_num: number of neurons in each hidden layer
    '''
    input=[1,0,1]
    wlist=[[[-0.4,0.2,0.4,-0.5],[0.2,-0.3,0.1,0.2]],[[0.1,-0.3,-0.2]]]

    percent=1
    max=1
    hid_num=1
    neuron_num_list=[2]
    target_output=[1]
    neuron_num_list.append(len(target_output))
    print(neuron_num_list)

    error=1.00
    iteration=0
    train_input=input

    # while error is not 0 or iteration is not max:
    if iteration==0:

            iteration_wlist=random_w(len(input),neuron_num_list, len(target_output), hid_num)
            iteration_outlist=forward(len(input),neuron_num_list, len(target_output), hid_num,input,wlist)

def random_w(input_neuron, layer_neuron_list, output_num,hid_num1):
    pre_layer_neuron = input_neuron
    hid_num = hid_num1


    iteration_wlist = []

    # hidden layer
    while hid_num is not 0:

        layer_neuron=layer_neuron_list[hid_num1-hid_num]
        layer_neuron_count=layer_neuron



        layer_wlist = []
        while layer_neuron_count is not 0: #每个下个layer的xi寻找w，所有的list1存到wlist
            pre_layer_neuron_count = pre_layer_neuron
            bias = random.randint(1, 5)
            neuron_wlist = [bias]
            while pre_layer_neuron_count is not 0: #一个xi对应的wi，存入list1

                w = random.randint(1, 5)
                neuron_wlist.append(w)
                pre_layer_neuron_count = pre_layer_neuron_count-1
            layer_wlist.append(neuron_wlist)

            layer_neuron_count = layer_neuron_count-1
        iteration_wlist.append(layer_wlist)
        hid_num = hid_num - 1
        pre_layer_neuron=layer_neuron

    #output layer

    layer_wlist=[]

    layer_neuron = output_num
    layer_neuron_count = layer_neuron
    while layer_neuron_count is not 0:

        pre_layer_neuron_count=pre_layer_neuron
        bias = random.randint(1, 5)
        neuron_wlist = [bias]
        while pre_layer_neuron_count is not 0:

            w = random.randint(1, 5)
            neuron_wlist.append(w)
            pre_layer_neuron_count = pre_layer_neuron_count - 1
        layer_wlist.append(neuron_wlist)
        layer_neuron_count = layer_neuron_count - 1

    iteration_wlist.append(layer_wlist)
    print(iteration_wlist)

    return iteration_wlist


def forward(input_neuron,layer_neuron_list,output_num,hid_num1,input,iteration_wlist):
    '''
    给所有w随机赋值，计算net和sigmoid（）
    pre_num1: 本layer的节点数
    post_num1: 下一个layer节点数
    hid_num: hidden layer数
    input： input数据
    output_node: 最后output几个节点
    :return: outlist（所有h（）和output）, weightlist(所有w)
    '''
    pre_layer_neuron=input_neuron

    hid_num=hid_num1
    layer_outlist=[]
    iteration_outlist=[]
    layer_input=input
    i=0
    while i is not hid_num+1:

        if i==0:
            print('Input Layer:')
        else:
            print('Layer %d (hidden Layer): ' % i)

        layer_wlist=iteration_wlist[i]
        layer_neuron=layer_neuron_list[i]

        j=0
        layer_outlist = []
        while j is not layer_neuron: #每个下个layer的xi寻找w，所有的list1存到wlist
            neuron_wlist=layer_wlist[j]


            k=0
            net=0
            while k is not pre_layer_neuron+1: #一个xi对应的wi，存入list1

                if k==0:
                    net=net+neuron_wlist[k]
                else:
                    net=net+neuron_wlist[k]*layer_input[k-1]
                k=k+1
            neuron_output=sigmoid(net)

            layer_outlist.append(neuron_output)
            j=j+1
        iteration_outlist.append(layer_outlist)
        i=i+1
        layer_input=layer_outlist
        pre_layer_neuron=layer_neuron
        print(iteration_outlist)
    return iteration_outlist



def sigmoid(netsum):
    e = 2.718
    result=1/(e**(-netsum)+1)
    return result
