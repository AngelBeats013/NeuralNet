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
    input=[1,2,3]
    percent=1
    max=1
    hid_num=1
    neuron_num_list=[2]
    target_output=[1,0]

    error=1
    iteration=0
    train_input=input

    while error is not 0 or iteration is not max:
        if iteration==0:
            outlist,weightlist,biaslist= forward1(len(train_input), neuron_num_list, hid_num, train_input,len(target_output))


def forward1(input_neuron,layer_neuron_list,hid_num1,input,output_num):
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
    layer_neuron=layer_neuron_list[0]
    hid_num=hid_num1

    neuron_wlist=[]
    layer_wlist=[]
    iteration_wlist=[]
    iteration_biaslist=[]
    layer_outlist=[]
    iteration_outlist=[]
    layer_input=input



    weightlist=[]
    list1 = []
    wlist = []
    netlist=[]
    input_list = input  # input的x值
    outlist=[]#所有h（net（））
    biaslist=[]

    while hid_num is not 0:
        if hid_num==hid_num1:
            print('Input Layer:')
        else:
            print('Layer %d (hidden Layer): ' % hid_num1-hid_num)

        while layer_neuron is not 0: #每个下个layer的xi寻找w，所有的list1存到wlist
            while pre_layer_neuron is not 0: #一个xi对应的wi，存入list1
                w = random.randint(1, 5)
                neuron_wlist.append(w)
                layer_neuron = layer_neuron-1
            layer_wlist.append(neuron_wlist)
            print('Neuron %d weights:' % input_neuron - pre_layer_neuron+1, neuron_wlist)
            neuron_wlist.clear()
            pre_layer_neuron = pre_layer_neuron-1
        iteration_wlist.append(layer_wlist)

        bias=random.randint(1, 5)#bias
        iteration_biaslist.append(bias)
        #算出所有的h（net（））存入netlist
        layer_outlist=(cal_sigmoid(layer_wlist, layer_input, bias))
        iteration_outlist.append(layer_outlist)
        layer_outlist.clear()
        layer_wlist.clear()

        layer_input=iteration_outlist
        hid_num = hid_num - 1
        pre_layer_neuron=layer_neuron
        input_neuron=layer_neuron
        layer_neuron=layer_neuron_list[hid_num1-hid_num]

    print('Layer %d (hidden Layer): ' % hid_num1)
    layer_neuron=output_num
    while layer_neuron is not 0:
        while pre_layer_neuron is not 0:
            w = random.randint(1, 5)
            neuron_wlist.append(w)
            layer_neuron = layer_neuron - 1
        layer_wlist.append(neuron_wlist)
        print('Neuron %d weights:' % input_neuron - pre_layer_neuron+1, neuron_wlist)
        neuron_wlist.clear()
        pre_layer_neuron = pre_layer_neuron - 1

    iteration_wlist.append(layer_wlist)
    bias = random.randint(1, 5)
    iteration_biaslist.append(bias)
    layer_outlist = (cal_sigmoid(layer_wlist, layer_input, bias))
    iteration_outlist.append(layer_outlist)
    layer_outlist.clear()
    layer_wlist.clear()



    return iteration_outlist, iteration_wlist, iteration_biaslist

def cal_sigmoid(layer_wlist , layer_input, bias):
    '''
    wlist：该layer所有的weight
    input_list: 该layer所有的x
    算出下个layer某个点的net经过sigmoid()处理后的节点output
    :return: 一个layer的sigmoid()处理后的节点output
    '''
    result=[]
    for neuron_wlist in layer_wlist:
        netsum = 0
        i=0

        for w in neuron_wlist:
            netsum=netsum+w*layer_input[i]+bias
            i=i+1
        result.append(sigmoid(netsum))
    return result

def sigmoid(netsum):
    e = 2.718
    result=1/(e**(-netsum)+1)
    return result