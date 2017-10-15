import random


def ANN_Training(input, percent ,max, hid_num, neuron_num, learnrate):
    '''
    每次iteration输入一个instance
    input: input_dataset
    percent: training_percent
    max: maximum_iterations
    hid_num: number of hidden layers
    neuron_num: number of neurons in each hidden layer
    '''
    # file = open("sample.txt")
    #
    # while 1:
    #     line = file.readline()
    #     if not line:
    #         break
    #     pass  # do something
    error = 0
    iteration = 0
    pre_num = 10#一共几个属性，不会读。。
    post_num = neuron_num
    targetlist=[]#target 就是class有几种，列成list，是那种就把对应位置的数字写为1，剩下的都是0
    weightlist=[]
    biaslist=[]
    output_node=10#class 有几种，不会读。。。

    train_input, test_input = percent_input(input, percent)#分离train和test
    while error is not 0 or iteration is not max: #training

        if iteration==1:
            outlist, weightlist , biaslist= forward1(pre_num, post_num, hid_num, train_input,output_node)
            weightlist, biaslist=backward(learnrate, outlist, weightlist,biaslist, hid_num)
        else:
            outlist=forward(pre_num, post_num, hid_num, input,weightlist,biaslist)
            weightlist,biaslist= backward(learnrate, outlist, weightlist, biaslist, hid_num)
        iteration=iteration+1
        outputlist=outlist[hid_num]
        error=0
        i=0
        for output in outputlist:
            error=error+0.5*(targetlist[i]-output)^2
            i=i+1
        print('Total training error =  ' , error)


    for input in test_input:#testing
        outlist = forward(pre_num, post_num, hid_num, input, weightlist, biaslist)
        #计算error
        print('Total testing error =  ', error)



def percent_input(input, percent):
    '''
    分离train 和test用的数据
    :param input: 输入的数据
    :param percent: 用来train的百分比
    :return: train数据+test数据
    '''
    dictionary=dict()
    total_num=0
    train_input=[]
    test_input=[]

    for data in input:
        dict[total_num]=data
        total_num=total_num+1
    train_num=total_num*percent
    while train_num is not 0:
        i = random.randint(0, total_num)
        if dictionary.has_key(i):#has_key()错了?
            train_input.append(dictionary.get(
                i,0
            ))
            del dictionary[i]
            train_num=train_num-1

    for testdata in dictionary:
        test_input.append(dictionary.get(testdata))

    return train_input, test_input






def forward1(pre_num1, post_num1 , hid_num, input, output_node):
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


def forward(pre_num1, post_num1 , hid_num, input, output_node,weightlist,biaslist):
    '''
   利用上一次的bias和weight，计算net和sigmoid（）
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

        bias=biaslist[hidden_num-1]
        wlist=weightlist[hidden_num-1]
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
        for wlist in weightlist:  # 算出所有的h（net（））存入netlist
            netlist.append(cal_w_x(wlist, input_list, bias))
        outlist.append(netlist)



    return outlist, weightlist, biaslist


def cal_w_x(wlist , input_list, bias):
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

def sigmoid(netsum):
    e = 2.718
    result=1/(e**(-netsum)+1)
    return result


def backward(learnrate, outlist, weightlist, biaslist, hid_num):
    '''

    :param learnrate:
    :param outlist: 所有out和sigmoid处理的net
    :param weightlist: 所有的w
    :param hid_num:
    :return:更新过的weightlist
    '''
    i=hid_num
    for wlist in weightlist:#output layer
        if i is hid_num:
            netlist= outlist[i]
            xlist=outlist[i-1]
            targetlist=#target 还是最后一行。。。
            j = 0
            for list in wlist:
                k=0
                net= netlist[j]
                target= targetlist[j]
                for w in list:
                    delta_w= learnrate*(target-net)*net*(1-net)*xlist[k]
                    w=w+delta_w#更新w怎么做？
                    k=k+1
                bias=biaslist[j]
                delta_bias=learnrate*(target-net)*net*(1-net)
                bias=bias+delta_bias
                j=j+1

        else:#hidden layer
            netlist = outlist[i]
            xlist = outlist[i - 1]
            j = 0
            for list in wlist:
                k = 0
                net = netlist[j]

                for w in list:
                    delta_w = learnrate *  net * (1 - net) * xlist[k]
                    w = w + delta_w
                    k = k + 1
                bias = biaslist[j]
                delta_bias = learnrate *  net * (1 - net)
                bias = bias + delta_bias
                j = j + 1
        i=i-1
        return weightlist, biaslist



