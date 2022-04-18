#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 
@Date        : 2021/12/11 23:32:50
@Author      : ifish
@version     : 1.0
'''

import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./dataset'))

from matplotlib import lines
import pickle
import numpy as np
from network.network import AdaGrad, TowLayerNet, TensorFlowTowLayerNetWork
from dataset.dataset import fetch_fer2013
import matplotlib.pyplot as plt
from numpy.lib.function_base import average


def fetch_dataset():
    print('\r装配FER2013数据集...', end='')
    dataset = fetch_fer2013(x_num=999999, t_num=999999)
    global x_img
    global x_label
    global t_img
    global t_label
    x_img = dataset['train_img']
    x_label = dataset['train_label']
    t_img = dataset['test_img']
    t_label = dataset['test_label']
    print('\r装配FER2013数据集完成!')


def simple_test_fer2013():
    # network
    img_size = 48 * 48 * 1
    network = TowLayerNet(input_size=img_size, hiden_size=128,  output_size=7)
    optimizer = AdaGrad()
    train_size = len(x_img)
    batch_size = 100
    iters_num = 1000
    
    epoch = max(1, train_size // batch_size)
    acc_train_list = []
    acc_test_list  = []
    train_loss_list = []
    acc = 0
    acc_tmp = 0
    count_epoch = 0
    weight_file = None

    dir_list = os.listdir('./weight')
    weight_epoch_list = []
    # read
    for dir in dir_list:
        if dir == 'weight.bak.pkl':
            continue
        try:
            id = int(dir.split('.')[0].split('_')[1])
        except(IndexError) as e:
            print(dir)
            print(e)
            return
        weight_epoch_list.append(id)
    if len(weight_epoch_list) > 0:
        weight_file = open('./weight/weight.bak.pkl', 'rb')
        if weight_file:
            params = pickle.load(weight_file)
            network = params['network']
            # for key in ('w1', 'b1', 'w2', 'b2'):
            #     network.params[key] = params[key]
            acc_train_list = params['acc_train']
            acc_test_list = params['acc_test']
            optimizer = params['optimizer']
            weight_file.close()
        else:
            count_epoch = np.max(weight_epoch_list)
            with open('./weight/weight_%d.bak.pkl' % (count_epoch), 'rb') as f:
                params = pickle.load(f)
                for key in ('w1', 'b1', 'w2', 'b2'):
                    network.params[key] = params[key]
                acc_train_list = params['acc_train']
                acc_test_list = params['acc_test']
                optimizer = params['optimizer']
                f.close()
    
    if not weight_file:
        for i in range(count_epoch * epoch, iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_img[batch_mask]
            t_batch = x_label[batch_mask]

            grads = network.gradient(x_batch, t_batch)  

            # for key in ('w1', 'b1', 'w2', 'b2'):
            #     network.params[key] -= lr * grads[key]

            optimizer.update(network.params, grads)

            # print(grad['w1'])
            # if i > 3:
            #     exit(0)

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
            
            if i % epoch == 0:
                # test
                acc_tmp = network.accuracy(t_img, t_label)
                acc_test_list.append(acc_tmp)
                # train
                acc_tmp = network.accuracy(x_img, x_label)
                acc_train_list.append(acc_tmp)
                
                # save
                with open('./weight/weight_%d.bak.pkl' % count_epoch, 'wb') as f:
                    params = {}
                    # for key in ('w1', 'b1', 'w2', 'b2'):
                    #     params[key] = network.params[key]
                    params['network'] = network
                    params['acc_train'] = acc_train_list
                    params['acc_test'] = acc_test_list
                    params['optimizer'] = optimizer
                    pickle.dump(params, f)
                    count_epoch += 1
                    f.close()
                
            print('\rtrain acc: %.4f%% learning: %.2f%%' % (acc_tmp*100, i / iters_num * 100), end='')
        
        # save
        with open('./weight/weight.bak.pkl', 'wb') as f:
            params = {}
            # for key in ('w1', 'b1', 'w2', 'b2'):
            #     params[key] = network.params[key]
            params['network'] = network
            params['acc_train'] = acc_train_list
            params['acc_test'] = acc_test_list
            params['optimizer'] = optimizer
            pickle.dump(params, f)
            f.close()
        acc_tmp = network.accuracy(x_img, x_label)
        print('\rtrain acc: %.4f%% learning: 100.00%%' % (acc_tmp*100))

    acc = network.accuracy(x_img, x_label)
    print('train acc: %.2f%%' % (acc * 100))

    acc = network.accuracy(t_img, t_label)
    print('test acc: %.2f%%' % (acc * 100))

    x = np.arange(len(acc_train_list)) + 1
    plt.plot(x, acc_train_list, label='train acc')
    plt.plot(x, acc_test_list, label='test acc', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.show()

    acc_train_list.remove(np.max(acc_train_list))
    acc_train_list.remove(np.min(acc_train_list))
    print('训练集平均准确率: %.2f%%' % (np.average(acc_train_list) * 100))
    

def simple_test_weight():
    img_size = 48 * 48 * 1
    network = TowLayerNet(input_size=img_size, hiden_size=100,  output_size=7)
    optimizer = AdaGrad()
    acc_test_list = []
    acc_train_list = []

    # read
    weight_file = None
    f = None
    dir_list = os.listdir('./weight')
    weight_epoch_list = []
    for dir in dir_list:
        if dir == 'weight.bak.pkl':
            continue
        id = int(dir.split('.')[0].split('_')[1])
        weight_epoch_list.append(id)
    if len(weight_epoch_list) > 0:
        weight_file = open('./weight/weight.bak.pkl', 'rb')
        if weight_file:
            params = pickle.load(weight_file)
            # for key in ('w1', 'b1', 'w2', 'b2'):
            #     network.params[key] = params[key]
            network = params['network']
            acc_train_list = params['acc_train']
            acc_test_list = params['acc_test']
            optimizer = params['optimizer']
            weight_file.close()
        else:
            count_epoch = np.max(weight_epoch_list)
            f = open('./weight/weight_%d.bak.pkl' % (count_epoch), 'rb')
            if f :
                params = pickle.load(f)
                # for key in ('w1', 'b1', 'w2', 'b2'):
                #     network.params[key] = params[key]
                network = params['network']
                acc_train_list = params['acc_train']
                acc_test_list = params['acc_test']
                optimizer = params['optimizer']
                f.close()
    
    if not (f or weight_file):
        print('weight file not found!')
        return
        
    acc = network.accuracy(x_img, x_label)
    print('train acc: %.2f%%' % (acc * 100))
    acc = network.accuracy(t_img, t_label)
    print('test acc: %.2f%%' % (acc * 100))

    test_size = len(x_img)
    batch_size = 100
    x_time = 10000
    acc_list = []
    for i in range(x_time):
        batch_mask = np.random.choice(test_size, batch_size)
        x_batch = x_img[batch_mask]
        t_batch = x_label[batch_mask]
        acc = network.accuracy(x_batch, t_batch)
        acc_list.append(acc)

    x = np.arange(len(acc_list)) + 1
    plt.plot(x, acc_list, label='acc rate')
    plt.xlabel('time(s)')
    plt.ylabel('acc rate')
    plt.ylim(0, 1.0)
    plt.legend()
    
    acc_list.remove(np.max(acc_list))
    acc_list.remove(np.min(acc_list))
    print('平均准确率: %.2f%%' % (np.average(acc_list) * 100))

    plt.show()


def tensor_test_fer2013():
    img_size = 48 * 48 * 1
    network = TensorFlowTowLayerNetWork(input_size=img_size, hidden_size=128,  output_size=7)
    batch_size = 100
    iters_num = 1000
    train_size = len(x_img)
    epoch = max(1, train_size // batch_size)
    # epoch = 80
    
    if not network.is_load:
        history = network.train(x_img, x_label, batch_size=batch_size, epochs=epoch)
    acc = network.accuracy(x_test=t_img, y_test=t_label)
    print(acc)

import random

def subset(alist, idxs):
    '''
        用法: 根据下标idxs取出列表alist的子集
        alist: list
        idxs: list
    '''
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list

def split_list(alist, group_num=4, shuffle=True, retain_left=False):
    '''
        用法: 将alist切分成group个子列表，每个子列表里面有len(alist)//group个元素
        shuffle: 表示是否要随机切分列表，默认为True
        retain_left: 若将列表alist分成group_num个子列表后还要剩余，是否将剩余的元素单独作为一组
    '''

    index = list(range(len(alist))) # 保留下标

    # 是否打乱列表
    if shuffle: 
        random.shuffle(index) 
    
    elem_num = len(alist) // group_num # 每一个子列表所含有的元素数量
    sub_lists = {}
    
    # 取出每一个子列表所包含的元素，存入字典中
    for idx in range(group_num):
        start, end = idx*elem_num, (idx+1)*elem_num
        sub_lists['set'+str(idx)] = subset(alist, index[start:end])
    
    # 是否将最后剩余的元素作为单独的一组
    if retain_left and group_num * elem_num != len(index): # 列表元素数量未能整除子列表数，需要将最后那一部分元素单独作为新的列表
        sub_lists['set'+str(idx+1)] = subset(alist, index[end:])
    
    return sub_lists

if __name__ == '__main__':
    fetch_dataset()
    # simple_test_fer2013()
    # simple_test_weight()

    tensor_test_fer2013()
    