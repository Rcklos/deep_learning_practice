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

import pickle
import numpy as np
from network.network import TowLayerNet
from dataset.dataset import fetch_fer2013
import matplotlib.pyplot as plt
from numpy.lib.function_base import average


def simple_test_fer2013():
    print('\r装配FER2013数据集...', end='')
    dataset = fetch_fer2013(x_num=99999999, t_num=100)
    x_img = dataset['train_img']
    x_label = dataset['train_label']
    t_img = dataset['test_img']
    t_label = dataset['test_label']
    print('\r装配FER2013数据集完成!')
    # network
    img_size = 48 * 48 * 1
    network = TowLayerNet(input_size=img_size, hiden_size=50,  output_size=7)
    train_size = len(x_img)
    batch_size = 100
    iters_num = 10000
    lr = 0.1
    epoch = max(1, train_size // batch_size)
    acc_train_list = []
    acc_test_list  = []
    train_loss_list = []
    acc = 0
    count_epoch = 0

    # read
    dir_list = os.listdir('./weight')
    weight_epoch_list = []
    for dir in dir_list:
        id = int(dir.split('.')[0].split('_')[1])
        weight_epoch_list.append(id)
    if len(weight_epoch_list) > 0:
        count_epoch = np.max(weight_epoch_list)
        with open('./weight/weight_%d.bak.pkl' % (count_epoch), 'rb') as f:
            params = pickle.load(f)
            for key in ('w1', 'b1', 'w2', 'b2'):
                network.params[key] = params[key]
            acc_train_list = params['acc_train']
            acc_test_list = params['acc_test']
            f.close()

    for i in range(count_epoch * epoch, iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_img[batch_mask]
        t_batch = x_label[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for key in ('w1', 'b1', 'w2', 'b2'):
            network.params[key] -= lr * grad[key]

        # print(grad['w1'])
        # if i > 3:
        #     exit(0)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % epoch == 0:
            acc_tmp = None
            acc_tmp = network.accuracy(x_img, x_label)
            acc_train_list.append(acc_tmp)
            acc_tmp = None
            acc_tmp = network.accuracy(t_img, t_label)
            acc_test_list.append(acc_tmp)
            # save
            with open('./weight/weight_%d.bak.pkl' % count_epoch, 'wb') as f:
                params = {}
                for key in ('w1', 'b1', 'w2', 'b2'):
                    params[key] = network.params[key]
                params['acc_train'] = acc_train_list
                params['acc_test'] = acc_test_list
                pickle.dump(params, f)
                count_epoch += 1
                f.close()

        print('\rnewer acc: %4.2f%% learning: %.2f%%' % (acc_tmp*100, i / iters_num * 100), end='')
    print('\rnewer acc: %4.2f%% learning: 100.00%%' % (acc_tmp*100))

    acc = network.accuracy(x_img, x_label)
    print('train acc: %.2f%%' % (acc * 100))
    acc = network.accuracy(t_img, t_label)
    print('test acc: %.2f' % (acc))

    x = np.arange(len(acc_train_list)) + 1
    plt.plot(x, acc_train_list, label='train acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.0)
    plt.show()

    acc_train_list.remove(np.max(acc_train_list))
    acc_train_list.remove(np.min(acc_train_list))
    print('训练集平均准确率: %.2f' % np.average(acc_train_list))
    
        

if __name__ == '__main__':
    simple_test_fer2013()