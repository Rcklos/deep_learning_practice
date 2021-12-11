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

import numpy as np
from network.network import TowLayerNet
from dataset.dataset import fetch_fer2013


def simple_test_fer2013():
    dataset = fetch_fer2013(x_num=500, t_num=100)
    x_img = dataset['train_img']
    x_label = dataset['train_label']
    t_img = dataset['test_img']
    t_label = dataset['test_label']

    # network
    img_size = 48 * 48 * 1
    network = TowLayerNet(input_size=img_size, hiden_size=50,  output_size=7)
    iters_num = 10000
    lr = 0.1
    for i in range(iters_num):
        grad = network.gradient(x_img, x_label)

        for key in ('w1', 'b1', 'w2', 'b2'):
            network.params[key] -= lr * grad[key]
        
        print('\r learning: %.4f%%' % (i / iters_num * 100), end='')
    print('\r learning: %.4f%%' % (i / iters_num * 100))

    acc = network.accuracy(x_img, x_label)
    print('train acc: %.2f' % (acc))
    acc = network.accuracy(t_img, t_label)
    print('test acc: %.2f' % (acc))
        

if __name__ == '__main__':
    simple_test_fer2013()