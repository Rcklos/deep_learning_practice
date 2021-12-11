#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 神经网络框架
@Date        : 2021/12/11 04:23:47
@Author      : ifish
@version     : 1.0
'''
from collections import OrderedDict
from layer.layer import *
import numpy as np


class TowLayerNet:
    def __init__(self, input_size, hiden_size, output_size, weight_init_std=0.01) -> None:
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hiden_size)
        self.params['b1'] = np.zeros(hiden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hiden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Affine2'] = AffineLayer(self.params['w2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def saveNetWork(self, path):
        pass
    
    def gradient(self, x, t):
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self,x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
