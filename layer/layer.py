#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 层实现，已实现反向传播
@Date        : 2021/12/11 03:57:35
@Author      : ifish
@version     : 1.0
'''
import numpy as np
import math
from function.function import softmax, cross_entropy_error

class ReluLayer:
    '''
    @description: Relu层
    ------------
    @param:
    ------
    @Returns:
    --------
    '''
    def __init__(self) -> None:
        self.x = None
        self.mask = None

    def forward(self, x):
        if type(x) is int:
            if x > 0:
                self.x = x
                return x
            else:
                self.x = 0
                return 0
        else:
            self.mask = x <= 0
            out = x.copy()
            out[self.mask] = 0
            return out
    
    def backward(self, dout):
        if self.x is None and self.mask is not None:
            dout[self.mask] = 0
            dx = dout
            return dx
        else:
            return dout * 1 if self.x > 0 else 0


class SigmoidLayer:
    '''
    @description: sigmoid层
    ------------
    @param:
    ------
    @Returns:
    --------
    '''
    def __init__(self) -> None:
        self.out = None
        pass

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class AffineLayer:
    '''
    @description: Affine层
    ------------
    @param:
    ------
    @Returns:
    --------
    '''
    def __init__(self, w, b) -> None:
        self.x = None
        self.w = w
        self.b = b
        self.dw = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    

class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx