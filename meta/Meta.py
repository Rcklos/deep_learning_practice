#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 神经元实现
@Date        : 2021/12/11 03:19:08
@Author      : ifish
@version     : 1.0
'''
import math
import numpy as np


class AddMeta:
    def __init__(self) -> None:
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class MulMeta:
    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
    

class DivMeta:
    def __init__(self) -> None:
        self.x = None

    def forward(self, x):
        self.x = x
        out = 1 / x
        return out

    def backward(self, dout):
        dx = dout * (-1 / (self.x ** 2))
        return dx


class ExpMeta:
    def __init__(self) -> None:
        self.x = None

    def forward(self, x):
        self.x = x
        out = math.exp(x)
        return out

    def backward(self, dout):
        dx = dout * math.exp(self.x)
        return dx


class DotMeta:
    def __init__(self) -> None:
        self.x = None
        self.w = None

    def forward(self, x, w):
        self.x = x
        self.w = w
        out = np.dot(x, w)

    def forward(self, dout):
        dx = dout * self.w.T
        dw = self.x.T * dout
        return dx, dw