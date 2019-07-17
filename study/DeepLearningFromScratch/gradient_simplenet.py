# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:40:05 2019

@author: Thinkpad
"""

import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.functions import cross_entropy_error
from common.functions import softmax
from common.functions import numerical_gradient_2d

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss
    
    def f(W):
        return net.loss(x,t)
    
    
net = simpleNet()
print(net.W)

x = np.array([0.6,0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
t = np.array([0,0,1])
print(net.loss(x,t))

dW = numerical_gradient_2d(net.f,net.W)
print(dW)