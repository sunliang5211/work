# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:27:43 2019

@author: Thinkpad
"""
import numpy as np
import matplotlib.pylab as plt

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

def step_function(x):
    return np.array(x > 0,dtype = np.int)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def rule(x):
    return np.maximum(0,x)


x = np.arange(-5,5,0.1)
y = step_function(x)

plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

x = np.arange(-5,5,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

x = np.arange(-5,5,0.1)
y = rule(x)
plt.plot(x,y)
plt.ylim(-0.1,5)
plt.show()




