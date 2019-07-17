# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:33:58 2019

@author: Thinkpad
"""

import numpy as np

arr = np.array([1,1,1])
arr1 = np.array([[1,2],[3,4],[5,6]])

print(np.dot(arr,arr1))
#rint(np.dot(arr1,arr))

def mean_squared_err(y,t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error_batch(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
           
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7 )) / batch_size

def cross_entropy_error_noonehot(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size

t = [0,0,1,0,0,0,0,0,0,0]
t1 = [2]
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

print(mean_squared_err(np.array(y),np.array(t)))
print(cross_entropy_error(np.array(y),np.array(t)))
print(cross_entropy_error_batch(np.array(y),np.array(t)))
print(cross_entropy_error_noonehot(np.array(y),np.array(t1)))

import sys,os
sys.path.append(os.pardir)
from ch01.dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test) = load_mnist(normalize = True,one_hot_label = True)
print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
































