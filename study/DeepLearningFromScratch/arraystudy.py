# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:52:19 2019

@author: Thinkpad
"""

import numpy as np

A = np.array([1,2,3,4])
print(A)
print(A.ndim)
print(A.shape)
print(A.shape[0])

B = np.array([[1,2],[3,4],[5,6]])
print(B)
print(B.ndim)
print(B.shape)

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print(np.dot(A,B))
print(np.dot(B,A))

A = np.array([[1,2],[3,4],[5,6]])
B = np.array([7,8])
print(np.dot(A,B))
#print(np.dot(B,A))

X = np.array([1,2])
W = np.array([[1,3,5],[2,4,6]])
print(np.dot(X,W))