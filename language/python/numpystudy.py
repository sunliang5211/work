# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:30:48 2019

@author: Thinkpad
"""

from numpy import *

a = random.rand(4,4)
print(a)

b = mat(a)
print(b)

c = b.I
print(c)

d = b * c
print(d)

print(d-eye(4))