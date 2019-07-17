# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:07:19 2019

@author: Thinkpad
"""

import numpy as np
import matplotlib.pyplot as plt


#####e的x次幂的函数图像

x = np.arange(-10,10,0.1)
y = np.power(np.e,x)

plt.plot(x,y)
plt.show()

#x = np.arange(-10,0,0.1)
#y = np.power(np.e,x)

#plt.plot(x,y)
#plt.show()

#x = np.arange(0,16,0.1)
#y = np.power(np.e,x)

#plt.plot(x,y)
#plt.show()


#x = np.arange(-10,10,0.1)
#y = np.power(10,x)

#plt.plot(x,y)
#plt.show()

#x = np.arange(0,10,0.1)
#y = np.power(10,x)

#plt.plot(x,y)
#plt.show()

#####e的-x次幂的函数图像

x = np.arange(-10,10,0.1)
y = np.power(np.e,-x)

plt.plot(x,y)
plt.show()

#####e的1/1+e的-x次幂 的函数图像
x = np.arange(-10,10,0.1)
y = 1 / (1 + np.power(np.e,-x))

plt.plot(x,y)
plt.show()
