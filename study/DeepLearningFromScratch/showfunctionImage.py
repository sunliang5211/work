# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:26:22 2019

@author: Thinkpad
"""

import numpy as np
import matplotlib.pyplot as plt


x1 = np.arange(-10,10,0.1)
y1 = np.power(np.e,x1)

x2 = np.arange(-10,10,0.1)
y2 = np.power(np.e,-x2)

x3 = np.arange(-10,10,0.1)
y3 = 1 / np.power(np.e,x3)

x4 = np.arange(-10,10,0.1)
y4 = 1 /np.power(np.e,-x4)

x5 = np.arange(-10,10,0.1)
y5 = 1 / (1 + np.power(np.e,x5))

x6 = np.arange(-10,10,0.1)
y6 = 1 / (1 + np.power(np.e,-x6))


plt.plot(x1,y1)
plt.show()

plt.plot(x2,y2)
plt.show()

plt.plot(x3,y3)
plt.show()

plt.plot(x4,y4)
plt.show()

plt.plot(x5,y5)
plt.show()

plt.plot(x6,y6)
plt.show()

####三角函数图像
x = np.arange(0,10,0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle = "--",label = "cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin&cos")
plt.legend()
plt.show()







