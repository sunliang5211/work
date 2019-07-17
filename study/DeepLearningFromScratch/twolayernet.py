# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:04:50 2019

@author: Thinkpad
"""

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    