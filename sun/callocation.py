# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 19:07:29 2019

@author: Thinkpad
"""

import pandas as pd 

df = pd.read_csv('D:/work/sun/compare_result.csv')
df['area_e'] = df['east'] - df ['west']
df['area_n'] = df['north'] - df['south']
df['area'] = abs(df['area_e'])*111 * abs(df['area_n'])*111

df.to_csv('D:/work/sun/compare_result1111.csv')
 
    