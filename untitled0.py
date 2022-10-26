# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:54:08 2022

@author: anuna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("homeprices (2).csv")


a = df.isnull().sum()

df['bedrooms'] = df['bedrooms'].bfill()

#x = df.iloc[:,:-1]
#y = df.iloc[:,-1]

x = df[['area','bedrooms','age']]
y = df['price']

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)


reg.score(x,y)


reg.predict([[3000,4,5]])

