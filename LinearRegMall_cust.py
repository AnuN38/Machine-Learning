# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:01:29 2022

@author: anuna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Mall_Customers.csv")

a = df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Genre']=le.fit_transform(df['Genre'])


x = df.iloc[:,0:4].values
y = df.iloc[:,-1].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
x_pred = reg.predict(x_train)



