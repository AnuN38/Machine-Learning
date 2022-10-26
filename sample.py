# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:58:28 2022

@author: anuna
"""

import pandas as pd

df = pd.read_csv("50_Startups.csv")


a = df.isnull().sum()


df.dtypes

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['State'] = le.fit_transform(df['State'])


m = df.corr()


x = df.iloc[:,:-1]
y = df.iloc[:,-1]


import numpy as np
np.std(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train,y_train)


y_pred = mlr.predict(x_test)

mlr.score(x_test,y_test)


