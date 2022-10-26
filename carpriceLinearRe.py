# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:06:35 2022

@author: anuna
"""

import pandas as pd

df = pd.read_csv("CarPrice_Assignment (2).csv")


df2 = df[['symboling','CarName','fueltype','carbody','wheelbase','carlength','enginetype','cylindernumber','enginesize','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']]

a = df2.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2['CarName'] = le.fit_transform(df2['CarName'])
 

fuelt = le.fit_transform(df2['fueltype'])
df2['fueltype'] = fuelt

Cbody = le.fit_transform(df2['carbody'])
df2['carbody'] = Cbody

Etype = le.fit_transform(df2['enginetype'])
df2['enginetype'] = Etype


CyNo = le.fit_transform(df2['cylindernumber'])
df2['cylindernumber'] = CyNo


x = df2[['symboling','CarName','fueltype','carbody','wheelbase','carlength','enginetype','cylindernumber','enginesize','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg']]
y = df2['price']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=0)


from sklearn.linear_model import LinearRegression
rg = LinearRegression()
rg.fit(x_train,y_train)



y_pred = rg.predict(x_test)


rg.score(x_test,y_test)

rg.predict([[3,4,1,2,105.8,192.7,3,2,130,2.68,7,111,5500,19,22,]])
