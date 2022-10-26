# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:09:23 2022

@author: anuna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Titanic.csv')
df


df1 = df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


a = df1.isnull().sum()

df1['Age'] = df1['Age'].fillna(df1['Age'].mean())
df1['Age'].isnull().sum()


df1['Embarked'] = df1['Embarked'].ffill()
df1['Embarked'].isnull().sum()

df1.dtypes

le = LabelEncoder()
df1['Embarked'] = le.fit_transform(df1['Embarked'])
df1['Sex'] = le.fit_transform(df1['Sex'])


x = df1.iloc[:,1:8].values
y = df1.iloc[:,0].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


reg = LinearRegression()
reg.fit(x_train,y_train)


y_pred = reg.predict(x_test)
x_pred= reg.predict(x_train)


plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,x_pred,color="blue")
plt.show()






