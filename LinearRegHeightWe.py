# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 10:16:46 2022

@author: anuna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("SOCR-HeightWeight.csv")

a = df.isnull().sum()


x = df.iloc[:,1:2].values
y = df.iloc[:,2].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
x_pred = reg.predict(x_train)


plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,x_pred,color="blue")
plt.title("Height vs Weight (Training set)")
plt.show()


plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,y_pred,color="green")
plt.title("Height vs Weight (Testing set)")
plt.show()


#mse = (sum((y-(y_pred))**2))/len(x)
y=df.iloc[:,2]
x=df.iloc[:,1:2]
ypred=reg.predict(x)
b=y-ypred
sq=b**2
s=sum(sq)
mse = s/len(x)


from sklearn.metrics import mean_squared_error
mse1 = mean_squared_error(y,ypred)
mse2=mean_squared_error(y_test,y_pred)



