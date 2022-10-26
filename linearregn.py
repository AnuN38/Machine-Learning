# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Salary_Data.csv')
df.head(5)
df.describe()
a = df.isnull().sum()


x= df.iloc[:, :-1].values  
y= df.iloc[:, 1].values 
  
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred= reg.predict(x_test)  
x_pred= reg.predict(x_train)  

plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,x_pred,color="red")
plt.title("Salary vs Experience (Training Dataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary(In Rupees)")
plt.show()


plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="green")
plt.title("Salary vs Experience (Test Dataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary(In Rupees)")
plt.show()