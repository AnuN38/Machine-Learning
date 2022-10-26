# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:27:50 2022

@author: anuna
"""

import pandas as pd


df = pd.read_csv("carprices.csv")

a = df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

CarModel = le.fit_transform(df['Car Model'])
df['Car Model'] = CarModel



x = df[['Car Model','Mileage','Age(yrs)']] 
y = df['Sell Price($)']



from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)


reg.score(x,y)

reg.predict([[2,45000,7]])




