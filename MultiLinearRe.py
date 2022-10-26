# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:37:13 2022

@author: anuna
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv("50_Startups.csv")


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['State'] = le.fit_transform(df['State'])


m = df.corr()

x = df.iloc[:,:-1]
y = df.iloc[:,-1]


np.std(x)


mlr = LinearRegression()

mlr.fit(x,y)

y_pred = mlr.predict(x)


for i in range(50):
    print("Actual point %0.3f & Prediction point %0.3f"%(y[i],y_pred[i]))
    

from sklearn.metrics import r2_score
r2_score(y,y_pred)


df.head(5)


rd_spend = 200000
Admin = 180000
mr_spend = 1000000
state = "California"

data = {"R&D Spend":rd_spend,"Administration":Admin,"Marketing Spend":mr_spend,"State":state}

new_data = pd.DataFrame(data,index=[0])

new_data['State'] = le.transform(new_data['State'])
new_data

a = mlr.predict(new_data)

print("According to Multiple linear Rergression model profit is %0.3f"%(a))
