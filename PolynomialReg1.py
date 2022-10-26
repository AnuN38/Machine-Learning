# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 11:17:23 2022

@author: anuna
"""

import pandas as pd
import numpy as np

age = np.array([20,21,23,24,45,53,32,28,30,43])
glucose = np.array([70,78,56,45,77,90,98,99,88,79])

age = age.reshape(-1,1)


from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()

lr_model.fit(age,glucose)

y_pred = lr_model.predict(age)

import matplotlib.pyplot as plt
plt.scatter(age,glucose,color="green")
plt.plot(age,y_pred,color="red")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=5)
poly_x = poly.fit_transform(age)

lr_model.fit(poly_x,glucose)

y_pred1 = lr_model.predict(poly_x)


from sklearn.metrics import r2_score
r = r2_score(glucose,y_pred1)

rl = r2_score(glucose,y_pred)
