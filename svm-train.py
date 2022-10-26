# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:54:41 2022

@author: anuna
"""

import pandas as pd

df = pd.read_csv("train (1).csv")
test = pd.read_csv("test (1).csv")

a = df.isnull().sum()

x=df.iloc[:,0:20].values
y=df.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X= sc.fit_transform(x)  


#splitting into training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=0)


from sklearn.svm import SVC
sv = SVC(kernel='linear',random_state=0)
sv.fit(x_train,y_train)

y_pred = sv.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report  
cm= confusion_matrix(y_test, y_pred) 
accuracy_score(y_test,y_pred)
cr = classification_report(y_test,y_pred) 
