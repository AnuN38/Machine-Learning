# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:34:19 2022

@author: anuna
"""

import pandas as pd
df = pd.read_csv("Social_Network_Ads.csv")

x=df.iloc[:,2:4].values
y=df.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X= sc.fit_transform(x)  



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=0)


from sklearn.naive_bayes import GaussianNB,BernoulliNB
model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report  
cm= confusion_matrix(y_test, y_pred) 
accuracy_score(y_test,y_pred)
cr = classification_report(y_test,y_pred) 


Bmodel = BernoulliNB()
Bmodel.fit(x_train,y_train)
y_pred2 = Bmodel.predict(x_test)
cm= confusion_matrix(y_test, y_pred) 
accuracy_score(y_test,y_pred2)
cr = classification_report(y_test,y_pred2)
