# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:18:04 2022

@author: anuna
"""

from sklearn.datasets import load_iris


iris=load_iris()
X=iris.data
y=iris.target

print(iris.feature_names)
print(X.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=0)


from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier(n_estimators= 10, criterion="entropy")
rfc1.fit(x_train,y_train)


y_pred = rfc1.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report  
cm= confusion_matrix(y_test, y_pred) 
accuracy_score(y_test,y_pred)
cr = classification_report(y_test,y_pred)
