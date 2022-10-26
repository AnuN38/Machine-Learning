# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:13:06 2022

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


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report  
cm= confusion_matrix(y_test, y_pred) 
accuracy_score(y_test,y_pred)
cr = classification_report(y_test,y_pred) 


model2 = DecisionTreeClassifier(criterion='gini', random_state=0)
model2.fit(x_train,y_train)

y_pred2 = model2.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report  
cm= confusion_matrix(y_test, y_pred2) 
accuracy_score(y_test,y_pred2)
cr = classification_report(y_test,y_pred2) 


#tree visualization

from sklearn.tree import export_text
tree = export_text(model,feature_names=['age','salary'])
