# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:40:03 2022

@author: anuna
"""

import pandas as pd
df = pd.read_csv("framingham.csv")


a = df.isnull().sum()


df['education'] = df['education'].bfill()



df['BPMeds'] = df['BPMeds'].ffill()



df['totChol'] = df['totChol'].fillna(df['totChol'].mean())



df = df.drop(['glucose'], axis = 1)



df = df.dropna()
b = df.isnull().sum()


x = df.iloc[:,1:14].values
y = df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.1,random_state=0)


from sklearn.linear_model import LogisticRegression
mlr = LogisticRegression()
mlr.fit(x_train,y_train)

y_pred = mlr.predict(x_test)


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
accuracy_score(y_test,y_pred)



import matplotlib.pyplot as mtp
import numpy as nm
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2,mlr.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75, cmap = ListedColormap(('purple','green')))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Logistic Regression (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()




from sklearn.metrics import roc_auc_score,roc_curve,auc
fpr,tpr,threshold = roc_curve(y_test,y_pred)
a = auc(fpr,tpr)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr,color="green",label=("AUC value : %0.2f"%(a)))
plt.plot([0,1],[0,1],"--",color="red")
plt.xlabel("False positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="best")
plt.show()


import pickle
f1 = open(file = 'logistimodel1.pkl',mode = 'bw')
pickle.dump(mlr,f1)
f1.close()
