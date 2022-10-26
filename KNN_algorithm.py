#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 19:32:11 2021

@author: irfana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Social_Network_Ads.csv")


x=df.iloc[:,2:4].values
y=df.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X= sc.fit_transform(x)  


#splitting into training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=0)

#Fitting K-NN classifier to the training set  


#how to choose value of k
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score
acc =[]
for i in range(1,11):
    classifier= KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2 )  
    classifier.fit(x_train, y_train)
    #Predicting the test set result  
    y_pred= classifier.predict(x_test) 
    a=accuracy_score(y_test,y_pred)
    acc.append(a)

    
plt.plot(range(1,11),acc)    
    







#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report  
cm= confusion_matrix(y_test, y_pred) 
accuracy_score(y_test,y_pred)
cr = classification_report(y_test,y_pred) 



#visualisation of actual data set


plt.scatter(x=X[y==0,0],y=X[y==0,1],color="red")
plt.scatter(x=X[y==1,0],y=X[y==1,1],color="green")
plt.show()




#Visulaizing the testing set result  
from matplotlib.colors import ListedColormap  

x_set, y_set = x_test, y_test  

x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1, stop = x_set[:,0].max()+1, step=0.01),  
                     np.arange(start = x_set[:,1].min()-1, stop = x_set[:1].max()+1, step=0.01))  

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('red','green' )))  

plt.xlim(x1.min(), x1.max()) 
 
plt.ylim(x2.min(), x2.max())  

for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('red', 'green'))(i), label = j)  
    
    
plt.title('K-NN Algorithm (Testing set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated salary')  
plt.legend()  
plt.show()  







#Visulaizing the trianing set result  
from matplotlib.colors import ListedColormap  

x_set, y_set = x_train, y_train  

x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.5, cmap = ListedColormap(('red','green' )))  

plt.xlim(x1.min(), x1.max()) 
 
plt.ylim(x2.min(), x2.max())  

for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('red', 'green'))(i), label = j)  
    
    
plt.title('K-NN Algorithm (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated salary')  
plt.legend()  
plt.show()  




