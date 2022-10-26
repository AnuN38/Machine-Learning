# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:02:49 2022

@author: anuna
"""

import pandas as pd
import numpy as np

df = pd.read_csv("Social_Network_Ads.csv")

a = df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
gen = le.fit_transform(df['Gender'])
df['Gender'] = gen


x = df.iloc[:,2:4].values
y = df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

df.describe

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x_train = st.fit_transform(x_train)
x_test = st.transform(x_test)



from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)


lr.score(x_train,y_train)


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
accuracy_score(y_test,y_pred)



#training 
import matplotlib.pyplot as mtp
import numpy as nm
from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2,lr.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75, cmap = ListedColormap(('purple','green')))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Logistic Regression (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()


#testing

x_set, y_set = x_test, y_test  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2,lr.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75, cmap = ListedColormap(('purple','green')))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Logistic Regression (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()




#Visualizing the training set results
'''import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train


plt.scatter(x=x_set[y_set==0,0],y=x_set[y_set==0,1],color="red",label="Not purchased")
plt.scatter(x=x_set[y_set==1,0],y=x_set[y_set==1,1],color="green",label="Purchased")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Graph between Age & Estimated Salary")
plt.legend()

X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:,0].max()+1, step = 0.01),np.arange(start = x_set[:, 1].min()-1, stop = x_set[:,1].max()+1, step = 0.01))

plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.5, cmap = ListedColormap(('red', 'green')))


plt.show()'''
 

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


#model saving model
import pickle
f1 = open(file = 'logistimodel1.pkl',mode = 'bw')
pickle.dump(lr,f1)
f1.close()