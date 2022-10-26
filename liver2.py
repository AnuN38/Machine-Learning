#Making your dataset ready for developing ML models
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


import pandas as pd


data=pd.read_csv("liver2.csv")

data1=data.fillna(data.mean()) #to fill na values

data1['Gender']=pd.get_dummies(data1['Gender']) #Encoding


print (data.head())
print (data1.head())



data=data1.to_numpy()

X=data[:,0:10]
y=data[:,-1]

y=y.astype('int') # Convert to discrete values for classification problems.


print(y)
print(X)
print(X.shape)
print(y.shape)

knn=KNeighborsClassifier(n_neighbors=1)

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2)

knn.fit(X_train,y_train)
predictions=knn.predict(X_test)

print(predictions)
print(y_test)

from sklearn.metrics import accuracy_score,confusion_matrix



print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))





