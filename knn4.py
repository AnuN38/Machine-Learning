from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd

train1=pd.read_csv('train.csv',delimiter='\t')
test1=pd.read_csv('test.csv',delimiter='\t')

train=train1.to_numpy()
test=test1.to_numpy()


print(train[:,[0,1]])
X_train,X_test,y_train,y_test=train[:,[0,1]],test[:,[0,1]],train[:,2],test[:,2]

print(X_test.shape)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
predictions=knn.predict(X_test)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))


