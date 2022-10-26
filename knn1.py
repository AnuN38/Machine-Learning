from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report

iris=load_iris()
X=iris.data
y=iris.target

print(iris.feature_names)
print(X.shape)
print(y.shape)




knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)

print(knn.predict([[3,5,4,2]]))



p=knn.predict(X)
print(p)
print(y)

print(confusion_matrix(y,p))

print(classification_report(y,p))

