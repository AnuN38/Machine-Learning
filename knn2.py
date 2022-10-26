from sklearn.datasets import load_iris
iris =load_iris()
print(iris.feature_names)
print(iris.data)
X=iris.data
y=iris.target

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=12)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


knn.fit(X_train,y_train)
predictions=knn.predict(X_test)

print(predictions)
print(y_test)

from sklearn.metrics import accuracy_score,confusion_matrix



print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))


