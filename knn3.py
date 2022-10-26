from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

k_range=list(range(1,16))
scores=[]
for k in k_range:
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	predictions=knn.predict(X_test)
	
	a=accuracy_score(y_test,predictions)
	scores.append(a)
	print(confusion_matrix(y_test,predictions))

plt.plot(k_range,scores)
plt.xlabel("Values of k")
plt.ylabel("Accuracy Scores")
plt.show()

