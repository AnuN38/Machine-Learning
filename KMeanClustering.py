# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:54:22 2022

@author: anuna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#reading data set
df= pd.read_csv("Mall_Customers.csv")


x=df.iloc[:,3:5].values

#visualisation of data

plt.scatter(x[:,0],x[:,1])
plt.title("K-mean visualisation")
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.show()


#apply k-mean clusturing



from sklearn.cluster import KMeans
#HOW TO CHOOSE NUMBER OF K
wcss = []

for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(x)
    a = km.inertia_
    wcss.append(a)


plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.xlabel("number of K")
plt.ylabel("WCSS")
plt.show()

#After Elbow method we found best number of Ki 5


from sklearn.cluster import KMeans
km=KMeans(n_clusters=5)
km.fit(x)


#prediction
y = km.predict(x)

set(y)

#Visualization of clusterd data

plt.scatter(x[y==0,0],x[y==0,1],color="red",label="1st Cluster")
plt.scatter(x[y==1,0],x[y==1,1],color="blue",label="2nd Cluster")
plt.scatter(x[y==2,0],x[y==2,1],color="green",label="3rd Cluster")
plt.scatter(x[y==3,0],x[y==3,1],color="yellow",label="4th Cluster")
plt.scatter(x[y==4,0],x[y==4,1],color="orange",label="5th Cluster")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="black",
            label="Centeroid",s=100)
plt.title("K-mean visualisation")
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.legend()
plt.show()

# applying KNN algorithm
from sklearn.cluster import KMeans  
classifier= KMeans(n_clusters=5)  
classifier.fit(x)

y1=classifier.predict(x)


plt.scatter(x[y1==0,0],x[y1==0,1],color="red",label="1st Cluster")
plt.scatter(x[y1==1,0],x[y1==1,1],color="blue",label="2nd Cluster")
plt.scatter(x[y1==2,0],x[y1==2,1],color="green",label="3rd Cluster")
plt.scatter(x[y1==3,0],x[y1==3,1],color="yellow",label="4th Cluster")
plt.scatter(x[y1==4,0],x[y1==4,1],color="orange",label="5th Cluster")
plt.scatter(classifier.cluster_centers_[:,0],classifier.cluster_centers_[:,1],color="black",
            label="Centeroid",s=100)
plt.title("K-mean visualisation")
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.legend()
plt.show()

#new user input



def prediction(a,b):

    d={"Annual income":a,"Spending Score":b}
    new_data = pd.DataFrame(d,index=[0])
   
   
    result = classifier.predict(new_data)
    result = int(result)
   
   
    if result==0:
        print("This customer from 1st clsuter (high income & Low spending score)")
       
       
    elif result==1:
        print("This customer from 2nd clsuter (low income & Low spending score)")
       
       
    elif result==2:
        print("This customer from 3rd clsuter (High income & High spending score)")
       
       
    elif result==3:
        print("This customer from 4th clsuter (avg income & avg spending score)")
       
       
    elif result==4:
        print("This customer from 5th clsuter (low income & High spending score)")
       
       
       
       
a =int(input("Enter Annual Income (k$)"))
b = int(input("Enter spending score in mall: "))


prediction(a,b)
