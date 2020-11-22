from sklearn import datasets
import numpy as np
from sklearn.tree import DecisionTreeClassifier
data= datasets.load_iris()
print('Features of Iris')
print(data.data)
print('Targets of iris')
print(data.target)

index=[1,5,51,56,101,120]
X_train= np.delete(data.data, index, axis=0)
y_train= np.delete(data.target, index)

X_test= data.data[index]
y_test= data.target[index]

clf= DecisionTreeClassifier().fit(X_train, y_train)

print('Values kept out for test')
print(y_test)
print('Predicted outputs for above')
print(clf.predict(X_test))