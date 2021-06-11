from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score

data= load_iris()

#Normal Code
xtrain, xtest, ytrain, ytest= train_test_split(data.data, data.target, test_size=0.4)
clf= svm.SVC(kernel='linear', C=1).fit(xtrain, ytrain)
print(clf.score(xtest, ytest))

#K-FOLDING CODE
scores= cross_val_score(clf, data.data, data.target, cv=5)
print(scores)
print(scores.mean())

#FOR POLY
clf2= svm.SVC(kernel='poly', C=1).fit(xtrain, ytrain)
print(clf2.score(xtest, ytest))
scores= cross_val_score(clf2, data.data, data.target, cv=5)
print(scores)
print(scores.mean())