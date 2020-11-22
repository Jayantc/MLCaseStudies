from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def decisionTreeClassifier():
    iris= load_iris()
    data= iris.data
    target= iris.target
    X_train, X_test, y_train, y_test= train_test_split(data, target, test_size=0.3, random_state=1)
    clf= DecisionTreeClassifier().fit(X_train, y_train)
    predictions= clf.predict(X_test)
    return (accuracy_score(y_test, predictions))

def knnClassifier():
    iris= load_iris()
    data= iris.data
    target= iris.target
    X_train, X_test, y_train, y_test= train_test_split(data, target, test_size=0.3, random_state=1)
    clf= KNeighborsClassifier().fit(X_train, y_train)
    predictions= clf.predict(X_test)
    return (accuracy_score(y_test, predictions))

print('Accuracy score of Decision tree classifier for iris dataset is ', decisionTreeClassifier())
print('Accuracy score of KNeighbors classifier for iris dataset is ', knnClassifier())