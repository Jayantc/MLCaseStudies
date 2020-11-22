from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

data= load_breast_cancer()

print(data)

X= data.data
y= data.target

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.5, random_state=1)

model= RandomForestClassifier()

model.fit(X_train, y_train)
pred= model.predict(X_test)

print()
print()

print('Accuracy of random forest algorithm for breast cancer is ', accuracy_score(pred, y_test))
print('Confusion matrix is : ', confusion_matrix(y_test, pred))