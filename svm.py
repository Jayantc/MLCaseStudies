from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

data= load_breast_cancer()
xtrain, xtest, ytrain, ytest= train_test_split(data.data, data.target, random_state=2, test_size=0.2)

model= SVC() # gives accuracy 0.9035087719298246
model1= SVC(kernel='linear', C=1) # gives accuracy 0.9473684210526315
model2= SVC(kernel='rbf', C=1) # gives accuracy 0.9035087719298246
model3= SVC(kernel='poly', C=1) # gives accuracy 0.8947368421052632
model4= LinearSVC(C=1) # gives accuracy 9122807017543859
model1.fit(xtrain, ytrain)
predict= model1.predict(xtest)
print(accuracy_score(ytest, predict))