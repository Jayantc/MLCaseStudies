from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

data= load_iris()
xtrain, xtest, ytrain, ytest= train_test_split(data.data, data.target, random_state=2, test_size=0.2)

train= xgb.DMatrix(xtrain, label= ytrain)
test= xgb.DMatrix(xtest, lable= ytest)

param={
    'max_depth': 4,
    'eta': 0.3,
    'Objective': 'multi:softmax',
    'num_class': 3
}
epochs= 10

model= xgb.train(param, train, epochs)
prediction= model.predict(test)
print(prediction)

print(accuracy_score(ytest, prediction))