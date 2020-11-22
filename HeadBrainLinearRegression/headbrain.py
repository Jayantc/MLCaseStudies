from sklearn.linear_model import LinearRegression
import pandas as pd

data= pd.read_csv('HeadBrain.csv')

X= data['Head Size(cm^3)'].values
y= data['Brain Weight(grams)'].values
X=X.reshape(-1,1)

clf= LinearRegression().fit(X, y)
print('Training time accuracy is ', clf.score(X,y))

headSize= int(input('Enter size if your head : '))
print('Weight of your brain is ', clf.predict([[headSize]]))