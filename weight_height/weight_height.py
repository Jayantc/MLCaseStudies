import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

data= pd.read_csv('weight-height.csv')
print(data.head())
x= data[['Height']]
y= data[['Weight']]

model= LinearRegression()
model.fit(x, y)

pred= model.predict(x)

plt.scatter(x, y)
plt.plot(x, pred, 'r', linewidth=2)
plt.title('Height vs Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

weight= int(input('Enter your height : '))
height= model.predict([[weight]])
print('According to linear regression & our dataset, your height is ', height)