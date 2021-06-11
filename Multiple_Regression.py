import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

data= pd.read_csv("cars.csv")

df1= data[["Mileage", "Price"]]
bins= np.arange(0, 50000, 10000)
groups= df1.groupby(pd.cut(df1["Mileage"], bins)).mean()
print(groups)
plt.plot(groups["Price"])
plt.show()

scale= StandardScaler()
x= data[['Mileage', 'Cylinder', 'Doors']]
y= data['Price']

x[['Mileage', 'Cylinder', 'Doors']]= scale.fit_transform(x[['Mileage', 'Cylinder', 'Doors']].values)

#add constant column to our model so we have y-intercept
x= sm.add_constant(x)
print(x)

est= sm.OLS(y, x).fit()
print(est.summary()) #gives much information on equation created

scaled=scale.transform([[45000, 8, 4]])
scaled= np.insert(scaled[0], 0, 1)
print(scaled)
print(est.predict(scaled))
