import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

data= pd.read_csv("PastHires.csv")
#print(data)

# data Pre-processing
d={"Y":1, "N":0}
data['Hired']= data['Hired'].map(d)
data['Interned']= data['Interned'].map(d)
data['Top-tier school']= data['Top-tier school'].map(d)
data['Employed?']= data['Employed?'].map(d)
d={"BS":0, "MS": 1, "PhD":2}
data['Level of Education']= data['Level of Education'].map(d)
#print(data)

x= data[list(data.columns[:6])]
y= data["Hired"]

clf= DecisionTreeClassifier()
clf.fit(x, y)

clf2= RandomForestClassifier(n_estimators=10)
clf2.fit(x, y)
print(clf2.predict(np.array([10, 1, 4, 0, 0, 0]).reshape(1, -1)))