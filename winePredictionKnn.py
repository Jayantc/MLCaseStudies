from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

wineData= load_wine()
print(wineData.data)
print(wineData.target)

X= wineData.data
y= wineData.target
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=1)

knn= KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

predict= knn.predict(X_test)

print('Accuracy is : ', accuracy_score(y_test, predict))