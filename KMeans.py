import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

def createClusteredData(n, k):
    np.random.seed(8)
    pointsPerCluster= float(n)/k
    x=[]
    for i in range(k):
        incomeCentroid= np.random.uniform(20000, 2000000)
        ageCentroid= np.random.uniform(20, 70)
        for j in range(int(pointsPerCluster)):
            x.append([np.random.normal(incomeCentroid, 1000), np.random.normal(ageCentroid, 2)])
    x= np.array(x)
    return x

data= createClusteredData(100, 5)
model= KMeans(n_clusters=5)

#normalizing (scaling) data for good result
model= model.fit(scale(data))

print(model.labels_)
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(np.float))
plt.show()