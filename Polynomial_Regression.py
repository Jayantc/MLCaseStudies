import numpy as np
import matplotlib.pyplot as plt
from  sklearn.metrics import r2_score
np.random.seed(4)
pageSpeeds= np.random.normal(3, 1, 1000)
purchaseAmt= np.random.normal(50, 10, 1000) / pageSpeeds
plt.scatter(pageSpeeds, purchaseAmt)
plt.show()

x= np.array(pageSpeeds)
y= np.array(purchaseAmt)
p4= np.poly1d(np.polyfit(x, y, 4))
xp= np.linspace(0, 7, 100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()

print(r2_score(y, p4(x)))
