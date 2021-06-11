import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

pageSpeeds= np.random.normal(3, 1, 1000)
purchaseAmt= 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3
plt.scatter(pageSpeeds, purchaseAmt)
plt.show()
print(stats.linregress(pageSpeeds, purchaseAmt))
slop, intercept, r_value, p_value, std_err= stats.linregress(pageSpeeds, purchaseAmt)

def predict(x):
    return slop*x+intercept
fitline= predict(pageSpeeds)

plt.scatter(pageSpeeds, purchaseAmt)
plt.plot(pageSpeeds, fitline, c='r')
plt.show()
