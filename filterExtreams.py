import numpy as np
import matplotlib.pyplot as plt
income= np.random.normal(40000, 20000, 1000)
incomes= np.append(income, [10000000, 20000000])
print(incomes.mean()) #mean is getting disturbed due to 2 new data
#68537.73806204788
plt.hist(incomes, 50)
plt.show()

u= incomes.mean()
s= incomes.std()
incomes=[i for i in incomes if(u-2*s < i < u+2*s)]

print(np.mean(incomes)) #this mean is perfect
#38674.81353817198

plt.hist(incomes, 50)
plt.show()
