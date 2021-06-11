from scipy.stats import norm, mode, expon, binom, poisson, skew, kurtosis
import matplotlib.pyplot as plt
import numpy as np

### Basic Statistics ###
a=[4, 2, 8, 5, 3, 6, 1, 2, 4, 3, 6, 5, 2, 9, 8, 5, 9, 4, 2, 1]
b=[4, 7, 3, 6, 1, 5, 9, 5, 3, 7, 9, 4, 1, 7, 9, 8, 4, 2, 4, 4]
print(np.mean(a))
print(np.median(a))
print(mode(a)) #imported from scipy
print(np.var(a))
print(np.std(a))
print(np.cov(a))
print(np.cov(a, b))
print(np.corrcoef(a))
print(np.corrcoef(a, b))

###Basic Probability ###

## use proper pmf or pdf according to input data & output required ###

#Probability density function
a= np.arange(-3, 3, 0.001)
plt.plot(a, norm.pdf(a))
plt.plot(a, norm.pdf(a, 1, 0.5))
# 1 is mean, 0.5 is sd
plt.show()

#Exponential
a= np.arange(0, 10, 0.001)
plt.plot(a, expon.pdf(a)) #pdf is probability distribution fun
plt.show()

#Probability mass function
mu= 5 #mean
sigma= 2 #Standard Deviation
values= np.random.normal(mu, sigma, 10000)
plt.hist(values, 50)
plt.show()

#Binomial probability mass function
n, p= 10, 0.5
#n is no. of times experiment runs, p is probability of one outcome
a= np.arange(0, 10, 0.001)
plt.plot(a, binom.pmf(a, n, p)) #pmf is probability mass fun
plt.show()

#Poisson probability mass function
# eg. My Website gets 500 visits avg per day, what is odds of getting 550??
mu= 500 #mean
a= np.arange(400, 600, 0.5)
plt.plot(a, poisson.pmf(a, mu))
plt.show()

# Percentile & Moments
vals= np.random.normal(0, 0.5, 1000)
print(np.percentile(vals, 90)) #value at 90 percentile
print(np.percentile(vals, 50))
print(np.mean(vals), np.var(vals), skew(vals), kurtosis(vals))
