from numpy import random
random.seed(0)

total= {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
purchases= {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
totalPurchases= 0
for i in range(100000):
    ageDecade= random.choice([20, 30, 40, 50, 60, 70])
    purchaseProbability= float(ageDecade)/100
    total[ageDecade] += 1
    if(random.random()< purchaseProbability):
        totalPurchases += 1
        purchases[ageDecade] += 1
print(total, purchases, totalPurchases)

# P(E/F) E is purchase, F is you are under 30
PEF= float(purchases[30])/float(total[30])
print("P(Purchase / 30s) is "+ str(PEF))
PF= float(total[30])/100000
print("P(30's) is "+ str(PF))
PE= float(totalPurchases)/100000
print("P(purchase) is "+ str(PE))
#bellow is you are 30 & buying someting(no conditional)
print("P(30s, purchase) is "+ str(float(purchases[30])/100000))
print("P(30s)*P(Purchase) is "+ str(PE*PF))
print((purchases[30]/100000)/PF) #same as P(Purchase / 30s)
