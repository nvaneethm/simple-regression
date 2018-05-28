import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#input
x = [[17],[18],[21],[22],[23],[24],[25],[26],[27]]
#target
y = [[79.44], [78.9], [80.73], [81.05], [81.36], [81.67], [82.04], [82.14], [82.13]] 

#crossvalidation/trainigandtestingset
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.4, random_state=101)

#model
reg = LinearRegression()
reg.fit(x,y)

#plotting
plt.scatter(x,y)
plt.plot(x,y,"r")
#prediction
x1 = [[29]]
plt.scatter(xtest, reg.predict(xtest), color="black")
plt.scatter(x1, reg.predict(x1), color="red")
print(reg.predict(x1))
plt.show()
