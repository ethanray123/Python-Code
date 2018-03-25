from sklearn import linear_model
import csv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

def summation(arr):
    s = 0
    for x in arr:
        s += x
    return s

def regression(arr,xVals,slope,intercept):
    i=0
    for x in xVals:
        arr[i] = slope*x + intercept
        i += 1


dataset = pd.read_csv("C:\\Users\\Ethan Ray Mosqueda\\PycharmProjects\\EMain\\LinearRegression\\data.csv")
print("Dataset: ")
print(dataset)
x = np.array(dataset['x'])
y = np.array(dataset['y'])

sumX = summation(x)
sumY = summation(y)

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

print("x bar: ", sumX)
print("y bar: ", sumY)
print("regression model: f(x) = ",slope,"x + ",intercept)
yhat = np.array(len(y));
yhat = regression(yhat,x,slope,intercept)
print("yhat: ")
print(yhat)

plt.scatter(x, y, color="black")
plt.plot(x, x*slope+intercept, 'blue')
plt.xlabel("Lot Size")
plt.ylabel("Man Hours")
plt.axis([0, 100, 0, 200])
plt.show()

#PREDICTION
#regression = linear_model.LinearRegression()
#regression.fit(x,y)

#a = regression.intercept_
#b = regression.coef_[0]

#print("Coefficients: this is the y-intercept b0 = ",a[0])
#print("Coefficients: this is the slope b1 = ",b[0])
#print("f(x) = ",a[0]," + ",b[0],"x")