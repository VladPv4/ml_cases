import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('data_1d.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))
    
X = np.array(X)
Y = np.array(Y)

# plt.scatter(X,Y)
# plt.title('X,Y data')
# plt.show

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean()* X.dot(X) - X.mean() * X.dot(Y)) / denominator

#Linear Regression
Yhat = a*X + b

plt.scatter(X,Y)
plt.plot(X, Yhat, 'r')
plt.title('Linear Regression Line. Slope is: ' + str(round(a)) + '. Intercept: ' + str(round(b)))
plt.show()