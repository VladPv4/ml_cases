import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#load the data
X = []
Y = []
for line in open('data_2d.csv'):
	x1, x2, y = line.split(',')
	X.append([float(x1), float(x2),1])
	Y.append(float(y))

#Turn data to numpy arrays - it is faster
X = np.array(X)
Y = np.array(Y)

#plot data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

#calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# compute r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print('r-squared: ', r2)
