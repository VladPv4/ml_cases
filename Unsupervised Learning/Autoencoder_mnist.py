import numpy as np
from sklearn.neural_network import MLPRegressor
from util import getKaggleMNIST

X, _, Xt, _ = getKaggleMNIST()

model = MLPRegressor()
model.fit(X, X)

print('Train R^2:', model.score(X, X))
print('Test R^2:', model.score(Xt, Xt))

Xhat = model.predict(X)
mse = ((Xhat - X)**2).mean()
print('Train MSE:', mse)

Xhat = model.predict(Xt)
mse = ((Xhat - Xt)**2).mean()
print('Test MSE:', mse)