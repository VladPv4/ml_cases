#dataset: https://archive.ics.uci.edu/ml/datasets/Spambase
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

data = pd.read_csv('spambase.data').values
np.random.shuffle(data)

X = data[:,:48]
Y = data[:,-1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print('Classification Rate for NB:', model.score(Xtest, Ytest))

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print('Classification Rate for AdaBoost:', model.score(Xtest, Ytest))