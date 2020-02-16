import requests
import numpy as np
import matplotlib.pyplot as plt
from utils import get_data

X, Y = get_data()
N = len(Y)
while True:
	i = np.random.choice(N)
	r = requests.post('http://localhost:8887/predict', data={'input': X[i]})
	print('Response:')
	j = r.json()
	print(j)
	print('Target:', Y[i])

	plt.imshow(X[i].reshape(28,28), cmap='gray')
	plt.title('Target: %d, Prediction: %d' % (Y[i], j['prediction']))
	plt.show()

	response = input('Continue? (Y/N)\n')
	if response in ('n','N'):
		break
