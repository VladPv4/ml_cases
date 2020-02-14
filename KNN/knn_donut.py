from KNN import KNN
from utils import get_donut
import matplotlib.pyplot as plt

if __name__ == '__main__':
	X, Y = get_donut()

	plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
	plt.show()

	for i in range(20):
		model = KNN(i+1)
		model.fit(X,Y)
		print('Neighbours:',i+1,'Train accuracy:', model.score(X,Y))