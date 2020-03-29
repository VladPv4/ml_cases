import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

if __name__ == '__main__':
	centers = np.array([
    [ 1,  1,  1],
    [ 1,  1, -1],
    [ 1, -1,  1],
    [ 1, -1, -1],
    [-1,  1,  1],
    [-1,  1, -1],
    [-1, -1,  1],
    [-1, -1, -1],
  ])*3

	data = []
	pts_per_cloud = 100
	for c in centers:
		cloud = np.random.randn(pts_per_cloud, 3) + c
		data.append(cloud)
	data = np.concatenate(data)

	colors = np.array([[i]*pts_per_cloud for i in range(len(centers))]).flatten()
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(data[:,0], data[:,1], data[:,2], c=colors)
	plt.show()

	tsne = TSNE()
	transformed = tsne.fit_transform(data)

	plt.scatter(transformed[:,0], transformed[:,1], c=colors)
	plt.show()
