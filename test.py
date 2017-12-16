import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import hdbscan
import dbcv_hdbscan
from scipy.spatial.distance import euclidean
from DBCV import DBCV

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}

moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)

test_data = np.vstack([moons, blobs])

# plt.scatter(test_data.T[0], test_data.T[1], color='r', **plot_kwds)
# plt.show()
### load iris
# test_data = data.load_iris()['data'][:, :]
# load triange
# test_data = np.genfromtxt(fname='triangle-trimmed.csv', delimiter=',')[:, 0:2]
# print (test_data)
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)
condensed_tree = clusterer.condensed_tree_.to_numpy()
print (condensed_tree)
clusterer.condensed_tree_.plot()
plt.show()

print ('hdbscan labels...')
hdbscan_lables = clusterer.fit_predict(test_data)
print ('hdbscan score...')
hdbscan_score = DBCV(test_data, hdbscan_lables, dist_function=euclidean)
print ('HDBScan: score: {}, labels: {}'.format(hdbscan_score, hdbscan_lables))

dbcv_hdbscan.extract_cluster(condensed_tree, test_data)