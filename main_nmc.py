
from matplotlib.pyplot import plot
from data_loader.loader_mnist_data import LoaderMNISTData
from utils import split_dataset, plot_ten_images, compute_ts_error
from classifiers.nearest_mean_centroid import NearestMeanCentroid
import numpy as np
import matplotlib.pyplot as plt


data_loader = LoaderMNISTData()

X, y = data_loader.load_data()
Xtr, ytr, Xts, yts = split_dataset(X, y, tr_fraction=0.7)

nmc = NearestMeanCentroid()
nmc.fit(Xtr, ytr)

classes = np.unique(ytr).size
plot_ten_images(nmc.centroids, np.array(range(classes)))

yc = nmc.predict(Xts)
test_error = compute_ts_error(yc, yts)

plt.show()