import numpy as np
from sklearn.metrics import pairwise_distances


class NearestMeanCentroid:

    def __init__(self):
        self._centroids = None

    @property
    def centroids(self):
        """
        Centroids learnt by the NMC classifier.
        Returns
        -------
        Centroids...
        """
        return self._centroids

    def fit(self, Xtr, ytr):
        """
        Estimate the centroid for each class from the training data
        Parameters
        ----------
        Xtr
            Training data
        ytr
            Labels
        Returns
        -------
        Classifier with estimated centroids.
        """
        labels = np.unique(ytr).size # Number of classes
        num_pixels = Xtr.shape[1]

        self._centroids = np.zeros(shape=(labels, num_pixels))

        for label in range(labels):
            xk = Xtr[ytr == label, :] # Get all images which belong to the 'label' class
            self._centroids[label, :] = np.mean(xk, axis=0)

        return self

    def predict(self, Xts):
        """
        Compute predictions on test data.
        Parameters
        ----------
        Xts
            Test data
        Returns
        -------
        Predicted labels for the test data.
        """
        if self._centroids is None:
            raise ValueError("Centroids not set. Run fit(x,y) first!")

        ########
        # Loop Version, only for understanding the logic behind pairwise_distances
        ########
        # n_samples = xts.shape[0]
        # n_classes = centroids.shape[0]
        # dist = np.zeros(shape=(n_samples, n_classes))
        # ypred = np.zeros(shape=(n_samples, ), dtype='int')

        # for i in range(n_samples):
        #     for k in range(n_classes):
        #         dist[i,k] = np.linalg.norm(xts[i,:]-centroids[k,:], ord=2)
        #     ypred[i] = np.argmin(dist[i,:])

        dist = pairwise_distances(Xts, self._centroids)
        ypred = np.argmin(dist, axis=1)
        return ypred