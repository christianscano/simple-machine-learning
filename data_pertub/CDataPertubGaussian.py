from matplotlib import pyplot as plt
from data_pertub.CDataPertub import CDataPertub
import numpy as np
from utils import plot_image

class CDataPertubGaussian(CDataPertub):

    def __init__(self, min_value=0, max_value=255, sigma=100.0):
        self.min_value = min_value
        self.max_value = max_value
        self.sigma = sigma
    
    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        if value >= 0:
            self._min_value = float(value)
        else:
            raise ValueError("Value not valid!")

    @property
    def max_value(self):
        return self._max_value

    @max_value.setter
    def max_value(self, value):
        if value <= 255:
            self._max_value = float(value)
        else: 
            raise NotImplementedError("Value not valid")

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = float(value)

    def data_perturbation(self, x):
        noise = np.random.standard_normal(size=x.size)
        noise = noise * self.sigma

        # Apply the gaussian noise
        x = x + noise

        x[x >= self.max_value] = self.max_value
        x[x <= self.min_value] = self.min_value

        return x
        