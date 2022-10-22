from data_pertub.CDataPertub import CDataPertub 
import numpy as np


class CDataPerturbRandom(CDataPertub):

    def __init__(self, min_value=0, max_value=255, k=100):
        self.min_value = min_value
        self.max_value = max_value
        self.k = k

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
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = int(value)

    def data_perturbation(self, x):
        k = min(self.k, x.size)
        idx = np.array(range(x.size))
        np.random.shuffle(idx)
        
        x[idx[:k]] = self.max_value * np.random.rand(k) + self.min_value
    
        return x
    

