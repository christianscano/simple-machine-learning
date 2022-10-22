import pandas as pd
import numpy as np

class LoaderMNISTData:

    def __init__(self, filename='./data/mnist_train_small.csv'):
        self.filename = filename

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not isinstance(value, str):
            raise TypeError("Filename is not a string!")
        # else
        self._filename = value

    def load_data(self):
        data = pd.read_csv(self._filename)
        data = np.array(data)  # cast pandas dataframe to numpy array
       
        y = data[:, 0]
        X = data[:, 1:] / 255
        return X, y 