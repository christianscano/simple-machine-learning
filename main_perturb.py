from data_loader.loader_mnist_data import LoaderMNISTData
from utils import plot_ten_images
from data_pertub.CDataPertubGaussian import CDataPertubGaussian
from data_pertub.CDataPerturbRandom import CDataPerturbRandom
import matplotlib.pyplot as plt


data_loader = LoaderMNISTData()

X, y = data_loader.load_data()
plot_ten_images(X, y)

random_mode = CDataPerturbRandom(min_value=0, max_value=1, k=200)
X_random = random_mode.pertub_dataset(X)

plot_ten_images(X_random, y)

gaussian_mode = CDataPertubGaussian(min_value=0, max_value=1, sigma=0.80)
X_gaussian = gaussian_mode.pertub_dataset(X)

plot_ten_images(X_gaussian, y)

plt.show()