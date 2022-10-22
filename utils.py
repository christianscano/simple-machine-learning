import numpy as np
import matplotlib.pyplot as plt

######
# Plot Functions
######

def plot_image(x, shape=(28,28)):
    """
    Plots an image along with a title
    Parameters
    ----------
    x: the image given as a flat vector of size: (h*w, )
    shape: a tuple (image height, image width)
    Returns
    -------
    None.
    """
    plt.imshow(np.reshape(x, newshape=shape), cmap='gray')


def plot_ten_images(X, y=None):
    """
    Display ten images given as rows in x in a 2x5 subplot,
    along with their titles (passed as a list of 10 strings)
    """
    plt.figure(figsize=(10,5))

    for i in range(10):
        plt.subplot(2, 5, i+1)
        
        plot_image(X[i, :])

        if y is not None:
            plt.title('Label: ' + str(y[i]))


######
#
######

def split_dataset(X, y, tr_fraction=0.5):
    """
    Split the data X,y into two random subsets
    """
    num_samples = y.size
    # Number of elements for the training set
    num_tr = int(tr_fraction * num_samples)
    
    idx = np.array(range(0, num_samples))
    np.random.shuffle(idx)

    # Create Training Set
    ytr = y[idx[0:num_tr]]
    Xtr = X[idx[0:num_tr], :]

    # Create Test Set
    yts = y[idx[num_tr:]]
    Xts = X[idx[num_tr:], :]

    return Xtr, ytr, Xts, yts


def compute_ts_error(ypred, yts):
    """
    Compute the fraction of elements that are different in ypred and yts
    (classification errors)
    Parameters
    ----------
    ypred: the set of predicted class labels
    yts: the true labels of test samples
    Returns
    -------
    test_error: the classification error
    """
    test_error = 100 * (ypred != yts).mean()
    print(f"Test error (on {yts.size} test samples): {test_error:.2f}%")
    return test_error


def count_samples_per_class(y):
    """
    Count the number of elements in each class
    Parameters
    ----------
    y : ndarray
        the labels of each sample.
    Returns
    -------
    v : ndarray
        the number of elements in each class.
    """
    classes = np.unique(y)
    num_classes = classes.size  # number of unique elements in y
    p = np.zeros(shape=(num_classes,))

    for k in range(num_classes):
        p[k] = np.sum(y == classes[k])

    return p