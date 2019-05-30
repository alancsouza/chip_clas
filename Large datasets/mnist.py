## Comparing nn-clas and par-clas in mnist dataset

from keras.datasets import mnist
from chip_clas_new import chip_clas_new
#from functions import remove_noise
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

data_name = "mnist"

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# re-scale the image data to values between (0.0,1.0]
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Reshaping the data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# making binary classification problem: 2 vs rest
y_train = (y_train % 2 == 0).astype(np.int)
y_test = (y_test % 2 == 0).astype(np.int)

# Dimensionality reduction using PCA
pca_train = PCA(n_components=16, svd_solver='randomized', whiten=True).fit(X_train)
pca_test = PCA(n_components=16, svd_solver='randomized', whiten=True).fit(X_test)

X_train_pca = pca_train.transform(X_train)
X_test_pca = pca_test.transform(X_test)

X_train = pd.DataFrame(X_train_pca)
X_test = pd.DataFrame(X_test_pca)

# Filtering data:
#X_new, y_new = remove_noise(X, y)

# Comparing methods:
method = ["parallel", "nn_clas", "pseudo_support_edges"]

print("Dataset: {}".format(data_name))

for model in method:
    y_hat, y_test, result, runtime = chip_clas_new(X_train, X_test, y_train, y_test, method = model)

    print(" \n Method: {0} \n AUC: {1:.4f} \n Runtime: {2:.4f} \n".format(model, result, runtime))

    f = open("results_mnist.txt", "a+")
    f.write("\n\nMétodo: %s \n" % model)
    f.write("AUC: %d \n" % result)
    f.write("Runtime: %d \n" % runtime)
    f.close()