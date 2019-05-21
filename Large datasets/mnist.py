## Comparing nn-clas and par-clas in mnist dataset

from keras.datasets import mnist
from chip_clas_new import chip_clas_new
from functions import remove_noise
import numpy as np
import pandas as pd

data_name = "mnist"

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping the data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Small data due to memory error
X_train = X_train[:10000,:]
X_test = X_test[:1000,:]

# re-scale the image data to values between (0.0,1.0]
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Converting to binary classification problem, odd digits or even digits
y_train_binary = (y_train % 2 == 0).astype(np.int)
y_test_binary = (y_test % 2 == 0).astype(np.int)

# Converting the numpy array to pandas data frame
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

y_train_binary = pd.DataFrame(y_train_binary)
y_test_binary = pd.DataFrame(y_test_binary)

# Filtering data:
X_train_new, y_train_new = remove_noise(X_train, y_train_binary)

# Comparing methods:
method = ["nn_clas", "parallel", "pseudo_support_edges"]

print("Dataset: {}".format(data_name))

for model in method:
    y_hat, y_test, result, runtime = chip_clas_new(X_train_new, X_test, y_train_new, y_test_binary, method = model)

    print(" \n Method: {0} \n AUC: {1:.4f} \n Runtime: {2:.4f} \n".format(model, result, runtime))

    f = open("results.txt", "a+")
    f.write("Dataset: %d \n", data_name)



