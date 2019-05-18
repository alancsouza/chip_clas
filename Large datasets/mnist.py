
## Comparing nn-clas and par-clas in mnist dataset

from keras.datasets import mnist
from functions import *

data_name = "mnist"

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# re-scale the image data to values between (0.0,1.0]
X_train = X_train.astype('float32') / 255.
X_train = X_train.astype('float32') / 255.

# Converting to binary classification problem, odd digits or even digits
y_train_binary = (y_train % 2 == 0).astype(np.int)
y_test_binary = (y_test % 2 == 0).astype(np.int)

# Filtering data:
X_new, y_new = remove_noise(X_train, y_train_binary)

# Comparing methods:
method = ["nn_clas", "parallel", "pseudo_support_edges"]

print("Dataset: {}".format(data_name))

for model in method:
    y_hat, y_test, result, runtime = chip_clas(X_new, y_new, method = model, kfold = 1)

    mean_AUC = result.mean()
    std = result.std()

    print(" \n Method: {0} \n Avarege AUC: {1:.4f} \n Std. Deviation {2:.4f} \n Avarege Runtime: {3:.4f} \n".format(model, mean_AUC[0], std[0], runtime.mean()[0]))

    f = open("results.txt", "a+")
    f.write("Dataset: %d \n", data_name)



