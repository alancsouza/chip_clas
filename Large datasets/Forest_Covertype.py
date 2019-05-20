## Comparing nn-clas and par-clas in mnist dataset

from chip_clas_new import chip_clas_new
from chip_clas_new import chip_clas_new
from functions import remove_noise
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data_name = "Forest Covertype"

data = pd.read_csv("covtype.csv")

X = data.drop('Cover_Type', axis = 1)
y = data.loc[:, 'Cover_Type']

# Class 2 vs rest
y_binary = (y  == 2).astype(np.int)

# Preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(X))

# Filtering data:
X_new, y_new = remove_noise(X, y_binary)

# Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Comparing methods:
method = ["nn_clas", "parallel", "pseudo_support_edges"]

print("Dataset: {}".format(data_name))

for model in method:
    y_hat, y_test, result, runtime = chip_clas_new(X_train, X_test, y_train, y_test, method = model)

    print(" \n Method: {0} \n AUC: {1:.4f} \n Runtime: {2:.4f} \n".format(model, result, runtime))

    f = open("results.txt", "a+")
    f.write("Dataset: %d \n", data_name)

