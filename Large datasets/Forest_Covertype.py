## Comparing nn-clas and par-clas in Forest Covertype dataset

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
y = (y  == 2).astype(np.int)

# Preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(X))

# getting a smaller subset of the mnist dataset
Data = np.c_[X, y]

Data = pd.DataFrame(Data)

# separating the lables
c1 = Data[Data.iloc[:,-1] ==  1]
c2 = Data[Data.iloc[:,-1] == 0]

c1_small = c1.sample(n = 5000, random_state = 1)
c2_small = c2.sample(n = 5000, random_state = 1)

Small_data = pd.concat([c1_small, c2_small])

# shuffle data
Small_data = Small_data.sample(frac = 1 )

X = Small_data.iloc[:,:-1]
y = Small_data.iloc[:,-1]
y[y==0] = -1

# Filtering data:
#X_new, y_new = remove_noise(X, y_binary)

# Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training size: {}".format(X_train.shape))
print("Test size: {}".format(X_test.shape))
# Comparing methods:
method = ["nn_clas", "parallel", "pseudo_support_edges"]

print("Dataset: {}".format(data_name))


for model in method:
    y_hat, y_test, result, runtime = chip_clas_new(X_train, X_test, y_train, y_test, method = model, split_size = 6)

    print(" \n Method: {0} \n AUC: {1:.4f} \n Runtime: {2:.4f} \n".format(model, result, runtime))

    f = open("results.txt", "a+")
    f.write("Dataset: %d \n", data_name)

