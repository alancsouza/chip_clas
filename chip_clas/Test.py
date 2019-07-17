'''
CHIP-clas using NN-clas and parallel computing methods on 
Habermans Survival dataset
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing
from functions import remove_noise, chip_clas

data_name = "Habermans Survival"

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'
data = pd.read_csv(url, sep=',', header=None, skiprows=1)

X = data.iloc[:,:-1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) # Normalizing data between -1 and 1
X = pd.DataFrame(min_max_scaler.fit_transform(X))

y = data.iloc[:,-1].copy()
y[y == 2] = -1

# Filtering data:
X_new, y_new = remove_noise(X, y)

# Comparing methods:
method = ["nn_clas", "parallel"]

print("Dataset: {}".format(data_name))

for model in method:
    '''
    By using the chip_clas function, the number of data divisions is set automatically in the parallel 
    method.  Otherwise, use the chip_clas_new.py file to set the parameter of data divisions "split_size".
    See, for instance, the folder "Window size test" in "Experimental setup".
    '''
    y_hat, y_test, result, runtime = chip_clas(X_new, y_new, method = model, kfold = 4)

    mean_AUC = result.mean()
    std = result.std()

    print(" \n Method: {0} \n Avarege AUC: {1:.4f} \n Std. Deviation {2:.4f} \n Avarege Runtime: {3:.4f} \n".format(model, mean_AUC[0], std[0], runtime.mean()[0]))
