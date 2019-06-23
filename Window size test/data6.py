#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data6 = Breast cancer

"""
from chip_clas_new import chip_clas_new
import statistics
from functions import remove_noise
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

data_name = "Breast cancer"
print(data_name)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
data1 = pd.read_csv(url, sep=',', header=None, skiprows=1)

data = data1.iloc[:,1:].copy() # the first is the id

# converting object data into category dtype
data.iloc[:,5] = data.iloc[:,5].astype('category') 
# encoding labels
data.iloc[:,5] = data.iloc[:,5].cat.codes

X = data.iloc[:,:-1]
min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) # Normalizing data between -1 and 1
X = pd.DataFrame(min_max_scaler.fit_transform(X))

y = data.iloc[:,-1].copy() #  Class: (2 for benign, 4 for malignant cancer)
y[y == 2] = 1
y[y == 4] = -1

# Filtering data:
X_new, y_new = remove_noise(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

f = open("results_window_size.txt", "a+")
f.write("\n\nDatabase: %s \n" % data_name)
f.write("Size before filter: %d \n" % X.shape[0])
f.write("Dimension: %d \n" % X.shape[1])

f.write("Size after filter: %d \n" % X_new.shape[0])
f.write("Train Size: %d \n" % X_train.shape[0])

window_size = [50, 30, 20, 10, 5, 1]

for split in window_size:

    y_hat, y_test, result, runtime, final_split_size, arestas_suporte_size  = chip_clas_new(X_train, X_test, y_train, y_test, method = "parallel", split_size = split)


    f.write("\nSplit: %d \n" % split)
    f.write("AUC: %f \n" % result)
    f.write("Runtime: %d \n" % runtime)
    f.write("Final_split_size: %d \n" % final_split_size)
    f.write("arestas_suporte_size: %d \n" % arestas_suporte_size)
        
f.write("#######################################################################")        
f.close()