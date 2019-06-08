#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data1 = Banknote Auth.
"""

import chip_clas_new
import statistics

data_name = "Banknote Auth."

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
data = pd.read_csv(url, header = None)

X = data.iloc[:,:-1]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) # Normalizing data between -1 and 1
X = pd.DataFrame(min_max_scaler.fit_transform(X))
y = data.iloc[:,-1].copy()
y[y == 0] = -1

# Filtering data:
X_new, y_new = remove_noise(X, y)

# Implementing kfold cross validation:
k = 10

kf = KFold(n_splits=k, shuffle = True, random_state = 1)
results = []
runtime = []

split_size = 10

for train_index, test_index in kf.split(X_new):
    start = time.time()   

    X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
    y_train, y_test = y_new.iloc[train_index], y_new.iloc[test_index]

    y_hat, y_test, result, runtime = chip_clas_new(X_train, X_test, y_train, y_test, method = "parallel" , split_size)

    print(" \n Method: {0} \n AUC: {1:.4f} \n Runtime: {2:.4f} \n".format(model, result, runtime))

    runtime.append(runtime)
    results.append(result)


mean_AUC = sum(results)/len(results)
mean_runtime = sum(runtime)/len(runtime)
std = statistics.stdev(results)

f = open("results_window_size.txt", "a+")
f.write("\n\nSplit_size: %s \n" % split_size)
f.write("AUC: %d \n" % mean_AUC)
f.write("Std Deviation: %d \n" % std)
f.write("Runtime: %d \n" % mean_runtime)
f.close()


