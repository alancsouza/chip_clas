import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.matlib # use repmat function
from scipy.spatial import distance_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import random
import time
import copy
import concurrent.futures
random.seed(1)

# Compute Adjacency matrix
def get_adjacency(X):
  dist_matrix = distance_matrix(X,X)
  Adj_matrix = np.zeros(shape = dist_matrix.shape)
  nrow = dist_matrix.shape[0]
  for i in range(nrow):
    for j in range(nrow):
      if (i != j):          
        d1 = (dist_matrix[i,j])/2
        dist = pd.DataFrame((X.iloc[i,:]+X.iloc[j,:])/2).T 
        d = distance_matrix(dist, X)      
        d[0,i] = float("inf")
        d[0,j] = float("inf")      
        compara = (d<d1)
        
        if not compara.any():
          Adj_matrix[i,j] = 1
          Adj_matrix[j,i] = 1
          
  return Adj_matrix

# Removing overlapping samples:
def remove_noise(X, y):
  Adj_matrix = get_adjacency(X)

  c1 = np.asarray(np.where(y==1)).ravel()
  c2 = np.asarray(np.where(y==-1)).ravel()
  A1 = Adj_matrix[:,c1].sum(axis = 0) # sum over columns
  A2 = Adj_matrix[:,c2].sum(axis = 0)
  
  
  M = pd.DataFrame(Adj_matrix)
  adj_1 = np.asarray(M.iloc[c1,c1])
  adj_2 = np.asarray(M.iloc[c2,c2])

  A1h = adj_1.sum(axis = 0)
  A2h = adj_2.sum(axis = 0)
  
  #Computing the quality coefficient Q for each class
  Q_class1 = A1h / A1
  Q_class2 = A2h / A2

  # Computing the threshold value t for each class
  t_class1 = sum(Q_class1) / Q_class1.shape[0]
  t_class2 = sum(Q_class2) / Q_class2.shape[0]
  
  noise_c1 = np.where(Q_class1 < t_class1)
  noise_c2 = np.where(Q_class2 < t_class2)
  noise_data = np.c_[noise_c1, noise_c2]

  noise = noise_data.ravel()
  
  # Filtering the data
  X_new = X.drop(noise)
  y_new = y.drop(noise)
  
  print("{} samples where removed from the data. \n".format(X.shape[0]-X_new.shape[0]))
  print("The data set now has {} samples ".format(X.shape[0]))

  return X_new, y_new

# Split the data for concurrent computing
def split(X_train, y_train):
    if X_train.shape[0] > 400: # define the slot size
        split_size = round(X_train.shape[0]/100)
    else:
        split_size = round(X_train.shape[0]/50)

    data_train = np.c_[X_train, y_train]
    np.random.shuffle(data_train)

    data_split = np.array_split(data_train, split_size)

    return data_split, split_size


# Finding the separation border:
def get_borda(y, Adj_matrix):
  y_t = pd.DataFrame(y).T
  
  ncol = y_t.shape[1]
  mask = np.matlib.repmat(y_t, ncol, 1)
  mask2 = pd.DataFrame(mask*Adj_matrix)  
  borda = pd.DataFrame(np.zeros(ncol)).T

  for idx in range(ncol):
    a1 =  sum(-y_t.iloc[0, idx] == mask2.iloc[idx,:]) # check if the labels are different
    if a1 > 0:
      borda[idx] = 1
    
  return borda

# Finding the support edges:
def get_arestas_suporte(X, y, borda, Adj_matrix):
  X = np.asarray(X)
  y_t = pd.DataFrame(y).T
  ncol = y_t.shape[1]
  mask = np.matlib.repmat(y_t, ncol, 1)
  nrow = Adj_matrix.shape[0]
  maskBorda = np.matlib.repmat(borda == 1, nrow, 1)
  maskBorda = np.asarray(maskBorda)

  # Removing the lines that not belong to the margin
  aux = maskBorda * np.transpose(maskBorda)

  # Removing edges that do not belong to the graph
  aux = Adj_matrix * aux

  # Removing edges from same labels vertices
  aux1 = aux + (mask * aux)
  aux2 = aux - (mask * aux)
  aux1 = np.asarray(aux1)
  aux2 = np.asarray(aux2)
  aux = aux1 * np.transpose(aux2)

  # converting matrix to binary
  aux  = (aux != 0)

  arestas = np.where(aux == 1)

  arestas = np.transpose(np.asarray(arestas))
  nrow_arestas = arestas.shape[0]
  ncol_arestas = arestas.shape[1]

  arestas_suporte = []
  y_suporte = []

  y_arr = np.asarray(y)

  for i in range(nrow_arestas):
    for j in range(ncol_arestas):
    
      idx = arestas[i,j]
      arestas_suporte.append(X[idx,:])
      y_suporte.append(y_arr[idx])

  
  
  X_suporte = np.asarray(arestas_suporte)
  y_suporte = np.asarray(y_suporte)
  
  return X_suporte, y_suporte

# Another support edges function that contains the other functions
def support_edges(data):  

  if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)   
  X_train = data.iloc[:,:-1]
  y_train = data.iloc[:, -1]
  Adj_matrix = get_adjacency(X_train)
  
  borda = get_borda(y_train, Adj_matrix)
  X_suporte, y_suporte = get_arestas_suporte(X_train, y_train, borda, Adj_matrix)
  
  arestas_suporte = np.c_[X_suporte, y_suporte]
  if arestas_suporte.shape[0] > 0:
    arestas_suporte = np.unique(arestas_suporte, axis = 0)
  
  return arestas_suporte

# Classification
def classify_data(X_test, y_test, arestas_suporte):
  
  X_suporte = arestas_suporte[:,:-1]
  #y_suporte = arestas_suporte[:,-1]
  nrow = X_test.shape[0]
  dist_test = distance_matrix(X_test, X_suporte) # compute the distance from the sample to the support egdes

  y_hat = np.zeros(nrow)

  for idx in range(nrow):
    dist = dist_test[idx,:]
    min_idx = np.argmin(dist)
    y_hat[idx] = arestas_suporte[min_idx, -1] 
  
  return y_hat

# Performance measure using AUC
def compute_AUC(y_test, y_hat):
  fpr, tpr, _ = roc_curve(y_test, y_hat)
  if fpr.shape[0] < 2 or tpr.shape[0] < 2:
      roc_auc = float('nan')
  else:
      roc_auc = auc(fpr, tpr)
  
  return roc_auc

# Parallel graph method:
def parallel_graph(X_train, y_train, split_size):
  data_train = np.c_[X_train, y_train]
  np.random.shuffle(data_train)

  data_split = np.array_split(data_train, split_size)
  arestas_suporte_final = []
  
  for i in range(split_size):
    data = pd.DataFrame(data_split[i])
    X_train = data.iloc[:,:-1]
    y_train = data.iloc[:, -1]

    # Finding the support edges from this slot of data:
    arestas_suporte = support_edges(data)

    arestas_suporte_final.append(arestas_suporte)
    
  arr = arestas_suporte_final[0]

  for i in range(split_size-1):
    i = i+1
    arr = np.concatenate((arr, arestas_suporte_final[i]), axis = 0)  
    
  data_train_new = pd.DataFrame(arr)
  X_train_new = data_train_new.iloc[:,:-1]
  y_train_new = data_train_new.iloc[:,-1]
  
  return X_train_new, y_train_new

# Gabriel Graph classifier
def gg_clas(X_train, y_train, X_test, y_test):

  X_train, y_train = remove_noise(X_train,y_train)
  data_train = np.c_[X_train, y_train]
  arestas_suporte = support_edges(data_train)
  y_hat = classify_data(X_test, y_test, arestas_suporte)

  return y_hat


def parallel_concurrent(X_train, y_train, X_test, y_test):
  # Splitting the data for concurrent computing
  data_split, split_size = split(X_train, y_train)
  
  # list for adding the support edges from each slot
  Support = []

  with concurrent.futures.ProcessPoolExecutor() as executor:
      for data, S in zip(data_split, executor.map(support_edges, data_split)):
          Support.append(S)
      
  Support_arr = np.vstack(Support) # transform list to array

  data_train_new = pd.DataFrame(Support_arr)
      
  # Finding the new set of support edges
  arestas_suporte = support_edges(data_train_new)

  # Classification:
  y_hat = classify_data(X_test, y_test, arestas_suporte)

  return y_hat


def chip_clas(X, y, parallel_method = True, kfold = 0, test_size = 0.2):
  # Filtering data:
  X_new, y_new = remove_noise(X, y)

  if kfold > 0 :    
    kf = KFold(n_splits = kfold, shuffle = True, random_state = 1)

    for train_index, test_index in kf.split(X_new):

      X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
      y_train, y_test = y_new.iloc[train_index], y_new.iloc[test_index]

      if parallel_method :
        y_hat = parallel_concurrent(X_train, y_train, X_test, y_test)      

      else:
        y_hat  = gg_clas(X_train, y_train, X_test, y_test)

  else:
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size)

    if parallel_method :
      y_hat = parallel_concurrent(X_train, y_train, X_test, y_test)      

    else:
      y_hat  = gg_clas(X_train, y_train, X_test, y_test)

  return y_hat, y_test