# data4 = Haberman’s Survival

from functions import *

data_name = "Habermans Survival"
result_name = "Result_Data4_parallel_16.csv"
runtime_name = "Runtime_data4_16_parallel.csv"

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'
data = pd.read_csv(url, sep=',', header=None, skiprows=1)

X = data.iloc[:,:-1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) # Normalizing data between -1 and 1
X = pd.DataFrame(min_max_scaler.fit_transform(X))

y = data.iloc[:,-1].copy()
y[y == 2] = -1

print("The {} dataset has {} samples".format(data_name, X.shape[0]))

y_hat, y_test = chip_clas(X, y)

AUC = compute_AUC(y_test, y_hat)

print("The AUC is: {}".format(AUC))
