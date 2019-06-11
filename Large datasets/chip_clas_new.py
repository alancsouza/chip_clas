from functions import parallel_concurrent, nn_clas, pseudo_support_edges, compute_AUC
import time

# inputs: X_train, y_train, X_test, y_test

def chip_clas_new(X_train, X_test, y_train, y_test, method, split_size):
    
    """
    Available methods:
    parallel: Implements concurrent futures and parallelization technique
    nn_clas: Implements nn_clas classification
    pseudo_support_edges = Implements pseudo_support method

    """

    start = time.time()

    if method == "parallel" :
        y_hat = parallel_concurrent(X_train, y_train, X_test, y_test, split_size)      

    elif method == "nn_clas":
        y_hat  = nn_clas(X_train, y_train, X_test, y_test)

    elif method == "pseudo_support_edges":
        y_hat = pseudo_support_edges(X_train, y_train, X_test, y_test)

    else :
        print("Method not available")
        return None

    end =  time.time()
    runtime = end - start

    results = compute_AUC(y_test, y_hat)
    
    return y_hat, y_test, results, runtime


