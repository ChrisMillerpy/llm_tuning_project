import numpy as np


def get_evaluation_ids(N_val=50, N_test=50):
    np.random.seed(10022002)
    indices = np.arange(0, 1000, 1)
    val_ids = np.random.choice(indices, N_val, replace=False)
    indices = np.setdiff1d(indices, val_ids)
    test_ids = np.random.choice(indices, N_test, replace=False)

    return val_ids, test_ids