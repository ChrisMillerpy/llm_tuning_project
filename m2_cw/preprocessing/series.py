import numpy as np


def get_evaluation_ids():
    np.random.seed(10022002)
    indices = np.arange(0, 1000, 1)
    series_ids = np.random.choice(indices, 100, replace=False)

    return series_ids

def get_train_ids():
    eval_ids = get_evaluation_ids()
    all_ids = np.arange(0, 1000, 1)
    train_ids = np.setdiff1d(all_ids, eval_ids)

    return train_ids