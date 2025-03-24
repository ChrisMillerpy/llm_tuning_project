import numpy as np


def mae(gt, pred):
    return np.mean(np.abs(gt - pred))