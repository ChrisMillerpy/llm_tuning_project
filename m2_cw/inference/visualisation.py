from ..preprocessing.preprocessor import load_and_preprocess
from .utils import load_forecast, string_to_array
from .metrics import mae
import matplotlib.pyplot as plt
import numpy as np


def visualise_forecasts(file_path, plots_per_row=5, eval_set="val", prefix=None):
    # Load the forecasts
    forecasts = load_forecast(file_path)

    # Load the training data
    _, val_data, test_data = load_and_preprocess(eval=True)
    if eval_set == "val":
        data = val_data
    elif eval_set == "test":
        data = test_data

    # Conver from strings to arrays of numbers
    for idx in data.keys():
        pair = data[idx]
        start_arr = string_to_array(pair[0], trim=80)
        end_arr = string_to_array(pair[1], trim=20)
        data[idx] = [start_arr, end_arr]

    errors = {}

    # Loop Through and plt the forecasts
    i = 0
    for idx, forecast in forecasts.items():

        if i == 0:
            fig, ax = plt.subplots(1, plots_per_row, figsize=(4*plots_per_row, 4))
        
        start = data[idx][0]
        end = data[idx][1]
        gt = np.concatenate((start, end), axis=0)
        context_length = start.shape[0]
        gt_length = gt.shape[0]
        forecast_length = forecast.shape[0]
        x1 = np.arange(0, gt_length, 1)
        x2 = np.arange(context_length, context_length + forecast_length, 1)

        error = mae(end, forecast)
        errors[idx] = error

        if isinstance(prefix, int):
            gt = gt[context_length-prefix:]
            x1 = x1[context_length-prefix:]

        ax[i].set_title(f"Series: {idx}, Error: {error}")
        # ax[i].plot(x1, context, c="k")
        ax[i].plot(x1, gt, c="k")
        ax[i].plot(x2, forecast, c="k", ls=":")

        for j in range(2):
            ax[i].fill_between(x2, end[:, j], forecast[:, j], color="red", alpha=0.5)

        if i == plots_per_row - 1:
            i = 0
            plt.show();
        else:
            i += 1
    
    plt.show();

    return errors