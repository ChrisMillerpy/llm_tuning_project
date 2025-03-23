from ..preprocessing.preprocessor import load_and_preprocess
from .utils import load_forecast, string_to_array
import matplotlib.pyplot as plt


def visualise_forecasts(file_path, plots_per_row=5):
    # Load the forecasts
    forecasts = load_forecast(file_path)

    # Load the training data
    train_data, gt_data = load_and_preprocess()

    # Conver from strings to arrays of numbers
    for i in range(len(train_data)):
        text = train_data[i]
        arr = string_to_array(text, trim=60)
        train_data[i] = arr

    for i in range(len(gt_data)):
        text = gt_data[i]
        arr = string_to_array(text, trim=60)
        gt_data[i] = arr

    # Loop Through and plt the forecasts
    i = 0
    for idx, forecast in forecasts.items():

        if i == 0:
            fig, ax = plt.subplots(1, plots_per_row, figsize=(4*plots_per_row, 4))
        
        context = train_data[idx] 
        gt = gt_data[idx]
        context_length = context.shape[0]
        forecast_length = forecast.shape[0]
        x1 = np.arange(0, context_length, 1)
        x2 = np.arange(context_length, context_length + forecast_length, 1)

        ax[i].set_title(f"Series: {idx}")
        ax[i].plot(x1, context, c="k")
        ax[i].plot(x2, gt, c="k")
        ax[i].plot(x2, forecast, c="red")

        if i == plots_per_row - 1:
            i = 0
            plt.show();
        else:
            i += 1
    
    plt.show();