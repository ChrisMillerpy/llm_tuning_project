from ..preprocessing.preprocessor import load_and_preprocess
from .utils import load_forecast, string_to_array
from .metrics import mae
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

def visualise_forecasts(file_path, plots_per_row=5, eval_set="val", prefix=None, save=False, save_path=None):
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

    indices = []
    errors = []
    max_distances = []

    # Loop Through and plt the forecasts
    plot = 0
    i = 0
    for idx, forecast in forecasts.items():

        if i == 0:
            fig, ax = plt.subplots(1, plots_per_row, figsize=(4*plots_per_row, 4), constrained_layout=True)
        
        start = data[idx][0]
        end = data[idx][1]
        gt = np.concatenate((start, end), axis=0)
        context_length = start.shape[0]
        gt_length = gt.shape[0]
        forecast_length = forecast.shape[0]
        x1 = np.arange(0, gt_length, 1)
        x2 = np.arange(context_length, context_length + forecast_length, 1)

        max_distance = np.max(np.abs(end - forecast))
        error = mae(end, forecast)
        indices.append(idx)
        errors.append(error)
        max_distances.append(max_distance)

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
            plot += 1
            
            if save:
                try:
                    fig.savefig(save_path / f"forecasts_{plot}")
                except:
                    print()
            plt.show();
        else:
            i += 1
        
    
    plt.show();

    df = pd.DataFrame(data={
        "series_id": indices,
        "MAE": errors,
        "max_dist": max_distances,
    })

    return df

def boxplot_maes(data, labels):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    means = [ np.mean(arr) for arr in data ]
    positions = list(range(len(data)))
    assert len(labels) == len(data)
    # colors = ['lightgray', 'lightgreen', 'lightcoral']
    colors = [
        "#FFCCCC",  # light red
        "#FFDDC1",  # peach
        "#FFF2CC",  # light yellow
        "#D5F4E6",  # light mint
        "#E0F7FA",  # light cyan
        "#D6EAF8",  # light blue
        "#E8DAEF",  # light lavender
        "#FDEBD0",  # light beige
        "#FADBD8",  # light pink
    ]

    box = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(color='gray', linewidth=1.2),
        capprops=dict(color='gray', linewidth=1.2),
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markersize=5, markerfacecolor='black', linestyle='none')
    )

    # Fill each box with color
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors[i % 9])

    ax.plot(positions, means, lw=1, ls="--", c="k", marker="x", label="Avg MAE")

    # Add labels and clean up
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE")
    # ax.set_title("MAE Distribution", fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    ax.legend()

    plt.tight_layout()

    print("Average MAE:")
    for label, mean in zip(labels, means):
        print(f" - {label}: {mean:.2f}")

    return fig, ax
    
