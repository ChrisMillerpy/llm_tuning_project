import numpy as np
from ..preprocessing.preprocessor import load_and_preprocess
import matplotlib.pyplot as plt


def string_to_array(string, trim=20):
    time_steps = string.split(";")[:trim]
    array = []
    for step in time_steps:
        next_entry = step.split(",")
        try:
            next_entry = [ float(val) for val in next_entry ]
            array.append(next_entry)
        except:
            pass
    array = np.array(array)
    return array

def load_forecast(file_path):
    forecasts = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            idx, forecast = line.split(",", 1)

            try:
                idx = int(idx)
                forecasts[idx] = forecast
            except:
                pass

    for k, v in forecasts.items():
        forecasts[k] = string_to_array(v)

    return forecasts