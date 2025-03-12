"""
Author: [Chris Miller]
Date: 2025-03-08
License: MIT
"""

from pathlib import Path
import h5py
from sklearn.model_selection import train_test_split

# HyperParameters
alpha = 6 / 10  # Scaling Constant
decimal_places = 2  # Decimal Place Precision

def read_data(data_path):
    """
    Reads time and trajectory data from an HDF5 file.

    Args:
        data_path (Path): The path to the directory containing the HDF5 file.

    Returns:
        tuple: A tuple containing:
            - time_data (numpy.ndarray): Array of time values.
            - traj_data (numpy.ndarray): Array of trajectory data.

    Raises:
        FileNotFoundError: If the specified HDF5 file does not exist.
        KeyError: If the expected datasets ("time" and "trajectories") are not found in the file.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    
    with h5py.File(data_path, "r") as f:
        time_data = f["time"][:]
        traj_data = f["trajectories"][:]
    
    return time_data, traj_data

def preprocess_event(data, event_id):
    """
    Processes a single event from the dataset by extracting prey and predator values,
    rounding them to a fixed number of decimal places, and formatting them as a string.

    Args:
        data (numpy.ndarray): A 3D array where the first dimension indexes events,
                              the second represents time steps, and the third contains 
                              prey and predator values.
        event_id (int): The index of the event to preprocess.

    Returns:
        str: A formatted string representing the event's time series, where each time step 
             is stored as "prey_value,predator_value" and time steps are separated by ";".

    Raises:
        IndexError: If `event_id` is out of bounds for `data`.
        NameError: If `decimal_places` is not defined before calling the function.
    """
    event_list = []
    event = data[event_id, :, :]
    for i in range(event.shape[0]):
        prey_value = event[i, 0] / alpha
        predator_value = event[i, 1] / alpha

        prey_str = str(round(prey_value, decimal_places))
        predator_str = str(round(predator_value, decimal_places))

        next_entry = f"{prey_str},{predator_str}"
        event_list.append(next_entry)
    
    event_string = ";".join(event_list)
    return event_string

def preprocess(data):
    """
    Preprocesses all events in the dataset by converting each event's time series 
    data into a formatted string.

    Args:
        data (numpy.ndarray): A 3D array where each event is indexed along the first 
                              dimension, the second dimension represents time steps, 
                              and the third dimension contains prey and predator values.

    Returns:
        list of str: A list where each element is a formatted string representation 
                     of an event's time series. Each string consists of multiple 
                     "prey_value,predator_value" pairs separated by semicolons (";").
    """
    event_strings = []
    for event_id in range(data.shape[0]):
        event_string = preprocess_event(data, event_id)
        event_strings.append(event_string)
    
    return event_strings

def split(event_strings):
    """
    Splits the preprocessed event strings into training, validation, and test sets.

    Args:
        event_strings (list of str): A list of formatted event strings.

    Returns:
        tuple: A tuple containing:
            - train_texts (list of str): Training set event strings.
            - val_texts (list of str): Validation set event strings.
            - test_texts (list of str): Test set event strings.
    """
    train_texts, temp_texts = train_test_split(event_strings, test_size=0.2, random_state=42)
    val_texts, test_texts = train_test_split(temp_texts, test_size=0.5, random_state=42)
    return train_texts, val_texts, test_texts

def save(test_texts, save_path):
    """
    Saves the test set event strings to a file.

    Args:
        test_texts (list of str): List of formatted event strings for testing.
        save_path (Path): Path to save the test set file.
    """
    with open(save_path, "w") as f:
        for test_event in test_texts:
            f.write(test_event + "\n")

def load_and_preprocess(data_path: str):
    """
    Loads, preprocesses, splits, and saves the test set from the event data.

    Args:
        data_path (str): Path to the HDF5 file containing event data.

    Returns:
        tuple: A tuple containing:
            - train_texts (list of str): Training set event strings.
            - val_texts (list of str): Validation set event strings.
    """
    data_path = Path(__file__).parent.parent / data_path
    
    # Read the original data
    time_data, traj_data = read_data(data_path) 
    
    # Preprocess the event strings
    event_strings = preprocess(traj_data)
    
    # Split into train, val, test
    train_texts, val_texts, test_texts = split(event_strings)
    
    # Save the test set
    save_path = data_path.parent / "test_texts.txt"
    save(test_texts, save_path)
    
    return train_texts, val_texts