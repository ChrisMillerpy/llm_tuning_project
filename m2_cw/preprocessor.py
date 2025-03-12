from pathlib import Path
import h5py
import numpy as np
from einops import rearrange, repeat


def read_data(data_path):
    """
    Reads trajectory data from an HDF5 file.

    Parameters:
        data_path (Path): Path to the HDF5 file containing the trajectory data.

    Returns:
        np.ndarray: Array of trajectory data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    
    with h5py.File(data_path, "r") as f:
        traj_data = f["trajectories"][:]
    
    return traj_data

def scale_and_round(data, alpha=90, decimal_places=2):
    """
    Scales and rounds trajectory data based on the alpha percentile.

    Parameters:
        data (np.ndarray): The trajectory data to be processed.
        alpha (int, optional): The percentile used for scaling. Defaults to 90.
        decimal_places (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        np.ndarray: Scaled and rounded trajectory data.
    """
    train = data[:, :60, :]

    train_flattened = rearrange(train, "b x y -> b (x y)")

    train_percentiles = repeat(np.percentile(train_flattened, alpha, axis=1), "b -> b 1 1")

    data_scaled = 10 * data / train_percentiles

    data_scaled_and_rounded = np.round(data_scaled, decimals=decimal_places)

    return data_scaled_and_rounded

def preprocess_series(series):
    """
    Converts a series of predator-prey values into formatted strings.

    Parameters:
        series (np.ndarray): A sequence of prey and predator values.

    Returns:
        tuple: Train, validation, and test strings formatted for processing.
    """
    series_list = []
    for (prey_val, pred_val) in series:
        prey_str = str(prey_val)
        pred_str = str(pred_val)

        next_entry = f"{prey_str},{pred_str}"
        series_list.append(next_entry)
    
    s1, s2 = 60, 80
    train_string = ";".join(series_list[:s1]) + ";"
    val_string = ";".join(series_list[s1:s2]) + ";"
    test_string = ";".join(series_list[s2:]) + ";"

    return train_string, val_string, test_string

def preprocess(data):
    """
    Preprocesses trajectory data into formatted strings.

    Parameters:
        data (np.ndarray): The trajectory data to be processed.

    Returns:
        tuple: Lists of train, validation, and test texts.
    """
    train_texts = []
    val_texts = []
    test_texts = []

    for series_id in range(data.shape[0]):
        train_string, val_string, test_string = preprocess_series(data[series_id, :, :])
        train_texts.append(train_string)
        val_texts.append(val_string)
        test_texts.append(test_string)
    
    return train_texts, val_texts, test_texts

def save(texts, save_path):
    """
    Saves a list of formatted strings to a file.

    Parameters:
        texts (list): List of strings to be saved.
        save_path (Path): Path to the output file.
    """
    with open(save_path, "w") as f:
        for string in texts:
            f.write(string + "\n")

def load_and_preprocess(data_path: str, alpha: int = 90, decimal_places: int = 2, random_seed: int = 42):
    """
    Loads trajectory data, preprocesses it, and saves test data to a file.

    Parameters:
        data_path (str): Path to the input HDF5 file.
        alpha (int, optional): Percentile for scaling. Defaults to 90.
        decimal_places (int, optional): Number of decimal places to round to. Defaults to 2.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: Lists of train and validation texts.
    """
    data_path = Path(__file__).parent.parent / data_path
    
    # Read the original data
    data = read_data(data_path) 

    # Scale the original data to alpha percentile on 10
    data = scale_and_round(data, alpha, decimal_places)
    
    # Preprocess the the serieses into strings
    train_texts, val_texts, test_texts = preprocess(data)

    # Save the test set
    save_path = data_path.parent / "test_texts.txt"
    save(test_texts, save_path)
    
    return train_texts, val_texts