from pathlib import Path
import h5py
import numpy as np
from einops import rearrange, repeat
import torch
from ..qwen.qwen import TokenConverter
from .series import get_evaluation_ids


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
    train = data[:, :80, :]

    train_flattened = rearrange(train, "b x y -> b (x y)")

    train_percentiles = repeat(np.percentile(train_flattened, alpha, axis=1), "b -> b 1 1")

    data_scaled = 10 * data / train_percentiles

    data_scaled_and_rounded = ( 10**decimal_places * np.round(data_scaled, decimals=decimal_places)).astype(int)

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
    
    split = 80
    start_string = ";".join(series_list[:split]) + ";"
    end_string = ";".join(series_list[split:]) + ";"

    return start_string, end_string

def preprocess(data):
    """
    Preprocesses trajectory data into formatted strings.

    Parameters:
        data (np.ndarray): The trajectory data to be processed.

    Returns:
        tuple: Lists of train, validation, and test texts.
    """
    val_ids, test_ids = get_evaluation_ids()
    train_texts = {}
    val_texts = {}
    test_texts = {}

    for series_id in range(data.shape[0]):
        # Format from numbers to strings
        start_string, end_string = preprocess_series(data[series_id, :, :])
        # If it is one of the series that we evaluate on, then add separate strings to train, val, test
        if series_id in val_ids:
            val_texts[series_id] = [start_string, end_string]
        elif series_id in test_ids:
            test_texts[series_id] = [start_string, end_string]
        else:
            train_texts[series_id] = start_string + end_string
            

    
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

def load_and_preprocess(data_path: str="data/lotka_volterra_data.h5", alpha: int = 90, decimal_places: int = 2, random_seed: int = 42, eval=False):
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
    data_path = Path(__file__).parent.parent.parent / data_path
    
    # Read the original data
    data = read_data(data_path) 

    # Scale the original data to alpha percentile on 10
    data = scale_and_round(data, alpha, decimal_places)
    
    # Preprocess the the serieses into strings
    train_texts, val_texts, test_texts = preprocess(data)
    
    if eval:
        return train_texts, val_texts, test_texts
    else:
        return train_texts

def load(data_path: str="data/lotka_volterra_data.h5"):
    data_path = Path(__file__).parent.parent.parent / data_path
    
    # Read the original data
    data = read_data(data_path) 

    return data

def chunk_sequences(texts, tokenizer, converter, max_length=512, stride=256):
    """
    Tokenizes and chunks text sequences for training.

    Args:
        texts (list[str]): List of text sequences.
        tokenizer: The tokenizer used to convert text to tokens.
        max_length (int): Maximum sequence length per chunk.
        stride (int): Step size for sliding window tokenization.

    Returns:
        torch.Tensor: A tensor containing tokenized and chunked sequences.
    """
    # Ensure we have a converter to convert to reduced tokens
    if not isinstance(converter, TokenConverter):
        raise TypeError("Token Converter is not valid.")

    all_input_ids = []
    for idx, text in texts.items():
        # Apply Qwen's tokenization scheme to the text:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)

        encoding = converter.to(encoding)

        seq_ids = encoding.input_ids[0]

        # Create sliding windows to further divide the data into chunks:
        for i in range(0, len(seq_ids), stride):
            # If our window overshoots the series
            if i + max_length > len(seq_ids):
                # reset to perfectly cover last part
                i = len(seq_ids) - max_length
            # slice the sequence to make the chunk
            chunk = seq_ids[i : i + max_length]

            all_input_ids.append(chunk)

    return torch.stack(all_input_ids)