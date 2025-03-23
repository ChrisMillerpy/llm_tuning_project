import torch
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm import tqdm

from m2_cw.preprocessing import load_and_preprocess, get_evaluation_ids
from m2_cw.qwen import load_qwen, TokenConverter


if __name__ == "__main__":
    """
    We do a full evaluation, that means that we generate 20 timesteps
    into the future for every series. Then we compare the predicted series
    with the validation series. Computing metrics of all values.
    Greedy sampling, so the prediction is completely deterministic.
    Hence one forecast per series.
    """
    # Loading Data
    # train_texts, 1000 strings containing first 60 time steps
    # Val texts, 1000 strings containging next 20 time steps
    train_texts, val_texts = load_and_preprocess("data/lotka_volterra_data.h5")

    # Load ids of series to evaluate on
    evaluation_ids = get_evaluation_ids()

    # Load model with reduced vocabulary and initialise the token converter
    model, tokenizer, token_map = load_qwen(small_vocabulary=True)
    converter = TokenConverter(token_map)
    
    # Move model to GPU
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)

    # Initialise the forecast_file
    save_path = Path(__file__).parent
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = save_path / f"forecast_{timestamp}.csv"

    # Loop Parameters
    max_new_tokens = 20 * 13 # 20 timesteps, of <= 13 tokens per step

    # Loop over all time series
    for idx in tqdm(evaluation_ids):

        # Tokenizer context
        context = train_texts[idx]
        tokens = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        tokens = converter.to(tokens)
        tokens = tokens.to(device)

        # Generate forecast
        with torch.no_grad():
            output = model.generate(**tokens, do_sample=False, max_new_tokens=max_new_tokens)
        output = output.to("cpu")

        # Untokenize forecast
        pred = output[:, -max_new_tokens:]
        pred = converter.back(pred)
        pred = tokenizer.batch_decode(pred, skip_special_tokens=True)

        # Save forecast
        with open(file_path, "a") as f:
            line = f"{idx},{pred[0]}\n"
            f.write(line)