from pathlib import Path
from datetime import datetime
import json
import re

from m2_cw.preprocessing import load_and_preprocess
from m2_cw.qwen import load_qwen, TokenConverter
from m2_cw.lora import add_LoRA
from m2_cw.inference import evaluate

import torch
import sys


if __name__ == "__main__":
    """
    We do a massively reduced forecast on the 5 worst performing series from 3a,
    that means that we generate 20 timesteps
    into the future for every series. Then we compare the predicted series
    with the validation series. Computing metrics of all values.
    Greedy sampling, so the prediction is completely deterministic.
    Hence one forecast per series.
    """
    # Loading Data
    _, val_texts, _ = load_and_preprocess("data/lotka_volterra_data.h5", eval=True)

    with open(Path(__file__).parent.parent / "a/val_ids_subsubset.json", "r") as f:
        already_forecast_ids = json.load(f)["all_ids"]

    with open(Path(__file__).parent.parent.parent / "02/val_ids_subset.json", "r") as f:
        subset_ids = json.load(f)["bad_ids"]
    
    new_ids = [ idx for idx in subset_ids if not idx in already_forecast_ids]

    val_texts_subset = {}
    for k, v in val_texts.items():
        if k in new_ids:
            val_texts_subset[k] = v
    val_texts = val_texts_subset

    assert len(val_texts) == len(new_ids)

    config_path_list = [ file for file in Path(__file__).parent.iterdir() if "_config.json" in str(file) ]
    assert len(config_path_list) == 9
    configs = {}
    for file in config_path_list:
        match = re.search(r"expt_(\d+)_config\.json", str(file))
        if match:
            n = int(match.group(1))
        else:
            raise ValueError("Couldn't find an experiment number in file {file}")
        if n >= 6:
            with open(file, "r") as f:
                configs[n] = json.load(f)
    
    for expt_number, config in configs.items():
        # Load model with reduced vocabulary and initialise the token converter
        model, tokenizer, token_map = load_qwen(small_vocabulary=True)
        converter = TokenConverter(token_map)

        dir_contents = [ model for model in Path(__file__).parent.iterdir() if f"model_expt_{expt_number}" in str(model) ]
        assert len(dir_contents) == 1
        model_path = dir_contents[0]

        model = add_LoRA(model, lora_rank=config["lora_rank"])
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Initialise the forecast_file
        save_path = Path(__file__).parent
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = save_path / f"forecast_expt_{expt_number}_{timestamp}.txt"

        # Run the evaluation
        evaluate(model, tokenizer, converter, "mps", val_texts, file_path)