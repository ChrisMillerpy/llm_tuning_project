from pathlib import Path
from datetime import datetime
import json

from m2_cw.preprocessing import load_and_preprocess
from m2_cw.qwen import load_qwen, TokenConverter
from m2_cw.lora import add_LoRA
from m2_cw.inference import evaluate

import torch


if __name__ == "__main__":
    """
    We do an evaluation on the subset of 25 val ids deduced from expt 2,
    that means that we generate 20 timesteps
    into the future for every series. Then we compare the predicted series
    with the validation series. Computing metrics of all values.
    Greedy sampling, so the prediction is completely deterministic.
    Hence one forecast per series.
    """
    # Loading Data
    _, val_texts, _ = load_and_preprocess("data/lotka_volterra_data.h5", eval=True)

    with open(Path(__file__).parent.parent.parent / "02/val_ids_subset.json", "r") as f:
        chosen_ids = json.load(f)["all_ids"]

    val_texts_subset = {}
    for k, v in val_texts.items():
        if k in chosen_ids:
            val_texts_subset[k] = v
    val_texts = val_texts_subset

    assert len(val_texts) == len(chosen_ids)

    # Load model with reduced vocabulary and initialise the token converter
    model, tokenizer, token_map = load_qwen(small_vocabulary=True)
    converter = TokenConverter(token_map)

    dir_contents = [ model for model in Path(__file__).parent.iterdir() if "model" in str(model) ]
    assert len(dir_contents) == 1
    model_path = dir_contents[0]

    model = add_LoRA(model, lora_rank=4)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialise the forecast_file
    save_path = Path(__file__).parent
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = save_path / f"forecast_{timestamp}.txt"

    # Run the evaluation
    evaluate(model, tokenizer, converter, "mps", val_texts, file_path)