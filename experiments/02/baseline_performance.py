from pathlib import Path
from datetime import datetime

from m2_cw.preprocessing import load_and_preprocess
from m2_cw.qwen import load_qwen, TokenConverter
from m2_cw.inference import evaluate



if __name__ == "__main__":
    """
    We do a full evaluation, that means that we generate 20 timesteps
    into the future for every series. Then we compare the predicted series
    with the validation series. Computing metrics of all values.
    Greedy sampling, so the prediction is completely deterministic.
    Hence one forecast per series.
    """
    # Loading Data
    _, val_texts, _ = load_and_preprocess("data/lotka_volterra_data.h5", eval=True)

    # Load model with reduced vocabulary and initialise the token converter
    model, tokenizer, token_map = load_qwen(small_vocabulary=True)
    converter = TokenConverter(token_map)

    # Initialise the forecast_file
    save_path = Path(__file__).parent
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = save_path / f"forecast_{timestamp}.txt"

    # Run the evaluation
    evaluate(model, tokenizer, converter, "mps", val_texts, file_path)