import torch
from tqdm import tqdm


def evaluate(model, tokenizer, converter, device, texts, file_path, max_new_tokens=11 * 20, context_length=512):

    # Move model to GPU
    device = torch.device(device if torch.mps.is_available() else "cpu")
    model.to(device)

    # Loop over all time series
    for idx, pair in tqdm(texts.items()):

        # Tokenizer context
        context = pair[0][-512:]
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
    return