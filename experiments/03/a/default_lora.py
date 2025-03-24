import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator
from pathlib import Path
from datetime import datetime
import wandb

from m2_cw.preprocessing import load_and_preprocess, chunk_sequences
from m2_cw.qwen import load_qwen, TokenConverter
from m2_cw.lora import add_LoRA

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":

    # Load model and tokenizer, then apply LoRA to the model
    model, tokenizer, token_map = load_qwen(small_vocabulary=True)
    converter = TokenConverter(token_map)

    lora_rank = 4
    model = add_LoRA(model, lora_rank=lora_rank)


    # Initialise WandB
    wandb.init(project="LoRA_Qwen")
    wandb.watch(model, log="all", log_freq=10)

    # Process the data into sequences of text
    train_texts = load_and_preprocess("data/lotka_volterra_data.h5")
    # ^Each of these is a `list[str]` representing contiguous parts of the time series,
    #  in text form (using the LLMTIME scheme).

    # Defines the maximum context length
    max_ctx_length = 512
    train_input_ids = chunk_sequences(
        train_texts, tokenizer, converter, max_ctx_length, stride=max_ctx_length // 2
    )

    batch_size = 2
    learning_rate = 1e-5

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    train_dataset = TensorDataset(train_input_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare components with Accelerator
    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    steps = 0
    print("Beginning Training Loop")
    max_steps = 5000
    while steps < max_steps:
        progress_bar = tqdm(train_loader, desc=f"Steps {steps}")
        for (batch,) in progress_bar:
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            wandb.log({"loss": loss})
            accelerator.backward(loss)
            optimizer.step()
            steps += 1 # Because we are doing batch size 4

            progress_bar.set_postfix(loss=loss.item())
            if steps > max_steps:
                break

    model.eval()

    # Stop tracking wandb
    wandb.finish()

    # Initialise the model savepath
    save_path = Path(__file__).parent
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = save_path / f"model_{timestamp}.pth"

    torch.save(model.state_dict(), file_path)