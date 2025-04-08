from pathlib import Path
import json
from dataclasses import asdict

from m2_cw.lora import train_LoRA, LoRAConfig


if __name__ == "__main__":
    learning_rate = 1e-4
    lora_rank = 2

    experiment_name = f"final"

    config = LoRAConfig(lora_rank=2,
                        sequence_length=512,
                        max_steps=15400,
                        learning_rate=1e-4,
                        batch_size=2)

    save_path = Path(__file__).parent
    
    with open(save_path/ f"{experiment_name}_config.json", "w") as f:
        json.dump(asdict(config), f)
    
    train_LoRA(config, experiment_name, save_path)