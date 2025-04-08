from pathlib import Path
import json
from dataclasses import asdict

from m2_cw.lora import train_LoRA, LoRAConfig


if __name__ == "__main__":
    learning_rate = 1e-4
    lora_rank = 2

    experiment_number = 0

    for sequence_length in [128, 512, 768]:
        experiment_name = f"hyper2_{experiment_number}"

        config = LoRAConfig(lora_rank=lora_rank,
                            sequence_length=sequence_length,
                            max_steps=1500,
                            learning_rate=learning_rate,
                            batch_size=2)

        save_path = Path(__file__).parent
        
        with open(save_path/ f"{experiment_name}_config.json", "w") as f:
            json.dump(asdict(config), f)
        
        train_LoRA(config, experiment_name, save_path)

        experiment_number += 1
            