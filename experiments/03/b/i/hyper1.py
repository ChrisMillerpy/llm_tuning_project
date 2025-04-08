from pathlib import Path
import json
from dataclasses import asdict

from m2_cw.lora import train_LoRA, LoRAConfig


if __name__ == "__main__":
    learning_rates = [1e-5, 5e-5, 1e-4]
    lora_ranks = [2, 4, 8]

    experiment_number = 0

    for lr in learning_rates:
        for rank in lora_ranks:

            experiment_name = f"expt_{experiment_number}"

            config = LoRAConfig(lora_rank=rank,
                                sequence_length=512,
                                max_steps=3150,
                                learning_rate=lr,
                                batch_size=2)

            save_path = Path(__file__).parent
            
            with open(save_path/ f"{experiment_name}_config.json", "w") as f:
                json.dump(asdict(config), f)
            
            train_LoRA(config, experiment_name, save_path)

            experiment_number += 1
            