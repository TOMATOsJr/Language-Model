from dataclasses import asdict
from pathlib import Path

import torch

from hindi.training.config import ModelConfig, TrainConfig

def serializable_config(config):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }

model_config = serializable_config(ModelConfig())
train_config = serializable_config(TrainConfig())

for checkpoint_path in Path("hindi/training/checkpoints").glob("*.pt"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint["model_config"] = model_config
    checkpoint["train_config"] = train_config
    checkpoint.setdefault("scaler_state_dict", {})
    checkpoint["scheduler_state_dict"] = {
        "type": "cosine_decay_with_linear_warmup",
        "step": checkpoint["step"],
        "warmup_steps": train_config["warmup_steps"],
        "max_steps": train_config["max_steps"],
        "peak_lr": train_config["peak_lr"],
        "min_lr": train_config["min_lr"],
    }

    backup_path = checkpoint_path.with_suffix(".pt.original")
    if not backup_path.exists():
        backup_path.write_bytes(checkpoint_path.read_bytes())

    torch.save(checkpoint, checkpoint_path)
    print(f"Updated {checkpoint_path}")