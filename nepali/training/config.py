"""
Configuration for Nepali 25M-parameter decoder-only transformer training.
"""

from dataclasses import dataclass
from pathlib import Path

LANGUAGE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    dmodel: int = 384
    num_layers: int = 12
    num_heads: int = 6
    dff: int = 1536
    dropout: float = 0.1
    context_length: int = 512
    pad_id: int = 0


@dataclass
class TrainConfig:
    train_bin: Path = LANGUAGE_DIR / "datasets" / "tokenized" / "train.bin"
    validation_bin: Path = LANGUAGE_DIR / "datasets" / "tokenized" / "validation.bin"
    test_bin: Path = LANGUAGE_DIR / "datasets" / "tokenized" / "test.bin"

    checkpoint_dir: Path = LANGUAGE_DIR / "training" / "checkpoints"

    micro_batch_size: int = 32
    grad_accum_steps: int = 2          # 32 * 512 * 2 = 32,768 tokens/update

    max_steps: int = 61040             # 4 epochs (~2B tokens) at 32,768 tokens/update
    warmup_steps: int = 500
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0

    eval_every: int = 500
    eval_max_batches: int = 100        # evaluates 819,200 tokens per validation pass
    log_every: int = 20
    checkpoint_every: int = 1000

    num_workers: int = 4
    seed: int = 42
    compile_model: bool = True

    wandb_project: str = "nepali-25m-lm"
    wandb_run_name: str = "nepali-decoder-25m"
