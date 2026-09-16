"""Configuration for finetuning the pretrained Hindi decoder on synthetic reasoning data."""

from dataclasses import dataclass
from pathlib import Path

from hindi.training.config import ModelConfig

LANGUAGE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class FinetuneConfig:
    train_jsonl: Path = LANGUAGE_DIR / "generator" / "data" / "train.jsonl"
    val_jsonl: Path = LANGUAGE_DIR / "generator" / "data" / "val.jsonl"
    test_jsonl: Path = LANGUAGE_DIR / "generator" / "data" / "test.jsonl"

    tokenizer_path: Path = LANGUAGE_DIR / "tokenizers" / f"unigram_{ModelConfig.vocab_size}.model"
    init_checkpoint: Path = LANGUAGE_DIR / "training" / "checkpoints" / "best.pt"
    checkpoint_dir: Path = LANGUAGE_DIR / "finetuning" / "checkpoints"

    max_length: int = 224
    batch_size: int = 16
    num_epochs: int = 8
    peak_lr: float = 5e-5
    min_lr: float = 5e-6
    warmup_steps: int = 100
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0

    eval_every_steps: int = 100
    log_every_steps: int = 20
    num_workers: int = 0
    seed: int = 42
