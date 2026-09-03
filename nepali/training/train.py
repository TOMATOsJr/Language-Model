"""
Training loop for the Nepali 25M-parameter decoder-only transformer.
"""

import math
from pathlib import Path

from dotenv import load_dotenv
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from nepali.training.config import ModelConfig, TrainConfig
from nepali.training.data import TokenStreamDataset
from nepali.training.evaluate import evaluate_loader
from nepali.training.model import TransformerLM


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def infinite_loader(loader):
    # Re-invokes iter(loader) every pass so shuffle=True reshuffles each epoch.
    while True:
        for batch in loader:
            yield batch


def lr_at_step(step, warmup_steps, max_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def build_optimizer(model, weight_decay, lr, betas, eps):
    # Matrices get weight decay; biases and norm gains do not.
    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def save_checkpoint(path, model, optimizer, step, val_loss):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def train():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    torch.manual_seed(train_cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TokenStreamDataset(train_cfg.train_bin, context_length=model_cfg.context_length)
    val_dataset = TokenStreamDataset(train_cfg.validation_bin, context_length=model_cfg.context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = TransformerLM(
        vocab_size=model_cfg.vocab_size,
        dmodel=model_cfg.dmodel,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        dff=model_cfg.dff,
        dropout=model_cfg.dropout,
        pad_id=model_cfg.pad_id,
    ).to(device)

    assert model.lm_head.weight is model.token_embedding.weight, "output projection must stay tied"

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = build_optimizer(
        model,
        train_cfg.weight_decay,
        train_cfg.peak_lr,
        (train_cfg.adam_beta1, train_cfg.adam_beta2),
        train_cfg.adam_eps,
    )

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and not use_bf16))

    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=train_cfg.wandb_project,
        name=train_cfg.wandb_run_name,
        config={**vars(model_cfg), **vars(train_cfg)},
    )

    data_iter = infinite_loader(train_loader)
    best_val_loss = float("inf")

    progress_bar = tqdm(range(1, train_cfg.max_steps + 1), desc="training")
    for step in progress_bar:
        lr = lr_at_step(step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.peak_lr, train_cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(train_cfg.grad_accum_steps):
            input_ids, labels = next(data_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = loss / train_cfg.grad_accum_steps

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        if step % train_cfg.log_every == 0:
            progress_bar.set_postfix(loss=f"{accumulated_loss:.4f}", lr=f"{lr:.2e}")
            wandb.log(
                {"train/loss": accumulated_loss, "train/lr": lr, "train/grad_norm": grad_norm.item()},
                step=step,
            )

        if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps:
            val_loss, val_ppl = evaluate_loader(model, val_loader, device, max_batches=train_cfg.eval_max_batches)
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step)
            tqdm.write(f"step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

            save_checkpoint(train_cfg.checkpoint_dir / "latest.pt", model, optimizer, step, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(train_cfg.checkpoint_dir / "best.pt", model, optimizer, step, val_loss)

        if step % train_cfg.checkpoint_every == 0:
            save_checkpoint(train_cfg.checkpoint_dir / f"step_{step}.pt", model, optimizer, step, None)

    # Final report uses the best validation checkpoint, evaluated once on the full held-out test set.
    test_dataset = TokenStreamDataset(train_cfg.test_bin, context_length=model_cfg.context_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    best_checkpoint = torch.load(train_cfg.checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_ppl = evaluate_loader(model, test_loader, device, max_batches=None)
    print(f"TEST  loss={test_loss:.4f}  perplexity={test_ppl:.2f}")
    wandb.summary["test/loss"] = test_loss
    wandb.summary["test/perplexity"] = test_ppl

    wandb.finish()


if __name__ == "__main__":
    train()
