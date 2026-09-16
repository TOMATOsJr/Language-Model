"""Finetunes the pretrained Hindi decoder on synthetic reasoning examples."""

import math

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hindi.finetuning.config import FinetuneConfig
from hindi.finetuning.dataset import IGNORE_INDEX, ReasoningDataset, collate_batch
from hindi.training.config import ModelConfig
from hindi.training.model import TransformerLM


def lr_at_step(step, warmup_steps, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def build_optimizer(model, cfg):
    # Matrices get weight decay; biases and norm gains do not.
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.peak_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps)


def shifted_loss(logits, labels, reduction="mean"):
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    model.eval()

    total_loss, total_tokens, total_correct = 0.0, 0, 0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids)
        loss = shifted_loss(logits, labels, reduction="sum")
        mask = labels[:, 1:] != IGNORE_INDEX
        total_loss += loss.item()
        total_tokens += mask.sum().item()
        total_correct += ((logits[:, :-1, :].argmax(-1) == labels[:, 1:]) & mask).sum().item()

    model.train(was_training)
    mean_loss = total_loss / max(1, total_tokens)
    accuracy = total_correct / max(1, total_tokens)
    return mean_loss, math.exp(mean_loss), accuracy


def scheduler_state(step, cfg, total_steps):
    return {
        "type": "cosine_decay_with_linear_warmup",
        "step": step,
        "warmup_steps": cfg.warmup_steps,
        "max_steps": total_steps,
        "peak_lr": cfg.peak_lr,
        "min_lr": cfg.min_lr,
    }

def save_checkpoint(path, model, optimizer, scaler, step, val_loss, model_cfg, train_cfg, total_steps):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state(step, train_cfg, total_steps),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": vars(model_cfg),
            "train_config": vars(train_cfg),
            "val_loss": val_loss,
        },
        path,
    )


def train():
    cfg = FinetuneConfig()
    model_cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    tokenizer = spm.SentencePieceProcessor(model_file=str(cfg.tokenizer_path))
    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else model_cfg.pad_id

    train_dataset = ReasoningDataset(cfg.train_jsonl, tokenizer, cfg.max_length)
    val_dataset = ReasoningDataset(cfg.val_jsonl, tokenizer, cfg.max_length)
    collate = lambda batch: collate_batch(batch, pad_id)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate,
    )

    model = TransformerLM(
        vocab_size=model_cfg.vocab_size,
        dmodel=model_cfg.dmodel,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        dff=model_cfg.dff,
        dropout=model_cfg.dropout,
        pad_id=pad_id,
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    total_steps = len(train_loader) * cfg.num_epochs

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = cfg.checkpoint_dir / "latest.pt"

    best_val_loss = float("inf")
    step = 0
    start_epoch = 1

    if latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        step = checkpoint["step"]
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        start_epoch = (step // len(train_loader)) + 1
        print(f"Resumed from {latest_checkpoint_path} at step {step}")
    else:
        checkpoint = torch.load(cfg.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized from {cfg.init_checkpoint} (pretrain step {checkpoint.get('step')})")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.num_epochs}")
        for batch_idx, (input_ids, labels) in enumerate(progress):
            if epoch == start_epoch and batch_idx < (step % len(train_loader)):
                continue  # Skip already processed batches in the resumed epoch

            step += 1
            lr = lr_at_step(step, cfg.warmup_steps, total_steps, cfg.peak_lr, cfg.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=device == "cuda"):
                logits = model(input_ids)
                loss = shifted_loss(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            if step % cfg.log_every_steps == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            if step % cfg.eval_every_steps == 0:
                val_loss, val_ppl, val_acc = evaluate(model, val_loader, device)
                print(f"step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} val_acc={val_acc:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(cfg.checkpoint_dir / "best.pt", model, optimizer, scaler, step, val_loss, model_cfg, cfg, total_steps)

        val_loss, val_ppl, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch} end: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} val_acc={val_acc:.4f}")
        save_checkpoint(cfg.checkpoint_dir / f"epoch_{epoch:02d}.pt", model, optimizer, scaler, step, val_loss, model_cfg, cfg, total_steps)
        save_checkpoint(cfg.checkpoint_dir / "latest.pt", model, optimizer, scaler, step, val_loss, model_cfg, cfg, total_steps)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(cfg.checkpoint_dir / "best.pt", model, optimizer, scaler, step, val_loss, model_cfg, cfg, total_steps)


if __name__ == "__main__":
    train()
