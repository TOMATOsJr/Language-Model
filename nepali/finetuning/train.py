"""Finetunes the pretrained Nepali decoder on synthetic reasoning examples."""

import math

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nepali.finetuning.config import FinetuneConfig
from nepali.finetuning.dataset import IGNORE_INDEX, ReasoningDataset, collate_batch
from nepali.training.config import ModelConfig
from nepali.training.model import TransformerLM


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


def save_checkpoint(path, model, step, val_loss, model_cfg):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "model_config": vars(model_cfg),
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

    checkpoint = torch.load(cfg.init_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Initialized from {cfg.init_checkpoint} (pretrain step {checkpoint.get('step')})")

    optimizer = build_optimizer(model, cfg)
    total_steps = len(train_loader) * cfg.num_epochs

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.num_epochs}")
        for input_ids, labels in progress:
            step += 1
            lr = lr_at_step(step, cfg.warmup_steps, total_steps, cfg.peak_lr, cfg.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = shifted_loss(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            if step % cfg.log_every_steps == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            if step % cfg.eval_every_steps == 0:
                val_loss, val_ppl, val_acc = evaluate(model, val_loader, device)
                print(f"step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} val_acc={val_acc:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(cfg.checkpoint_dir / "best.pt", model, step, val_loss, model_cfg)

        val_loss, val_ppl, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch} end: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} val_acc={val_acc:.4f}")
        save_checkpoint(cfg.checkpoint_dir / f"epoch_{epoch:02d}.pt", model, step, val_loss, model_cfg)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(cfg.checkpoint_dir / "best.pt", model, step, val_loss, model_cfg)


if __name__ == "__main__":
    train()
