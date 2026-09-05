"""
Training loop for the Hindi 25M-parameter decoder-only transformer.
"""

import math
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from hindi.training.config import ModelConfig, TrainConfig
from hindi.training.data import TokenStreamDataset
from hindi.training.evaluate import evaluate_loader
from hindi.training.model import TransformerLM


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def infinite_loader(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


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


def unwrap_model(model):
    if hasattr(model, "module"):
        model = model.module
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def config_to_dict(config):
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(config).items()
    }


def scheduler_state(step, train_cfg):
    return {
        "type": "cosine_decay_with_linear_warmup",
        "step": step,
        "warmup_steps": train_cfg.warmup_steps,
        "max_steps": train_cfg.max_steps,
        "peak_lr": train_cfg.peak_lr,
        "min_lr": train_cfg.min_lr,
    }


def save_checkpoint(path, model, optimizer, scaler, step, val_loss, model_cfg, train_cfg):
    torch.save(
        {
            "step": step,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state(step, train_cfg),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": config_to_dict(model_cfg),
            "train_config": config_to_dict(train_cfg),
            "val_loss": val_loss,
        },
        path,
    )


def train():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        master_process = (rank == 0)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(train_cfg.seed + rank)

    # Adjust gradient accumulation for multi-GPU if DDP is active
    grad_accum_steps = max(1, train_cfg.grad_accum_steps // world_size)

    train_dataset = TokenStreamDataset(train_cfg.train_bin, context_length=model_cfg.context_length)
    val_dataset = TokenStreamDataset(train_cfg.validation_bin, context_length=model_cfg.context_length)

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=train_cfg.seed,
            drop_last=True,
        )
        if is_ddp
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    if master_process:
        print(f"Steps per epoch: {steps_per_epoch:,}")

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

    assert unwrap_model(model).lm_head.weight is unwrap_model(model).token_embedding.weight, "output projection must stay tied"

    if master_process:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

    if device.startswith("cuda") and train_cfg.compile_model:
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = build_optimizer(
        model,
        train_cfg.weight_decay,
        train_cfg.peak_lr,
        (train_cfg.adam_beta1, train_cfg.adam_beta2),
        train_cfg.adam_eps,
    )

    def supports_native_bf16():
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 8 and torch.cuda.is_bf16_supported()

    use_bf16 = supports_native_bf16()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda") and not use_bf16))

    latest_checkpoint_path = train_cfg.checkpoint_dir / "latest.pt"
    best_checkpoint_path = train_cfg.checkpoint_dir / "best.pt"
    start_step = 1
    best_val_loss = float("inf")
    if train_cfg.resume_from_latest and latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = checkpoint["step"] + 1
        best_val_loss = checkpoint["val_loss"]
        if master_process:
            print(f"Resuming from {latest_checkpoint_path} at step {start_step}")

    if master_process:
        train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=train_cfg.wandb_project,
            name=train_cfg.wandb_run_name,
            config={**vars(model_cfg), **vars(train_cfg)},
        )

    data_iter = infinite_loader(train_loader, train_sampler)

    progress_bar = tqdm(range(start_step, train_cfg.max_steps + 1), desc="training", disable=not master_process)
    for step in progress_bar:
        lr = lr_at_step(step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.peak_lr, train_cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(grad_accum_steps):
            input_ids, labels = next(data_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=autocast_dtype, enabled=device.startswith("cuda")):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        if master_process and step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch

            save_checkpoint(
                train_cfg.checkpoint_dir / f"epoch_{epoch:02d}.pt",
                model,
                optimizer,
                scaler,
                step,
                val_loss=float("nan"),
                model_cfg=model_cfg,
                train_cfg=train_cfg,
            )

            tqdm.write(
                f"Saved epoch {epoch} checkpoint at step {step:,}"
            )

        if master_process and step % train_cfg.log_every == 0:
            progress_bar.set_postfix(loss=f"{accumulated_loss:.4f}", lr=f"{lr:.2e}")
            wandb.log(
                {"train/loss": accumulated_loss, "train/lr": lr, "train/grad_norm": grad_norm.item()},
                step=step,
            )

        if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps:
            val_loss, val_ppl = evaluate_loader(model, val_loader, device, max_batches=train_cfg.eval_max_batches)
            if master_process:
                wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step)
                tqdm.write(f"step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

                save_checkpoint(train_cfg.checkpoint_dir / "latest.pt", model, optimizer, scaler, step, val_loss, model_cfg, train_cfg)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(best_checkpoint_path, model, optimizer, scaler, step, val_loss, model_cfg, train_cfg)

    # Final report uses the best validation checkpoint, evaluated once on the full held-out test set.
    if master_process:
        test_dataset = TokenStreamDataset(train_cfg.test_bin, context_length=model_cfg.context_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_cfg.micro_batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=(device.startswith("cuda")),
        )

        best_checkpoint = torch.load(train_cfg.checkpoint_dir / "best.pt", map_location=device)
        unwrap_model(model).load_state_dict(best_checkpoint["model_state_dict"])

        test_loss, test_ppl = evaluate_loader(model, test_loader, device, max_batches=None)
        print(f"TEST  loss={test_loss:.4f}  perplexity={test_ppl:.2f}")
        wandb.summary["test/loss"] = test_loss
        wandb.summary["test/perplexity"] = test_ppl

        wandb.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
