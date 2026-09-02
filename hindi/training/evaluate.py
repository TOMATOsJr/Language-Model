"""
Shared evaluation: token-weighted cross-entropy and perplexity over a data loader.
"""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_loader(model, loader, device, max_batches=None):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_index, (input_ids, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(input_ids)
        # Sum (not mean) so batches of unequal size combine into a correct token-weighted average.
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )

        total_loss += loss_sum.item()
        total_tokens += labels.numel()

    model.train(was_training)

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity
