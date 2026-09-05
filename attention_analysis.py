"""Generate attention heatmaps, entropy, and attention-distance metrics."""

import argparse
import importlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import sentencepiece as spm
import torch


DEFAULT_PROMPTS = {
    "hindi": "भारत में भाषा मॉडल प्राकृतिक भाषा को समझने और उत्पन्न करने में सहायक हो सकते हैं।",
    "nepali": "नेपालमा भाषा मोडेलले प्राकृतिक भाषालाई बुझ्न र उत्पन्न गर्न मद्दत गर्न सक्छ।",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("hindi", "nepali"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_components(language):
    config_module = importlib.import_module(f"{language}.training.config")
    model_module = importlib.import_module(f"{language}.training.model")

    language_dir = Path(language)
    default_tokenizer = (
        language_dir / "tokenizers" / "unigram_8000.model"
        if language == "hindi"
        else language_dir / "tokenizers" / "unigram_10000.model"
    )

    return (
        config_module.ModelConfig,
        model_module.TransformerLM,
        default_tokenizer,
    )


def load_model(checkpoint_path, model_config_class, model_class, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_config = model_config_class(
        **checkpoint.get("model_config", asdict(model_config_class()))
    )

    model = model_class(
        vocab_size=model_config.vocab_size,
        dmodel=model_config.dmodel,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dff=model_config.dff,
        dropout=0.0,
        pad_id=model_config.pad_id,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, model_config


def capture_attention(model, input_ids):
    """Return attention weights as [layer, head, query_position, key_position]."""
    captured_attention = {}
    hooks = []

    def make_hook(layer_index):
        def hook(attention_module, inputs):
            x = inputs[0]
            _, sequence_length, _ = x.shape

            q = attention_module.Wq(x).view(
                1,
                sequence_length,
                attention_module.num_heads,
                attention_module.dhead,
            ).transpose(1, 2)

            k = attention_module.Wk(x).view(
                1,
                sequence_length,
                attention_module.num_heads,
                attention_module.dhead,
            ).transpose(1, 2)

            q, k = attention_module.rope(q, k, sequence_length)

            scores = q @ k.transpose(-2, -1)
            scores = scores / math.sqrt(attention_module.dhead)

            causal_mask = torch.tril(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    device=x.device,
                    dtype=torch.bool,
                )
            ).view(1, 1, sequence_length, sequence_length)

            scores = scores.masked_fill(~causal_mask, float("-inf"))
            captured_attention[layer_index] = torch.softmax(scores, dim=-1)[0].detach().cpu()

        return hook

    for layer_index, decoder_layer in enumerate(model.decoder.layers):
        hooks.append(
            decoder_layer.self_attn.register_forward_pre_hook(
                make_hook(layer_index)
            )
        )

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for hook in hooks:
            hook.remove()

    return torch.stack(
        [captured_attention[layer_index] for layer_index in range(len(model.decoder.layers))]
    )


def readable_token_labels(tokenizer, token_ids):
    labels = []

    for token_id in token_ids:
        piece = tokenizer.id_to_piece(int(token_id))
        piece = piece.replace("▁", " ")

        if piece == "":
            piece = "<EOS>"

        labels.append(piece)

    return labels


def plot_attention_heatmaps(attention, token_labels, layer_index, output_path):
    num_heads, sequence_length, _ = attention.shape
    columns = min(3, num_heads)
    rows = math.ceil(num_heads / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 5 * rows),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for head_index in range(num_heads):
        axis = axes[head_index]
        image = axis.imshow(
            attention[head_index].numpy(),
            cmap="magma",
            vmin=0.0,
            vmax=float(attention[head_index].max()),
            aspect="auto",
        )

        axis.set_title(f"Layer {layer_index + 1}, Head {head_index + 1}")
        axis.set_xlabel("Key position")
        axis.set_ylabel("Query position")

        tick_step = max(1, sequence_length // 12)
        tick_positions = list(range(0, sequence_length, tick_step))

        axis.set_xticks(tick_positions)
        axis.set_yticks(tick_positions)
        axis.set_xticklabels(
            [f"{position}: {token_labels[position]}" for position in tick_positions],
            rotation=90,
            fontsize=7,
        )
        axis.set_yticklabels(
            [f"{position}: {token_labels[position]}" for position in tick_positions],
            fontsize=7,
        )

        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in axes[num_heads:]:
        axis.axis("off")

    figure.suptitle(
        f"Attention Heatmaps: Layer {layer_index + 1}",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def attention_statistics(attention):
    """
    attention shape: [layers, heads, query_positions, key_positions].

    Entropy and distance exclude query position 0 because it can only attend
    to itself, which trivially gives entropy and distance of zero.
    """
    sequence_length = attention.size(-1)
    key_positions = torch.arange(sequence_length, dtype=torch.float32)

    entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1)
    distances = (
        attention
        * (
            torch.arange(sequence_length, dtype=torch.float32).view(1, 1, -1, 1)
            - key_positions.view(1, 1, 1, -1)
        )
    ).sum(dim=-1)

    return {
        "mean_attention_entropy": entropy[:, :, 1:].mean(dim=-1),
        "mean_attention_distance": distances[:, :, 1:].mean(dim=-1),
    }


def plot_head_metric(metric, metric_name, colorbar_label, output_path):
    figure, axis = plt.subplots(figsize=(10, 6))

    image = axis.imshow(metric.numpy(), cmap="viridis", aspect="auto")
    axis.set_title(metric_name)
    axis.set_xlabel("Attention head")
    axis.set_ylabel("Transformer layer")

    axis.set_xticks(range(metric.size(1)))
    axis.set_xticklabels(range(1, metric.size(1) + 1))
    axis.set_yticks(range(metric.size(0)))
    axis.set_yticklabels(range(1, metric.size(0) + 1))

    for layer_index in range(metric.size(0)):
        for head_index in range(metric.size(1)):
            axis.text(
                head_index,
                layer_index,
                f"{metric[layer_index, head_index]:.2f}",
                ha="center",
                va="center",
                color="white" if metric[layer_index, head_index] < metric.max() * 0.55 else "black",
                fontsize=8,
            )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def summarize_heads(statistics):
    entropy = statistics["mean_attention_entropy"]
    distance = statistics["mean_attention_distance"]

    results = []

    for layer_index in range(entropy.size(0)):
        for head_index in range(entropy.size(1)):
            results.append(
                {
                    "layer": layer_index + 1,
                    "head": head_index + 1,
                    "mean_entropy": float(entropy[layer_index, head_index]),
                    "mean_attention_distance": float(distance[layer_index, head_index]),
                }
            )

    return results


def main():
    args = parse_args()

    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2.")

    model_config_class, model_class, default_tokenizer = load_components(args.language)

    tokenizer_path = args.tokenizer or default_tokenizer
    output_dir = args.output_dir or (
        Path(args.language) / "training" / "attention_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt or DEFAULT_PROMPTS[args.language]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    token_ids = tokenizer.encode(prompt, out_type=int)[:args.max_tokens]

    if len(token_ids) < 2:
        raise ValueError("The prompt must produce at least two tokens.")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    model, model_config = load_model(
        args.checkpoint,
        model_config_class,
        model_class,
        device,
    )

    attention = capture_attention(model, input_ids)
    token_labels = readable_token_labels(tokenizer, token_ids)

    selected_layers = sorted({0, model_config.num_layers - 1})

    for layer_index in selected_layers:
        plot_attention_heatmaps(
            attention[layer_index],
            token_labels,
            layer_index,
            output_dir / f"attention_layer_{layer_index + 1}.png",
        )

    statistics = attention_statistics(attention)

    plot_head_metric(
        statistics["mean_attention_entropy"],
        "Mean Attention Entropy Per Head and Layer",
        "Entropy in nats",
        output_dir / "attention_entropy.png",
    )

    plot_head_metric(
        statistics["mean_attention_distance"],
        "Mean Attention Distance Per Head and Layer",
        "Average backward distance in tokens",
        output_dir / "attention_distance.png",
    )

    head_statistics = summarize_heads(statistics)

    report = {
        "language": args.language,
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(tokenizer_path),
        "device": device,
        "prompt": prompt,
        "token_ids": token_ids,
        "token_labels": token_labels,
        "selected_layers": [layer + 1 for layer in selected_layers],
        "head_statistics": head_statistics,
    }

    (output_dir / "attention_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved attention analysis to {output_dir}")
    print(f"Prompt: {prompt}")

    lowest_distance = sorted(
        head_statistics,
        key=lambda head: head["mean_attention_distance"],
    )[:5]

    highest_distance = sorted(
        head_statistics,
        key=lambda head: head["mean_attention_distance"],
        reverse=True,
    )[:5]

    print("\nMost local heads:")
    for head in lowest_distance:
        print(
            f"  L{head['layer']} H{head['head']}: "
            f"distance={head['mean_attention_distance']:.2f}, "
            f"entropy={head['mean_entropy']:.2f}"
        )

    print("\nLongest-range heads:")
    for head in highest_distance:
        print(
            f"  L{head['layer']} H{head['head']}: "
            f"distance={head['mean_attention_distance']:.2f}, "
            f"entropy={head['mean_entropy']:.2f}"
        )


if __name__ == "__main__":
    main()