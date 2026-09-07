"""Interactive Gradio inference UI for the Hindi and Nepali language models."""

import argparse
import importlib
from pathlib import Path

import gradio as gr
import sentencepiece as spm
import torch


PROJECT_DIR = Path(__file__).resolve().parent
LANGUAGE_CONFIGS = {
    "Hindi": {
        "module": "hindi.training.generate",
        "checkpoint": PROJECT_DIR / "hindi" / "training" / "checkpoints" / "best.pt",
        "tokenizer": PROJECT_DIR / "hindi" / "tokenizers" / "unigram_8000.model",
        "examples": [["भारत की राजधानी"], ["कृत्रिम बुद्धिमत्ता"], ["आज मौसम बहुत अच्छा है क्योंकि"]],
    },
    "Nepali": {
        "module": "nepali.training.generate",
        "checkpoint": PROJECT_DIR / "nepali" / "training" / "checkpoints" / "best.pt",
        "tokenizer": PROJECT_DIR / "nepali" / "tokenizers" / "unigram_10000.model",
        "examples": [["नेपालको राजधानी"], ["कृत्रिम बुद्धिमत्ता"], ["आज मौसम राम्रो छ किनभने"]],
    },
}


class ModelRegistry:
    """Lazily load each model the first time it is selected."""

    def __init__(self, device, checkpoints):
        self.device = device
        self.checkpoints = checkpoints
        self.models = {}

    def get(self, language):
        if language in self.models:
            return self.models[language]

        settings = LANGUAGE_CONFIGS[language]
        checkpoint_path = self.checkpoints.get(language, settings["checkpoint"])
        tokenizer_path = settings["tokenizer"]

        if not checkpoint_path.is_file():
            raise gr.Error(f"Checkpoint not found: {checkpoint_path}")
        if not tokenizer_path.is_file():
            raise gr.Error(f"Tokenizer not found: {tokenizer_path}")

        generation_module = importlib.import_module(settings["module"])
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        model = generation_module.load_model(
            checkpoint_path,
            generation_module.ModelConfig(),
            self.device,
        )
        self.models[language] = (generation_module, model, tokenizer)
        return self.models[language]


def create_app(device, checkpoints):
    registry = ModelRegistry(device, checkpoints)

    def generate_text(language, prompt, max_new_tokens, temperature, top_p):
        prompt = prompt.strip()
        if not prompt:
            raise gr.Error("Enter a prompt before generating text.")

        generation_module, model, tokenizer = registry.get(language)
        common_arguments = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "context_length": 512,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "device": device,
        }
        return generation_module.generate(**common_arguments)

    with gr.Blocks(title="Hindi and Nepali Language Models") as interface:
        gr.Markdown("# Hindi and Nepali Language Models")
        with gr.Row():
            language = gr.Dropdown(
                choices=list(LANGUAGE_CONFIGS),
                value="Hindi",
                label="Model",
            )
            max_new_tokens = gr.Slider(1, 512, value=128, step=1, label="New tokens")
        prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Write a prompt in Hindi or Nepali")
        with gr.Row():
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
        generate_button = gr.Button("Generate", variant="primary")
        output = gr.Textbox(label="Generated text", lines=12)
        gr.Examples(LANGUAGE_CONFIGS["Hindi"]["examples"], inputs=prompt)

        generate_button.click(
            generate_text,
            inputs=[language, prompt, max_new_tokens, temperature, top_p],
            outputs=output,
        )
        prompt.submit(
            generate_text,
            inputs=[language, prompt, max_new_tokens, temperature, top_p],
            outputs=output,
        )

    return interface.queue(default_concurrency_limit=1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hindi-checkpoint", type=Path)
    parser.add_argument("--nepali-checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = {
        language: checkpoint
        for language, checkpoint in {
            "Hindi": args.hindi_checkpoint,
            "Nepali": args.nepali_checkpoint,
        }.items()
        if checkpoint is not None
    }
    print(f"Using device: {device}")
    create_app(device, checkpoints).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()