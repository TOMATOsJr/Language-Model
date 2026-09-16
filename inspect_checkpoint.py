import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file.")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    print("--- Checkpoint Keys ---")
    print(list(checkpoint.keys()))

    print("\n--- Model Config ---")
    for k, v in checkpoint.get("model_config", {}).items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

