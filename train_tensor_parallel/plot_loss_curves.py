"""
Plot loss curves from training runs.

Usage:
    python plot_loss_curves.py --input_dir /tmp/loss_curves --output loss_curves.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_loss_history(filepath: str) -> dict:
    """Load loss history from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_loss_curves(input_dir: str, output_path: str):
    """Plot loss curves for all implementations."""
    # Load all loss histories
    implementations = {}

    files = {
        "DDP (TP=1, DP=2)": "loss_ddp.json",
        "DeepSpeed AutoTP (TP=2, DP=2)": "loss_deepspeed.json",
        "FSDP+DTensor (TP=2, DP=2)": "loss_fsdp.json",
    }

    colors = {
        "DDP (TP=1, DP=2)": "#1f77b4",  # blue
        "DeepSpeed AutoTP (TP=2, DP=2)": "#ff7f0e",  # orange
        "FSDP+DTensor (TP=2, DP=2)": "#2ca02c",  # green
    }

    for name, filename in files.items():
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            data = load_loss_history(filepath)
            implementations[name] = data["loss_history"]
            print(f"Loaded {name}: {len(data['loss_history'])} steps")
        else:
            print(f"Warning: {filepath} not found")

    if not implementations:
        print("No loss histories found!")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot loss curves
    for name, losses in implementations.items():
        steps = list(range(len(losses)))
        ax.plot(steps, losses, label=name, color=colors[name], linewidth=1.5)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curves", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("LOSS CURVE STATISTICS")
    print("=" * 60)

    for name, losses in implementations.items():
        losses_arr = np.array(losses)
        print(f"\n{name}:")
        print(f"  Initial loss: {losses_arr[0]:.6f}")
        print(f"  Final loss:   {losses_arr[-1]:.6f}")
        print(f"  Min loss:     {losses_arr.min():.6f}")
        print(f"  Mean loss:    {losses_arr.mean():.6f}")

    if "DDP (TP=1, DP=2)" in implementations:
        ddp_losses = np.array(implementations["DDP (TP=1, DP=2)"])
        print("\n" + "-" * 60)
        print("Differences from DDP baseline:")
        for name, losses in implementations.items():
            if name == "DDP (TP=1, DP=2)":
                continue
            losses_arr = np.array(losses)
            min_len = min(len(ddp_losses), len(losses_arr))
            diff = losses_arr[:min_len] - ddp_losses[:min_len]
            print(f"\n{name}:")
            print(f"  Max abs diff:  {np.abs(diff).max():.6f}")
            print(f"  Mean abs diff: {np.abs(diff).mean():.6f}")
            print(f"  Final diff:    {diff[-1]:.6f}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves")
    parser.add_argument("--input_dir", type=str, default="/tmp/loss_curves")
    parser.add_argument("--output", type=str, default="loss_curves.png")

    args = parser.parse_args()
    plot_loss_curves(args.input_dir, args.output)


if __name__ == "__main__":
    main()
