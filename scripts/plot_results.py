from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default=str(ROOT / "outputs"))
    parser.add_argument("--save-path", default=str(ROOT / "outputs" / "summary.png"))
    args = parser.parse_args()
    outputs_root = Path(args.outputs_root)
    algos = {
        "ppo": outputs_root / "ppo",
        "grpo": outputs_root / "grpo",
        "dppo_topk": outputs_root / "dppo_topk",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for algo, directory in algos.items():
        train_df = load_csv(directory / "train_metrics.csv")
        eval_df = load_csv(directory / "eval_metrics.csv")
        if not train_df.empty and "train_reward" in train_df:
            axes[0, 0].plot(train_df["step"], train_df["train_reward"], label=algo)
        if not eval_df.empty and "eval_acc" in eval_df:
            axes[0, 1].plot(eval_df["step"], eval_df["eval_acc"], label=algo)
        if not train_df.empty:
            if "kl_mean" in train_df:
                axes[1, 0].plot(train_df["step"], train_df["kl_mean"], label=f"{algo}_kl")
            if "mean_divergence" in train_df:
                axes[1, 0].plot(train_df["step"], train_df["mean_divergence"], label=f"{algo}_div")
            if "clip_fraction" in train_df:
                axes[1, 1].plot(train_df["step"], train_df["clip_fraction"], label=f"{algo}_clip")
            if "mask_fraction" in train_df:
                axes[1, 1].plot(train_df["step"], train_df["mask_fraction"], label=f"{algo}_mask")

    axes[0, 0].set_title("train reward")
    axes[0, 1].set_title("eval accuracy")
    axes[1, 0].set_title("kl / divergence")
    axes[1, 1].set_title("clip / mask fraction")
    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.legend()
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"wrote {save_path}")


if __name__ == "__main__":
    main()
