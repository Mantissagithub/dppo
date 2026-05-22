from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_RUNS = ["ppo", "grpo", "dppo_topk", "dppo_full"]


def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def resolve_eval_df(outputs_root, run_name, run_dir):
    train_eval = load_csv(run_dir / "eval_metrics.csv")
    if not train_eval.empty:
        return train_eval
    return load_csv(outputs_root / f"{run_name}_eval" / "eval_metrics.csv")


def build_run_table(outputs_root, run_names):
    runs = {}
    for run_name in run_names:
        run_dir = outputs_root / run_name
        train_df = load_csv(run_dir / "train_metrics.csv")
        eval_df = resolve_eval_df(outputs_root, run_name, run_dir)
        if train_df.empty and eval_df.empty:
            continue
        runs[run_name] = {
            "train": train_df,
            "eval": eval_df,
        }
    return runs


def plot_metric(runs, metric, phase, save_path, title, ylabel=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    found = False
    for run_name, data in runs.items():
        df = data[phase]
        if df.empty or metric not in df.columns:
            continue
        ax.plot(df["step"], df[metric], label=run_name)
        found = True

    if not found:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_system_usage(runs, save_path):
    metrics = [
        ("cpu_rss_gb", "train"),
        ("gpu_memory_allocated_gb", "train"),
        ("gpu_memory_reserved_gb", "train"),
        ("gpu_memory_peak_gb", "train"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    any_found = False
    for axis, (metric, phase) in zip(axes.flat, metrics):
        found = False
        for run_name, data in runs.items():
            df = data[phase]
            if df.empty or metric not in df.columns:
                continue
            axis.plot(df["step"], df[metric], label=run_name)
            found = True
            any_found = True
        axis.set_title(metric)
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)
        if found:
            axis.legend()
    if not any_found:
        plt.close(fig)
        return False
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_summary(runs, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for run_name, data in runs.items():
        train_df = data["train"]
        eval_df = data["eval"]
        if not train_df.empty and "train_reward" in train_df.columns:
            axes[0, 0].plot(train_df["step"], train_df["train_reward"], label=run_name)
        if not eval_df.empty and "eval_acc" in eval_df.columns:
            axes[0, 1].plot(eval_df["step"], eval_df["eval_acc"], label=run_name)
        if not train_df.empty and "kl_mean" in train_df.columns:
            axes[1, 0].plot(train_df["step"], train_df["kl_mean"], label=f"{run_name}_kl")
        if not train_df.empty and "mean_divergence" in train_df.columns:
            axes[1, 0].plot(train_df["step"], train_df["mean_divergence"], label=f"{run_name}_div")
        if not train_df.empty and "clip_fraction" in train_df.columns:
            axes[1, 1].plot(train_df["step"], train_df["clip_fraction"], label=f"{run_name}_clip")
        if not train_df.empty and "mask_fraction" in train_df.columns:
            axes[1, 1].plot(train_df["step"], train_df["mask_fraction"], label=f"{run_name}_mask")

    axes[0, 0].set_title("train reward")
    axes[0, 1].set_title("eval acc")
    axes[1, 0].set_title("kl / divergence")
    axes[1, 1].set_title("clip / mask fraction")
    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default=str(ROOT / "outputs"))
    parser.add_argument("--save-dir", default=str(ROOT / "outputs" / "plots"))
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    runs = build_run_table(outputs_root, args.runs)
    if not runs:
        raise SystemExit("no run metrics found")

    plot_summary(runs, save_dir / "summary.png")

    metric_specs = [
        ("train_reward", "train", "train reward", "train_reward.png"),
        ("eval_acc", "eval", "eval acc", "eval_acc.png"),
        ("eval_reward", "eval", "eval reward", "eval_reward.png"),
        ("kl_mean", "train", "kl mean", "kl_mean.png"),
        ("mean_divergence", "train", "mean divergence", "mean_divergence.png"),
        ("entropy_mean", "train", "entropy mean", "entropy_mean.png"),
        ("ratio_max", "train", "ratio max", "ratio_max.png"),
        ("clip_fraction", "train", "clip fraction", "clip_fraction.png"),
        ("mask_fraction", "train", "mask fraction", "mask_fraction.png"),
        ("response_length", "train", "response length", "response_length.png"),
        ("gpu_hours", "train", "gpu hours", "gpu_hours.png"),
    ]
    written = [save_dir / "summary.png"]
    for metric, phase, title, file_name in metric_specs:
        path = save_dir / file_name
        if plot_metric(runs, metric, phase, path, title):
            written.append(path)

    system_path = save_dir / "system_usage.png"
    if plot_system_usage(runs, system_path):
        written.append(system_path)

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
