from pathlib import Path
import argparse
import json
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


def load_training_log(path):
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_run_frames(outputs_root, run_name):
    run_dir = outputs_root / run_name
    training_log = load_training_log(run_dir / "training.log")
    if not training_log.empty and "phase" in training_log.columns:
        train_df = training_log[training_log["phase"] == "train"].copy()
        eval_df = training_log[training_log["phase"] == "eval"].copy()
    else:
        train_df = load_csv(run_dir / "train_metrics.csv")
        eval_df = load_csv(run_dir / "eval_metrics.csv")

    if eval_df.empty:
        eval_dir_df = load_csv(outputs_root / f"{run_name}_eval" / "eval_metrics.csv")
        if not eval_dir_df.empty:
            eval_df = eval_dir_df

    if not eval_df.empty and "step" in eval_df.columns:
        if eval_df["step"].nunique() == 1 and float(eval_df["step"].iloc[0]) == 0.0 and not train_df.empty:
            eval_df = eval_df.copy()
            eval_df["step"] = float(train_df["step"].max())

    return train_df, eval_df


def build_run_table(outputs_root, run_names):
    runs = {}
    for run_name in run_names:
        train_df, eval_df = load_run_frames(outputs_root, run_name)
        if train_df.empty and eval_df.empty:
            continue
        runs[run_name] = {"train": train_df, "eval": eval_df}
    return runs


def style_axis(axis, title, ylabel, xlabel="step"):
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)


def plot_timeseries_metric(runs, metric, phase, save_path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    found = False
    for run_name, data in runs.items():
        df = data[phase]
        if df.empty or metric not in df.columns:
            continue
        series = df[["step", metric]].dropna()
        if series.empty:
            continue
        marker = "o" if len(series) <= 8 else None
        ax.plot(series["step"], series[metric], label=run_name, linewidth=2, marker=marker, markersize=4)
        found = True
    if not found:
        plt.close(fig)
        return False
    style_axis(ax, title, metric)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_final_bar_metric(runs, metric, phase, save_path, title):
    values = []
    labels = []
    for run_name, data in runs.items():
        df = data[phase]
        if df.empty or metric not in df.columns:
            continue
        series = df[metric].dropna()
        if series.empty:
            continue
        labels.append(run_name)
        values.append(float(series.iloc[-1]))
    if not values:
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    style_axis(ax, title, metric, xlabel="run")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_system_usage(runs, save_path):
    metrics = ["cpu_rss_gb", "gpu_memory_allocated_gb", "gpu_memory_reserved_gb", "gpu_memory_peak_gb"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    any_found = False
    for axis, metric in zip(axes.flat, metrics):
        found = False
        for run_name, data in runs.items():
            df = data["train"]
            if df.empty or metric not in df.columns:
                continue
            series = df[["step", metric]].dropna()
            if series.empty:
                continue
            axis.plot(series["step"], series[metric], label=run_name, linewidth=2)
            found = True
            any_found = True
        style_axis(axis, metric, metric)
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
        if not train_df.empty and "train_reward" in train_df.columns:
            series = train_df[["step", "train_reward"]].dropna()
            if not series.empty:
                axes[0, 0].plot(series["step"], series["train_reward"], label=run_name, linewidth=2)

    final_eval = []
    for run_name, data in runs.items():
        eval_df = data["eval"]
        if not eval_df.empty and "eval_acc" in eval_df.columns:
            series = eval_df["eval_acc"].dropna()
            if not series.empty:
                final_eval.append((run_name, float(series.iloc[-1])))
    if final_eval:
        axes[0, 1].bar([item[0] for item in final_eval], [item[1] for item in final_eval])

    for run_name, data in runs.items():
        train_df = data["train"]
        if not train_df.empty and "kl_mean" in train_df.columns:
            series = train_df[["step", "kl_mean"]].dropna()
            if not series.empty:
                axes[1, 0].plot(series["step"], series["kl_mean"], label=f"{run_name}_kl", linewidth=2)
        if not train_df.empty and "mean_divergence" in train_df.columns:
            series = train_df[["step", "mean_divergence"]].dropna()
            if not series.empty:
                axes[1, 0].plot(series["step"], series["mean_divergence"], label=f"{run_name}_div", linewidth=2)
        if not train_df.empty and "clip_fraction" in train_df.columns:
            series = train_df[["step", "clip_fraction"]].dropna()
            if not series.empty:
                axes[1, 1].plot(series["step"], series["clip_fraction"], label=f"{run_name}_clip", linewidth=2)
        if not train_df.empty and "mask_fraction" in train_df.columns:
            series = train_df[["step", "mask_fraction"]].dropna()
            if not series.empty:
                axes[1, 1].plot(series["step"], series["mask_fraction"], label=f"{run_name}_mask", linewidth=2)

    style_axis(axes[0, 0], "train reward", "train_reward")
    style_axis(axes[0, 1], "final eval acc", "eval_acc", xlabel="run")
    style_axis(axes[1, 0], "kl / divergence", "value")
    style_axis(axes[1, 1], "clip / mask fraction", "value")
    for axis in axes.flat:
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

    written = []
    summary_path = save_dir / "summary.png"
    plot_summary(runs, summary_path)
    written.append(summary_path)

    train_specs = [
        ("train_reward", "train reward", "train_reward.png"),
        ("kl_mean", "kl mean", "kl_mean.png"),
        ("mean_divergence", "mean divergence", "mean_divergence.png"),
        ("entropy_mean", "entropy mean", "entropy_mean.png"),
        ("ratio_max", "ratio max", "ratio_max.png"),
        ("clip_fraction", "clip fraction", "clip_fraction.png"),
        ("mask_fraction", "mask fraction", "mask_fraction.png"),
        ("response_length", "response length", "response_length.png"),
        ("gpu_hours", "gpu hours", "gpu_hours.png"),
    ]
    for metric, title, file_name in train_specs:
        path = save_dir / file_name
        if plot_timeseries_metric(runs, metric, "train", path, title):
            written.append(path)

    eval_specs = [
        ("eval_acc", "final eval acc", "eval_acc.png"),
        ("eval_reward", "final eval reward", "eval_reward.png"),
    ]
    for metric, title, file_name in eval_specs:
        path = save_dir / file_name
        if plot_final_bar_metric(runs, metric, "eval", path, title):
            written.append(path)

    system_path = save_dir / "system_usage.png"
    if plot_system_usage(runs, system_path):
        written.append(system_path)

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
