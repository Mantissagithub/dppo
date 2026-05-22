from pathlib import Path
import argparse
import json
import sys
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dppo.utils.hub import build_repo_id_from_parts, download_model_repo, get_hf_token


def load_config(config_paths):
    config = {}
    for path in config_paths:
        with open(path, "r", encoding="utf-8") as handle:
            config.update(yaml.safe_load(handle) or {})
    return config


def load_training_log(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_training_log(df, save_path):
    train_df = df[df["phase"] == "train"].copy()
    eval_df = df[df["phase"] == "eval"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    if not train_df.empty and "train_reward" in train_df:
        axes[0, 0].plot(train_df["step"], train_df["train_reward"])
    if not eval_df.empty and "eval_acc" in eval_df:
        axes[0, 1].plot(eval_df["step"], eval_df["eval_acc"])
    if not train_df.empty and "kl_mean" in train_df:
        axes[0, 2].plot(train_df["step"], train_df["kl_mean"], label="kl_mean")
    if not train_df.empty and "mean_divergence" in train_df:
        axes[0, 2].plot(train_df["step"], train_df["mean_divergence"], label="mean_divergence")
    if not train_df.empty and "entropy_mean" in train_df:
        axes[1, 0].plot(train_df["step"], train_df["entropy_mean"])
    if not train_df.empty:
        if "clip_fraction" in train_df:
            axes[1, 1].plot(train_df["step"], train_df["clip_fraction"], label="clip_fraction")
        if "mask_fraction" in train_df:
            axes[1, 1].plot(train_df["step"], train_df["mask_fraction"], label="mask_fraction")
    if not train_df.empty and "response_length" in train_df:
        axes[1, 2].plot(train_df["step"], train_df["response_length"], label="response_length")
    if not train_df.empty and "gpu_hours" in train_df:
        axes[1, 2].plot(train_df["step"], train_df["gpu_hours"], label="gpu_hours")

    axes[0, 0].set_title("train reward")
    axes[0, 1].set_title("eval acc")
    axes[0, 2].set_title("kl / divergence")
    axes[1, 0].set_title("entropy")
    axes[1, 1].set_title("clip / mask fraction")
    axes[1, 2].set_title("response length / gpu hours")

    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def infer_method(repo_id):
    name = repo_id.split("/")[-1]
    for method in ["dppo-topk", "grpo", "ppo"]:
        if name.endswith(method):
            return method
    return name


def build_model_card(repo_id, config, df, plot_name):
    train_df = df[df["phase"] == "train"].copy()
    eval_df = df[df["phase"] == "eval"].copy()
    final_train = train_df.iloc[-1].to_dict() if not train_df.empty else {}
    final_eval = eval_df.iloc[-1].to_dict() if not eval_df.empty else {}
    method = infer_method(repo_id)
    lines = [
        f"# {repo_id}",
        "",
        "small single-gpu rl post-training run on a gsm8k subset.",
        "",
        "## summary",
        "",
        f"- base model: `{config['model_name']}`",
        f"- method: `{method}`",
        f"- dataset: `{config['dataset_name']}` config `{config['dataset_config']}`",
        f"- train / eval samples: `{config['train_samples']}` / `{config['eval_samples']}`",
        f"- prompt format: step-by-step reasoning with final answer after `####`",
        "",
        "## reward",
        "",
        "- `+1.0` if the final numeric answer is correct",
        "- `+0.1` if the output has a parseable final answer",
        "- `0.0` otherwise",
        "",
        "## final metrics",
        "",
        f"- eval_acc: `{final_eval.get('eval_acc', 'n/a')}`",
        f"- eval_reward: `{final_eval.get('eval_reward', 'n/a')}`",
        f"- train_reward: `{final_train.get('train_reward', 'n/a')}`",
        f"- kl_mean: `{final_train.get('kl_mean', 'n/a')}`",
        f"- entropy_mean: `{final_train.get('entropy_mean', 'n/a')}`",
        f"- ratio_max: `{final_train.get('ratio_max', 'n/a')}`",
        f"- response_length: `{final_train.get('response_length', 'n/a')}`",
        f"- gpu_hours: `{final_eval.get('gpu_hours', final_train.get('gpu_hours', 'n/a'))}`",
        "",
        "## artifacts",
        "",
        f"- training log: [`training.log`](./training.log)",
        f"- plots: ![{plot_name}](./{plot_name})",
        "",
        "## usage",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'repo_id = "{repo_id}"',
        "tokenizer = AutoTokenizer.from_pretrained(repo_id)",
        "model = AutoModelForCausalLM.from_pretrained(repo_id)",
        "```",
        "",
        "## notes",
        "",
        "- this is a small controlled experiment, not a benchmark",
        "- see `training.log` for the full tracked metrics over time",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=[str(ROOT / "configs" / "base.yaml")],
    )
    parser.add_argument("--repo-id")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-log")
    parser.add_argument("--plot-name", default="training_curves.png")
    args = parser.parse_args()

    config = load_config(args.config)
    repo_id = args.repo_id or build_repo_id_from_parts(
        repo_root=ROOT,
        model_name=config["model_name"],
        dataset_name=config["dataset_name"],
        run_name=args.run_name,
    )
    if not repo_id:
        raise SystemExit("missing HF_USERNAME in .env, or pass --repo-id explicitly")

    token = get_hf_token(ROOT)
    if not token:
        raise SystemExit("missing HF_TOKEN in .env")

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir) / "repo"
        download_model_repo(repo_id=repo_id, repo_root=ROOT, local_dir=work_dir)

        training_log = Path(args.training_log) if args.training_log else work_dir / "training.log"
        if not training_log.exists():
            raise SystemExit(f"training log not found at {training_log}")

        df = load_training_log(training_log)
        plot_path = work_dir / args.plot_name
        plot_training_log(df, plot_path)
        readme_path = work_dir / "README.md"
        readme_path.write_text(build_model_card(repo_id, config, df, args.plot_name), encoding="utf-8")

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        api.upload_file(
            path_or_fileobj=str(plot_path),
            path_in_repo=args.plot_name,
            repo_id=repo_id,
            token=token,
        )
        print(f"updated model card for {repo_id}")


if __name__ == "__main__":
    main()
