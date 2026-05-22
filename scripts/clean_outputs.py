from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_RUNS = ["ppo", "grpo", "dppo_topk", "dppo_full"]
TRAIN_ONLY_DROP_PREFIXES = ("eval_",)
EVAL_ONLY_DROP_PREFIXES = ("train_", "loss", "group_reward_std", "clip_fraction", "mask_fraction", "kl_mean", "entropy_mean", "ratio_max", "response_length", "mean_divergence")


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def normalize_columns(df):
    if df.empty:
        return df
    df = df.copy()
    if "phase" in df.columns:
        df = df.drop(columns=["phase"])
    if "mean_topk_divergence" in df.columns and "mean_divergence" not in df.columns:
        df = df.rename(columns={"mean_topk_divergence": "mean_divergence"})
    df = df.dropna(axis=1, how="all")
    return df


def strip_prefixed_columns(df, prefixes):
    if df.empty:
        return df
    drop_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def dedupe_train_df(df):
    if df.empty:
        return df
    df = normalize_columns(df)
    df = strip_prefixed_columns(df, TRAIN_ONLY_DROP_PREFIXES)
    if "step" not in df.columns:
        return df
    df["_row_id"] = range(len(df))
    df = df.sort_values(["step", "_row_id"]).drop_duplicates(subset=["step"], keep="last")
    df = df.drop(columns=["_row_id"]).sort_values("step").reset_index(drop=True)
    return df


def dedupe_eval_df(df):
    if df.empty:
        return df
    df = normalize_columns(df)
    drop_cols = [col for col in df.columns if col in EVAL_ONLY_DROP_PREFIXES]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.drop_duplicates().reset_index(drop=True)


def align_eval_steps(eval_df, train_df):
    if eval_df.empty or train_df.empty or "step" not in eval_df.columns or "step" not in train_df.columns:
        return eval_df
    eval_df = eval_df.copy()
    if eval_df["step"].nunique() == 1 and float(eval_df["step"].iloc[0]) == 0.0:
        eval_df["step"] = float(train_df["step"].max())
    return eval_df


def load_eval_override(outputs_root, run_name):
    eval_dir = outputs_root / f"{run_name}_eval"
    if not eval_dir.exists():
        return pd.DataFrame()
    rows = load_jsonl(eval_dir / "training.log")
    if rows:
        df = pd.DataFrame(rows)
        if "phase" in df.columns:
            df = df[df["phase"] == "eval"].copy()
        return dedupe_eval_df(df)
    csv_path = eval_dir / "eval_metrics.csv"
    if csv_path.exists():
        return dedupe_eval_df(pd.read_csv(csv_path))
    return pd.DataFrame()


def clean_run(outputs_root, run_name):
    run_dir = outputs_root / run_name
    if not run_dir.exists():
        return None

    log_rows = load_jsonl(run_dir / "training.log")
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        if "phase" in log_df.columns:
            train_df = log_df[log_df["phase"] == "train"].copy()
            eval_df = log_df[log_df["phase"] == "eval"].copy()
        else:
            train_df = log_df.copy()
            eval_df = pd.DataFrame()
    else:
        train_csv = run_dir / "train_metrics.csv"
        eval_csv = run_dir / "eval_metrics.csv"
        train_df = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()
        eval_df = pd.read_csv(eval_csv) if eval_csv.exists() else pd.DataFrame()

    train_df = dedupe_train_df(train_df)
    eval_override = load_eval_override(outputs_root, run_name)
    if not eval_override.empty:
        eval_df = eval_override
    else:
        eval_df = dedupe_eval_df(eval_df)
    eval_df = align_eval_steps(eval_df, train_df)

    if not train_df.empty:
        train_df.to_csv(run_dir / "train_metrics.csv", index=False)
        write_jsonl(run_dir / "train_metrics.jsonl", train_df.to_dict(orient="records"))
    if not eval_df.empty:
        eval_df.to_csv(run_dir / "eval_metrics.csv", index=False)
        write_jsonl(run_dir / "eval_metrics.jsonl", eval_df.to_dict(orient="records"))

    clean_rows = []
    for row in train_df.to_dict(orient="records"):
        clean_rows.append({"phase": "train", **row})
    for row in eval_df.to_dict(orient="records"):
        clean_rows.append({"phase": "eval", **row})
    if clean_rows:
        write_jsonl(run_dir / "training.log", clean_rows)

    return {"run_name": run_name, "train_rows": len(train_df), "eval_rows": len(eval_df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default=str(ROOT / "outputs"))
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    cleaned = []
    for run_name in args.runs:
        result = clean_run(outputs_root, run_name)
        if result:
            cleaned.append(result)
    if not cleaned:
        raise SystemExit("no run outputs found to clean")
    for item in cleaned:
        print(f"cleaned {item['run_name']}: train_rows={item['train_rows']} eval_rows={item['eval_rows']}")


if __name__ == "__main__":
    main()
