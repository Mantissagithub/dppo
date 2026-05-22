from pathlib import Path
import json
import re

from datasets import load_dataset


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)")


def format_prompt(question):
    return (
        "Solve this problem step by step. "
        "Put the final numeric answer after #### on its own line.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def extract_target(answer_text):
    match = FINAL_ANSWER_RE.search(answer_text)
    if not match:
        return None
    return match.group(1).replace(",", "")


def prepared_data_dir(root):
    return root / "data" / "processed"


def normalize_dataset_name(dataset_name):
    if dataset_name == "gsm8k":
        return "openai/gsm8k"
    return dataset_name


def prepare_gsm8k_subsets(root, train_samples, eval_samples, seed, dataset_name, dataset_config):
    dataset = load_dataset(normalize_dataset_name(dataset_name), dataset_config)
    train_split = dataset["train"].shuffle(seed=seed).select(range(train_samples))
    eval_split = dataset["test"].shuffle(seed=seed).select(range(eval_samples))
    output_dir = prepared_data_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"gsm8k_train_{train_samples}_{seed}.jsonl"
    eval_path = output_dir / f"gsm8k_eval_{eval_samples}_{seed}.jsonl"
    write_split(train_split, train_path)
    write_split(eval_split, eval_path)
    return {"train_path": train_path, "eval_path": eval_path}


def write_split(split, path):
    with open(path, "w", encoding="utf-8") as handle:
        for idx, row in enumerate(split):
            target = extract_target(row["answer"])
            record = {
                "id": idx,
                "question": row["question"],
                "prompt": format_prompt(row["question"]),
                "target": target,
                "answer": row["answer"],
            }
            handle.write(json.dumps(record) + "\n")


def load_prepared_split(root, split, samples, seed=42):
    path = prepared_data_dir(root) / f"gsm8k_{split}_{samples}_{seed}.jsonl"
    if not path.exists():
        config = {"train": samples, "eval": samples}
        prepare_gsm8k_subsets(root, config["train"], config["eval"], seed, "openai/gsm8k", "main")
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            items.append(json.loads(line))
    return items[:samples]
