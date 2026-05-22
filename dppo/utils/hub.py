from pathlib import Path
import re

from huggingface_hub import HfApi, create_repo


def load_env_file(path: Path):
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def slugify(value):
    value = value.strip().lower().replace("/", "-").replace("_", "-")
    value = re.sub(r"[^a-z0-9.-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-.")


def build_repo_id(config):
    env = load_env_file(Path(config["repo_root"]) / ".env")
    username = env.get("HF_USERNAME")
    if not username:
        return None
    base_model = slugify(config["model_name"].split("/")[-1])
    dataset_name = slugify(config["dataset_name"])
    run_name = slugify(config["run_name"])
    return f"{username}/{base_model}-{dataset_name}-{run_name}"


def save_and_maybe_push(model, tokenizer, accelerator, output_dir: Path, config):
    model_dir = output_dir / "final_model"
    if accelerator.is_main_process:
        model_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    accelerator.wait_for_everyone()

    env = load_env_file(Path(config["repo_root"]) / ".env")
    token = env.get("HF_TOKEN")
    repo_id = build_repo_id(config)
    if not token or not repo_id:
        if accelerator.is_main_process:
            print("hf push skipped because HF_USERNAME or HF_TOKEN is missing in .env")
        return

    if accelerator.is_main_process:
        create_repo(repo_id=repo_id, token=token, exist_ok=True)
        accelerator.unwrap_model(model).push_to_hub(repo_id=repo_id, token=token)
        tokenizer.push_to_hub(repo_id=repo_id, token=token)
        training_log = output_dir / "training.log"
        if training_log.exists():
            HfApi().upload_file(
                path_or_fileobj=str(training_log),
                path_in_repo="training.log",
                repo_id=repo_id,
                token=token,
            )
        print(f"pushed model to {repo_id}")
