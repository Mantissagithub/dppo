from pathlib import Path
import argparse
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dppo.utils.hub import build_repo_id_from_parts, download_model_repo


def load_config(config_paths):
    config = {}
    for path in config_paths:
        with open(path, "r", encoding="utf-8") as handle:
            config.update(yaml.safe_load(handle) or {})
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=[str(ROOT / "configs" / "base.yaml")],
    )
    parser.add_argument("--repo-id")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--local-dir", default=str(ROOT / "downloads"))
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

    local_dir = Path(args.local_dir) / repo_id.replace("/", "__")
    path = download_model_repo(repo_id=repo_id, repo_root=ROOT, local_dir=local_dir)
    print(f"downloaded {repo_id} to {path}")


if __name__ == "__main__":
    main()
