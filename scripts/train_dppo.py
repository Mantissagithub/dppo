from pathlib import Path
import argparse
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dppo.trainers.dppo_trainer import DPPOTrainer


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
        default=[
            str(ROOT / "configs" / "base.yaml"),
            str(ROOT / "configs" / "dppo_topk.yaml"),
        ],
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "dppo_topk"))
    parser.add_argument("--run-name", default="dppo-topk")
    args = parser.parse_args()
    config = load_config(args.config)
    config["run_name"] = args.run_name
    config["repo_root"] = str(ROOT)
    trainer = DPPOTrainer(config, Path(args.output_dir))
    trainer.train()


if __name__ == "__main__":
    main()
