from pathlib import Path
import argparse
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dppo.data.gsm8k import prepare_gsm8k_subsets


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
        help="Repeat for layered configs.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output = prepare_gsm8k_subsets(
        root=ROOT,
        train_samples=config["train_samples"],
        eval_samples=config["eval_samples"],
        seed=config["seed"],
        dataset_name=config["dataset_name"],
        dataset_config=config["dataset_config"],
    )
    print(f"wrote {output['train_path']}")
    print(f"wrote {output['eval_path']}")


if __name__ == "__main__":
    main()
