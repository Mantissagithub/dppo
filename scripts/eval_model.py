from pathlib import Path
import argparse
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dppo.models.load_model import fail_if_no_cuda, load_model_and_tokenizer
from dppo.data.gsm8k import load_prepared_split
from dppo.rewards.gsm8k_reward import score_completion
from dppo.utils.generation import generate_text
from dppo.utils.logging import ExperimentLogger
from dppo.utils.seed import set_seed
from dppo.utils.system import get_system_metrics


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
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "eval"))
    args = parser.parse_args()
    config = load_config(args.config)
    fail_if_no_cuda()
    set_seed(config["seed"])
    model, tokenizer, accelerator, _ = load_model_and_tokenizer(config, model_path=args.model_path)
    model = accelerator.prepare(model)
    eval_examples = load_prepared_split(ROOT, "eval", config["eval_samples"])
    logger = ExperimentLogger(Path(args.output_dir))
    scores = []
    for idx in range(0, len(eval_examples), config["batch_size"]):
        batch = eval_examples[idx : idx + config["batch_size"]]
        prompts = [item["prompt"] for item in batch]
        outputs = generate_text(model, tokenizer, accelerator, prompts, config)
        for example, text in zip(batch, outputs["texts"]):
            metric = score_completion(text, example["target"])
            scores.append(metric)
    accuracy = sum(item["is_correct"] for item in scores) / max(len(scores), 1)
    parse_rate = sum(item["is_parseable"] for item in scores) / max(len(scores), 1)
    logger.log_eval(
        {
            "step": 0,
            "eval_acc": accuracy,
            "eval_parse_rate": parse_rate,
            "eval_reward": sum(item["reward"] for item in scores) / max(len(scores), 1),
            "gpu_hours": 0.0,
            **get_system_metrics(),
        }
    )
    print(f"eval_acc={accuracy:.4f}")
    print(f"eval_parse_rate={parse_rate:.4f}")


if __name__ == "__main__":
    main()
