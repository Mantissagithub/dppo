from pathlib import Path

import torch
from tqdm import tqdm

from dppo.algos.ppo_loss import ppo_loss
from dppo.data.gsm8k import load_prepared_split
from dppo.models.load_model import fail_if_no_cuda, load_model_and_tokenizer
from dppo.rewards.gsm8k_reward import score_completion
from dppo.utils.generation import generate_batch, generate_text
from dppo.utils.hub import save_and_maybe_push
from dppo.utils.logging import ExperimentLogger
from dppo.utils.seed import set_seed
from dppo.utils.system import get_system_metrics
from dppo.utils.timer import Timer


class PPOTrainer:
    def __init__(self, config, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

    def train(self):
        fail_if_no_cuda()
        set_seed(self.config["seed"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger = ExperimentLogger(self.output_dir)
        model, tokenizer, accelerator, _ = load_model_and_tokenizer(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["learning_rate"])
        model, optimizer = accelerator.prepare(model, optimizer)
        train_examples = load_prepared_split(self.output_dir.parent.parent, "train", self.config["train_samples"], self.config["seed"])
        eval_examples = load_prepared_split(self.output_dir.parent.parent, "eval", self.config["eval_samples"], self.config["seed"])

        step = 0
        timer = Timer()
        model.train()
        for _ in range(self.config["num_epochs"]):
            for start in tqdm(range(0, len(train_examples), self.config["batch_size"]), desc="ppo"):
                batch = train_examples[start : start + self.config["batch_size"]]
                rollout = generate_batch(model, tokenizer, accelerator, [item["prompt"] for item in batch], self.config)
                rewards = torch.tensor(
                    [score_completion(text, item["target"])["reward"] for text, item in zip(rollout["texts"], batch)],
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                advantages = rewards - rewards.mean()
                old = rollout["logprobs"].detach()
                token_mask = rollout["token_mask"]
                response_length = rollout["response_length"]
                for _ in range(self.config["ppo_epochs"]):
                    outputs = model(
                        input_ids=rollout["sequences"],
                        attention_mask=rollout["attention_mask"],
                    )
                    new_logprobs = rollout["gather_logprobs"](outputs.logits)
                    entropy = rollout["entropy_from_logits"](outputs.logits)
                    loss, metrics = ppo_loss(
                        new_logprobs=new_logprobs,
                        old_logprobs=old,
                        advantages=advantages,
                        token_mask=token_mask,
                        clip_range=self.config["clip_range"],
                        entropy=entropy,
                        kl_coef=self.config["kl_coef"],
                    )
                    accelerator.backward(loss / self.config["gradient_accumulation_steps"])
                    if ((step + 1) % self.config["gradient_accumulation_steps"]) == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    step += 1
                    elapsed_sec = timer.elapsed()
                    logger.log_train(
                        {
                            "step": step,
                            "loss": float(loss.detach().cpu()),
                            "train_reward": float(rewards.mean().item()),
                            "clip_fraction": metrics["clip_fraction"],
                            "kl_mean": metrics["kl_mean"],
                            "entropy_mean": metrics["entropy_mean"],
                            "ratio_max": metrics["ratio_max"],
                            "response_length": float(response_length.mean().item()),
                            "gpu_hours": elapsed_sec / 3600.0,
                            **get_system_metrics(),
                        }
                    )

        if step % self.config["gradient_accumulation_steps"] != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        eval_metrics = run_eval(model, tokenizer, accelerator, eval_examples, self.config)
        eval_metrics["step"] = step
        eval_metrics["gpu_hours"] = timer.elapsed() / 3600.0
        eval_metrics.update(get_system_metrics())
        logger.log_eval(eval_metrics)
        save_and_maybe_push(model, tokenizer, accelerator, self.output_dir, self.config)


def run_eval(model, tokenizer, accelerator, eval_examples, config):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(eval_examples), config["batch_size"]):
            batch = eval_examples[start : start + config["batch_size"]]
            outputs = generate_text(model, tokenizer, accelerator, [item["prompt"] for item in batch], config)
            for item, text in zip(batch, outputs["texts"]):
                scores.append(score_completion(text, item["target"]))
    model.train()
    return {
        "eval_acc": sum(item["is_correct"] for item in scores) / max(len(scores), 1),
        "eval_parse_rate": sum(item["is_parseable"] for item in scores) / max(len(scores), 1),
        "eval_reward": sum(item["reward"] for item in scores) / max(len(scores), 1),
    }
