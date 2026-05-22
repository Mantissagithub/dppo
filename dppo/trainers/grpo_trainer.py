from pathlib import Path

import torch
from tqdm import tqdm

from dppo.algos.grpo_loss import grpo_loss
from dppo.data.gsm8k import load_prepared_split
from dppo.models.load_model import fail_if_no_cuda, load_model_and_tokenizer
from dppo.rewards.gsm8k_reward import score_completion
from dppo.trainers.ppo_trainer import run_eval
from dppo.utils.generation import generate_batch
from dppo.utils.hub import save_and_maybe_push
from dppo.utils.logging import ExperimentLogger
from dppo.utils.seed import set_seed
from dppo.utils.system import get_system_metrics
from dppo.utils.timer import Timer


class GRPOTrainer:
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
        generations = self.config["num_generations"]
        model.train()
        for _ in range(self.config["num_epochs"]):
            for start in tqdm(range(0, len(train_examples), self.config["batch_size"]), desc="grpo"):
                base_batch = train_examples[start : start + self.config["batch_size"]]
                prompts = []
                targets = []
                for item in base_batch:
                    prompts.extend([item["prompt"]] * generations)
                    targets.extend([item["target"]] * generations)
                rollout = generate_batch(model, tokenizer, accelerator, prompts, self.config)
                rewards = torch.tensor(
                    [score_completion(text, target)["reward"] for text, target in zip(rollout["texts"], targets)],
                    device=accelerator.device,
                    dtype=torch.float32,
                ).view(len(base_batch), generations)
                group_mean = rewards.mean(dim=1, keepdim=True)
                group_std = rewards.std(dim=1, keepdim=True, unbiased=False)
                advantages = ((rewards - group_mean) / (group_std + 1e-8)).reshape(-1)
                response_length = rollout["response_length"]
                for _ in range(self.config["grpo_epochs"]):
                    outputs = model(
                        input_ids=rollout["sequences"],
                        attention_mask=rollout["attention_mask"],
                    )
                    new_logprobs = rollout["gather_logprobs"](outputs.logits)
                    entropy = rollout["entropy_from_logits"](outputs.logits)
                    loss, metrics = grpo_loss(
                        new_logprobs=new_logprobs,
                        old_logprobs=rollout["logprobs"].detach(),
                        advantages=advantages,
                        token_mask=rollout["token_mask"],
                        clip_range=self.config["clip_range"],
                        entropy=entropy,
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
                            "group_reward_std": float(group_std.mean().item()),
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
