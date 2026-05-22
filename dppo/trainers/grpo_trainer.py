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
        update_batch_size = self.config.get("grpo_update_batch_size", self.config["batch_size"] * generations)
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
                    chunk_losses = []
                    chunk_metrics = []
                    total_examples = rollout["sequences"].size(0)
                    for chunk_start in range(0, total_examples, update_batch_size):
                        chunk_end = chunk_start + update_batch_size
                        chunk_sequences = rollout["sequences"][chunk_start:chunk_end]
                        chunk_attention_mask = rollout["attention_mask"][chunk_start:chunk_end]
                        chunk_labels = rollout["labels"][chunk_start:chunk_end]
                        chunk_old_logprobs = rollout["logprobs"][chunk_start:chunk_end].detach()
                        chunk_token_mask = rollout["token_mask"][chunk_start:chunk_end]
                        chunk_advantages = advantages[chunk_start:chunk_end]

                        outputs = model(
                            input_ids=chunk_sequences,
                            attention_mask=chunk_attention_mask,
                        )
                        shifted_logits = outputs.logits[:, :-1, :]
                        new_logprobs = torch.log_softmax(shifted_logits, dim=-1).gather(
                            dim=-1,
                            index=chunk_labels.unsqueeze(-1),
                        ).squeeze(-1)
                        probs = torch.softmax(shifted_logits, dim=-1)
                        entropy = -(probs * torch.log_softmax(shifted_logits, dim=-1)).sum(dim=-1)
                        loss, metrics = grpo_loss(
                            new_logprobs=new_logprobs,
                            old_logprobs=chunk_old_logprobs,
                            advantages=chunk_advantages,
                            token_mask=chunk_token_mask,
                            clip_range=self.config["clip_range"],
                            entropy=entropy,
                        )
                        scale = chunk_sequences.size(0) / total_examples
                        accelerator.backward((loss * scale) / self.config["gradient_accumulation_steps"])
                        chunk_losses.append(float(loss.detach().cpu()))
                        chunk_metrics.append(metrics)
                        del outputs, new_logprobs, entropy, loss

                    if ((step + 1) % self.config["gradient_accumulation_steps"]) == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    step += 1
                    elapsed_sec = timer.elapsed()
                    logger.log_train(
                        {
                            "step": step,
                            "loss": sum(chunk_losses) / max(len(chunk_losses), 1),
                            "train_reward": float(rewards.mean().item()),
                            "group_reward_std": float(group_std.mean().item()),
                            "clip_fraction": sum(item["clip_fraction"] for item in chunk_metrics) / max(len(chunk_metrics), 1),
                            "kl_mean": sum(item["kl_mean"] for item in chunk_metrics) / max(len(chunk_metrics), 1),
                            "entropy_mean": sum(item["entropy_mean"] for item in chunk_metrics) / max(len(chunk_metrics), 1),
                            "ratio_max": max(item["ratio_max"] for item in chunk_metrics),
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
