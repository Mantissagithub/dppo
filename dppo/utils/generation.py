import torch


def generate_text(model, tokenizer, accelerator, prompts, config):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_prompt_length"],
    )
    inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
    with torch.no_grad():
        sequences = accelerator.unwrap_model(model).generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=True,
            temperature=config["temperature"],
            top_p=config["top_p"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = inputs["input_ids"].shape[1]
    generated = sequences[:, prompt_width:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return {"texts": texts, "sequences": sequences, "prompt_width": prompt_width, "inputs": inputs}


def generate_batch(model, tokenizer, accelerator, prompts, config):
    rollout = generate_text(model, tokenizer, accelerator, prompts, config)
    sequences = rollout["sequences"]
    attention_mask = (sequences != tokenizer.pad_token_id).long()
    store_logits = config.get("store_rollout_logits", False)
    with torch.no_grad():
        outputs = model(input_ids=sequences, attention_mask=attention_mask)
    labels = sequences[:, 1:]
    shifted_logits = outputs.logits[:, :-1, :]
    target_logprobs = torch.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    entropy = -(torch.softmax(shifted_logits, dim=-1) * torch.log_softmax(shifted_logits, dim=-1)).sum(dim=-1)
    token_positions = torch.arange(labels.size(1), device=sequences.device).unsqueeze(0) + 1
    token_mask = ((token_positions >= rollout["prompt_width"]) & (labels != tokenizer.pad_token_id)).float()
    response_length = token_mask.sum(dim=1)

    def gather_logprobs(full_logits):
        shifted = full_logits[:, :-1, :]
        return torch.log_softmax(shifted, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    def entropy_from_logits(full_logits):
        shifted = full_logits[:, :-1, :]
        probs = torch.softmax(shifted, dim=-1)
        return -(probs * torch.log_softmax(shifted, dim=-1)).sum(dim=-1)

    def slice_logits(full_logits):
        return full_logits[:, :-1, :]

    rollout.update(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "labels": labels,
            "logprobs": target_logprobs,
            "entropy": entropy,
            "token_mask": token_mask,
            "response_length": response_length,
            "gather_logprobs": gather_logprobs,
            "entropy_from_logits": entropy_from_logits,
            "slice_logits": slice_logits,
        }
    )
    if store_logits:
        rollout["logits"] = shifted_logits
    return rollout
