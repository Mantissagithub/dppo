import torch

from dppo.algos.divergence import full_tv_divergence, topk_tv_divergence
from dppo.algos.ppo_loss import masked_max, masked_mean


def dppo_loss(
    new_logprobs,
    old_logprobs,
    old_logits,
    new_logits,
    advantages,
    token_mask,
    top_k,
    delta,
    entropy,
    divergence_mode,
):
    ratio = torch.exp(new_logprobs - old_logprobs)
    if divergence_mode == "full":
        divergence = full_tv_divergence(old_logits, new_logits)
    else:
        divergence = topk_tv_divergence(old_logits, new_logits, top_k=top_k)
    token_advantages = advantages[:, None]
    positive_block = (token_advantages > 0) & (ratio > 1.0) & (divergence > delta)
    negative_block = (token_advantages < 0) & (ratio < 1.0) & (divergence > delta)
    keep_mask = (~(positive_block | negative_block)).float()
    masked_objective = keep_mask * ratio * token_advantages * token_mask
    loss = -masked_objective.sum() / token_mask.sum().clamp_min(1.0)
    kl_mean = masked_mean(old_logprobs - new_logprobs, token_mask)
    entropy_mean = masked_mean(entropy, token_mask)
    return loss, {
        "mean_divergence": float(((divergence * token_mask).sum() / token_mask.sum().clamp_min(1.0)).detach().cpu()),
        "mask_fraction": float((1.0 - ((keep_mask * token_mask).sum() / token_mask.sum().clamp_min(1.0))).detach().cpu()),
        "kl_mean": float(kl_mean.detach().cpu()),
        "entropy_mean": float(entropy_mean.detach().cpu()),
        "ratio_max": float(masked_max(ratio, token_mask).detach().cpu()),
    }
