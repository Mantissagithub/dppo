import torch

from dppo.algos.ppo_loss import masked_max, masked_mean


def grpo_loss(new_logprobs, old_logprobs, advantages, token_mask, clip_range, entropy):
    token_advantages = advantages[:, None]
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * token_advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * token_advantages
    surrogate = torch.minimum(unclipped, clipped)
    loss = -masked_mean(surrogate, token_mask)
    entropy_term = masked_mean(entropy, token_mask)
    kl_mean = masked_mean(old_logprobs - new_logprobs, token_mask)
    clip_fraction = masked_mean((torch.abs(ratio - 1.0) > clip_range).float(), token_mask)
    return loss, {
        "entropy_mean": float(entropy_term.detach().cpu()),
        "kl_mean": float(kl_mean.detach().cpu()),
        "ratio_max": float(masked_max(ratio, token_mask).detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
    }
