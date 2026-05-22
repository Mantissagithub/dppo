import torch


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_max(values, mask):
    masked = values.masked_fill(mask <= 0, float("-inf"))
    return masked.max()


def ppo_loss(new_logprobs, old_logprobs, advantages, token_mask, clip_range, entropy, kl_coef):
    token_advantages = advantages[:, None]
    log_ratio = new_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * token_advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * token_advantages
    surrogate = torch.minimum(unclipped, clipped)
    pg_loss = -masked_mean(surrogate, token_mask)
    entropy_term = masked_mean(entropy, token_mask)
    approx_kl = masked_mean(old_logprobs - new_logprobs, token_mask)
    loss = pg_loss - kl_coef * entropy_term
    clip_fraction = masked_mean((torch.abs(ratio - 1.0) > clip_range).float(), token_mask)
    return loss, {
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "kl_mean": float(approx_kl.detach().cpu()),
        "entropy_mean": float(entropy_term.detach().cpu()),
        "ratio_max": float(masked_max(ratio, token_mask).detach().cpu()),
    }
