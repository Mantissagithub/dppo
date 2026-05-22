import torch


def topk_tv_divergence(old_logits, new_logits, top_k):
    old_probs = torch.softmax(old_logits, dim=-1)
    new_probs = torch.softmax(new_logits, dim=-1)
    topk = torch.topk(old_probs, k=min(top_k, old_probs.size(-1)), dim=-1)
    old_topk = topk.values
    new_topk = torch.gather(new_probs, dim=-1, index=topk.indices)
    return 0.5 * torch.abs(old_topk - new_topk).sum(dim=-1)
