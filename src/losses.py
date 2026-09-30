import torch
import torch.nn.functional as F
from torch import nn


def nt_xent(z1, z2, temperature=0.07):
    """Self-supervised NT-Xent. z1, z2: [N, D] embeddings of two views."""
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -9e15)
    pos = torch.cat([(z1 * z2).sum(-1), (z1 * z2).sum(-1)]) / temperature
    return (torch.logsumexp(sim, dim=1) - pos).mean()


def supervised_contrastive(z, y, temperature=0.07):
    """Supervised contrastive loss (Khosla et al., 2020)."""
    z = F.normalize(z, dim=-1)
    n = z.size(0)
    sim = z @ z.t() / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -9e15)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    y = y.view(-1, 1)
    pos = (y == y.t()).float() - self_mask.float()
    pos_count = pos.sum(1).clamp(min=1.0)
    return -((pos * log_prob).sum(1) / pos_count).mean()


class FocalLoss(nn.Module):
    """L_FL = -alpha_y (1 - p_y)^gamma log p_y, alpha given per class."""

    def __init__(self, alpha=(0.75, 0.25), gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha[targets] * (1 - pt) ** self.gamma * ce).mean()
