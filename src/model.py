import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


def dropout_augment(input_ids, attention_mask, pad_id, drop_prob=0.10):
    """Replace random real tokens with <pad>. The first token is kept."""
    drop = (torch.rand(input_ids.shape, device=input_ids.device) < drop_prob) \
        & attention_mask.bool()
    drop[:, 0] = False
    out = input_ids.clone()
    out[drop] = pad_id
    return out


class DualCLFModel(nn.Module):
    def __init__(self, encoder_name="xlm-roberta-base", proj_dim=256,
                 dropout=0.3, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, proj_dim),
        )
        self.classifier = nn.Linear(hidden, num_labels)

    @staticmethod
    def _pool(hidden, mask):
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.norm(self._pool(out.last_hidden_state, attention_mask))
        z = F.normalize(self.proj(h), dim=-1)
        return z, self.classifier(h)
