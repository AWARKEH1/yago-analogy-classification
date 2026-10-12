"""ANNc: CNN-based analogy classifier.

Architecture from Lim, Prade & Richard (2019), "Solving word analogies: a
machine learning perspective", as used in the analogy work of M. Couceiro's
team. The four embeddings a, b, c, d are stacked into an
``emb_size x 4`` "image":

* conv 1: 128 filters 1x2, stride (1, 2)  -> compares a with b and c with d
* conv 2:  64 filters 2x2, stride (2, 2)  -> compares the two pairs
* linear + sigmoid                         -> probability that a:b::c:d is valid
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ANNc(nn.Module):
    def __init__(self, emb_size: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, 128, (1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, (2, 2), stride=(2, 2))
        self.linear = nn.Linear(64 * (emb_size // 2), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, emb_size, 4] (columns a, b, c, d) -> logits [batch]."""
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x.unsqueeze(1)))   # [B, 128, emb, 2]
        x = F.relu(self.conv2(x))                # [B, 64, emb/2, 1]
        return self.linear(x.flatten(1)).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
