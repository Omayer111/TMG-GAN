import torch
from torch import nn


class TACGANTabularDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.source_head = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        source_logit = self.source_head(hidden).squeeze(1)
        class_logits = self.class_head(hidden)
        return source_logit, class_logits
