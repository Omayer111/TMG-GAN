import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from thesis_tmg_pipeline.src.utils import init_weights


class TMGGANCDModelTabular(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim // 2, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim // 2, hidden_dim // 4)),
            nn.LeakyReLU(0.2),
        )
        self.score_head = spectral_norm(nn.Linear(hidden_dim // 4, 1))
        self.class_head = nn.Linear(hidden_dim // 4, num_classes)
        self.hidden_status: torch.Tensor | None = None
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        self.hidden_status = hidden
        score = self.score_head(hidden).squeeze(1)
        class_logits = self.class_head(hidden)
        return score, class_logits, hidden
