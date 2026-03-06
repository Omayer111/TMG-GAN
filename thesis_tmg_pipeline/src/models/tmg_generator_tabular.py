import torch
from torch import nn

from thesis_tmg_pipeline.src.utils import init_weights


class TMGGANGeneratorTabular(nn.Module):
    def __init__(self, z_dim: int, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.z_dim = z_dim
        self.backbone = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.Sigmoid(),
        )
        self.hidden_status: torch.Tensor | None = None
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(z)
        self.hidden_status = hidden
        return self.head(hidden)

    def sample(self, num: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num, self.z_dim, device=device)
        return self.forward(z)
