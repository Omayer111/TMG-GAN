import torch
from torch import nn


class TMGGANGeneratorTabular(nn.Module):
    def __init__(self, z_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.backbone = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, feature_dim)
        self.hidden_status: torch.Tensor | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(z)
        self.hidden_status = hidden
        return self.head(hidden)

    def sample(self, num: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num, self.z_dim, device=device)
        return self.forward(z)
