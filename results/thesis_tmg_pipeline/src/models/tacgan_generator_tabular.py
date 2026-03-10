import torch
from torch import nn


class TACGANTabularGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_classes: int,
        feature_dim: int,
        hidden_dim: int = 256,
        label_dim: int = 32,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        y_embed = self.label_embedding(labels)
        return self.net(torch.cat([z, y_embed], dim=1))

    def sample(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        z = torch.randn(len(labels), self.z_dim, device=device)
        return self.forward(z, labels)
