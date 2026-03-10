import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class SNGANTabularDiscriminator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        label_dim: int = 32,
    ):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim + label_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        y_embed = self.label_embedding(labels)
        score = self.net(torch.cat([x, y_embed], dim=1))
        return score.squeeze(1)
