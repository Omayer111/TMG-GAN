import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from thesis_tmg_pipeline.src.utils import init_weights


class TMGGANCDModelTabular(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()

        h1 = hidden_dim
        h2 = hidden_dim // 2
        h3 = hidden_dim // 4

        self.backbone = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, h1)),
            nn.LayerNorm(h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            spectral_norm(nn.Linear(h1, h2)),
            nn.LayerNorm(h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            spectral_norm(nn.Linear(h2, h3)),
            nn.LayerNorm(h3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.embedding_head = nn.Sequential(
            spectral_norm(nn.Linear(h3, embedding_dim)),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.score_head = spectral_norm(nn.Linear(embedding_dim, 1))
        self.class_head = nn.Linear(embedding_dim, num_classes)

        self.hidden_status: torch.Tensor | None = None
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        embedding = self.embedding_head(hidden)
        self.hidden_status = embedding

        score = self.score_head(embedding).squeeze(1)
        class_logits = self.class_head(embedding)
        return score, class_logits, embedding
