import torch

from thesis_tmg_pipeline.src.models.tmg_cd_model_tabular import TMGGANCDModelTabular
from thesis_tmg_pipeline.src.models.tmg_generator_tabular import TMGGANGeneratorTabular


def test_tmg_tabular_shapes():
    batch_size = 16
    feature_dim = 20
    num_classes = 5
    hidden_dim = 128  # must be divisible by 4 for CD model architecture

    generator = TMGGANGeneratorTabular(z_dim=32, feature_dim=feature_dim, hidden_dim=hidden_dim)
    cd_model = TMGGANCDModelTabular(feature_dim=feature_dim, num_classes=num_classes, hidden_dim=hidden_dim)

    generated = generator.sample(batch_size, device=torch.device("cpu"))
    score, class_logits, hidden = cd_model(generated)

    assert generated.shape == (batch_size, feature_dim)
    assert score.shape == (batch_size,)
    assert class_logits.shape == (batch_size, num_classes)
    # CD backbone output is hidden_dim // 4
    assert hidden.shape == (batch_size, hidden_dim // 4)
    # Generator hidden is hidden_dim // 2
    assert generator.hidden_status.shape == (batch_size, hidden_dim // 2)
    # Output bounded [0, 1] by Sigmoid
    assert generated.min() >= 0.0
    assert generated.max() <= 1.0
