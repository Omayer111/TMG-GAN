import torch

from thesis_tmg_pipeline.src.models.tmg_cd_model_tabular import TMGGANCDModelTabular
from thesis_tmg_pipeline.src.models.tmg_generator_tabular import TMGGANGeneratorTabular


def test_tmg_tabular_shapes():
    batch_size = 16
    feature_dim = 20
    num_classes = 5

    generator = TMGGANGeneratorTabular(z_dim=32, feature_dim=feature_dim, hidden_dim=64)
    cd_model = TMGGANCDModelTabular(feature_dim=feature_dim, num_classes=num_classes, hidden_dim=64)

    generated = generator.sample(batch_size, device=torch.device("cpu"))
    score, class_logits, hidden = cd_model(generated)

    assert generated.shape == (batch_size, feature_dim)
    assert score.shape == (batch_size,)
    assert class_logits.shape == (batch_size, num_classes)
    assert hidden.shape == (batch_size, 64)
