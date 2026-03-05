import torch

from thesis_tmg_pipeline.src.models.sngan_discriminator_tabular import SNGANTabularDiscriminator
from thesis_tmg_pipeline.src.models.sngan_generator_tabular import SNGANTabularGenerator


def test_sngan_tabular_shapes():
    batch_size = 16
    feature_dim = 20
    num_classes = 5

    generator = SNGANTabularGenerator(z_dim=32, num_classes=num_classes, feature_dim=feature_dim, hidden_dim=64)
    discriminator = SNGANTabularDiscriminator(feature_dim=feature_dim, num_classes=num_classes, hidden_dim=64)

    labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
    samples = generator.sample(labels, device=torch.device("cpu"))
    scores = discriminator(samples, labels)

    assert samples.shape == (batch_size, feature_dim)
    assert scores.shape == (batch_size,)
