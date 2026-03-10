import torch

from thesis_tmg_pipeline.src.models.tacgan_discriminator_tabular import TACGANTabularDiscriminator
from thesis_tmg_pipeline.src.models.tacgan_generator_tabular import TACGANTabularGenerator


def test_tacgan_tabular_shapes():
    batch_size = 16
    feature_dim = 20
    num_classes = 5

    generator = TACGANTabularGenerator(z_dim=32, num_classes=num_classes, feature_dim=feature_dim, hidden_dim=64)
    discriminator = TACGANTabularDiscriminator(feature_dim=feature_dim, num_classes=num_classes, hidden_dim=64)

    labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
    samples = generator.sample(labels, device=torch.device("cpu"))
    source_logits, class_logits = discriminator(samples)

    assert samples.shape == (batch_size, feature_dim)
    assert source_logits.shape == (batch_size,)
    assert class_logits.shape == (batch_size, num_classes)
