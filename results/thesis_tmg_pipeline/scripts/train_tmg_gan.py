import torch
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import random
from thesis_tmg_pipeline.src.models.tmg_cd_model_tabular import TMGGANCDModelTabular
from thesis_tmg_pipeline.src.models.tmg_generator_tabular import TMGGANGeneratorTabular
from thesis_tmg_pipeline.src.utils import compute_metrics, load_dataset
import os
import math


# Generator training loss (class-separation cosine loss)
def generator_loss_fn(real_features, fake_features, class_target, generator, discriminator, device, hidden_loss_weight=1.0):
    label_loss = nn.CrossEntropyLoss()(fake_features, class_target)  # Classification loss
    hidden_loss = -cosine_similarity(real_features, fake_features, dim=1).mean()  # Class-separation loss

    return -discriminator.mean() + label_loss + hidden_loss_weight * hidden_loss


def train_gan_and_augment_data(config, device, data, num_classes):
    generator = TMGGANGeneratorTabular(config.z_dim, data["input_dim"], config.gan_hidden_dim).to(device)
    discriminator = TMGGANCDModelTabular(data["input_dim"], num_classes).to(device)

    optimizer_gen = Adam(generator.parameters(), lr=config.gan_lr, betas=(0.5, 0.999))
    optimizer_disc = Adam(discriminator.parameters(), lr=config.gan_lr, betas=(0.5, 0.999))

    train_data = DataLoader(data["x_train"], batch_size=config.batch_size, shuffle=True)
    for epoch in range(config.gan_epochs):
        generator.train()
        discriminator.train()

        for batch in train_data:
            real_samples = batch[0].to(device)
            class_target = batch[1].to(device)

            # Train discriminator on real data
            optimizer_disc.zero_grad()
            real_logits = discriminator(real_samples)
            loss_real = nn.BCEWithLogitsLoss()(real_logits, torch.ones_like(real_logits))

            # Train discriminator on fake data
            fake_samples = generator.sample(config.batch_size, device=device)
            fake_logits = discriminator(fake_samples)
            loss_fake = nn.BCEWithLogitsLoss()(fake_logits, torch.zeros_like(fake_logits))

            # Total discriminator loss
            loss_disc = (loss_real + loss_fake) / 2
            loss_disc.backward()
            optimizer_disc.step()

            # Train generator
            optimizer_gen.zero_grad()
            fake_samples = generator.sample(config.batch_size, device=device)
            real_features = discriminator.extract_features(real_samples)
            fake_features = discriminator.extract_features(fake_samples)
            loss_gen = generator_loss_fn(real_features, fake_features, class_target, generator, discriminator, device)

            loss_gen.backward()
            optimizer_gen.step()

        print(f"Epoch {epoch + 1}/{config.gan_epochs}: D Loss: {loss_disc.item()} | G Loss: {loss_gen.item()}")

    # Augment the dataset with generated samples
    augmented_data = generate_augmented_data(generator, discriminator, data, num_classes, device)
    return augmented_data


# The final method will use these augmented data to train the classifier
def generate_augmented_data(generator, discriminator, data, num_classes, device):
    augmented_data = []

    # Generate samples for each class
    for class_id in range(num_classes):
        class_target = torch.full((data["x_train"].size(0),), class_id, dtype=torch.long, device=device)
        samples = generator.sample(num_classes * 1000, device=device)  # Generate 1000 samples for each class
        augmented_data.append(samples)

    augmented_data = torch.cat(augmented_data, dim=0)
    return augmented_data