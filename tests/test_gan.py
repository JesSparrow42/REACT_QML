import pytest
import torch

from GenAI.models.gan import Generator, Discriminator

@pytest.fixture
def latent_dim():
    return 128

@pytest.fixture
def output_size():
    return 64

@pytest.fixture
def batch_size():
    return 4

def test_generator_forward(latent_dim, output_size, batch_size):
    generator = Generator(latent_dim=latent_dim, output_size=output_size)

    z = torch.randn(batch_size, latent_dim)
    generated_images = generator(z)

    assert generated_images.shape == (batch_size, 1, output_size, output_size)
    assert torch.all(generated_images >= 0) and torch.all(generated_images <= 1), "Generator output should be between 0 and 1"

def test_discriminator_forward(output_size, batch_size):
    discriminator = Discriminator()

    images = torch.randn(batch_size, 1, output_size, output_size)
    validity = discriminator(images)

    assert validity.shape == (batch_size,)

def test_gan_integration(latent_dim, output_size, batch_size):
    generator = Generator(latent_dim=latent_dim, output_size=output_size)
    discriminator = Discriminator()

    z = torch.randn(batch_size, latent_dim)
    generated_images = generator(z)
    validity = discriminator(generated_images)

    assert generated_images.shape == (batch_size, 1, output_size, output_size)
    assert validity.shape == (batch_size,)
