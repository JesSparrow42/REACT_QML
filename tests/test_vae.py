import os
import pytest
import torch
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution

# Import your VAE modules from your repository.# Adjust the import paths as necessary.
from GenAI.models.vae import VariationalAutoencoder, VariationalInference

# You may need to adjust the import path below to reference the actual location.
# Here, we assume that BosonSamplerWrapper.BosonLatentGenerator is available.
# We'll override it during tests that require a boson sampler.
# For tests that do not require it, we simply pass None.
# E.g., "your_module_path" could be replaced with the proper package/module name.

@pytest.fixture
def input_shape():
    # Example input shape (e.g. 28x28 for MNIST-like images)
    return (28, 28)

@pytest.fixture
def latent_features():
    return 16

@pytest.mark.parametrize("batch_size", [4, 8])
def test_vae_forward_without_boson_sampler(input_shape, latent_features, batch_size):
    """
    Test the forward pass of the VAE when no boson sampler is provided.
    """
    # Instantiate the VAE with boson_sampler_params set to None
    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_features,
        boson_sampler_params=None
    )
    # Create a dummy batch with shape [batch_size, *input_shape]
    x = torch.randn(batch_size, *input_shape)
    outputs = vae(x)
    # Ensure that all expected keys exist
    for key in ['px', 'pz', 'qz', 'z']:
        assert key in outputs, f"Missing key '{key}' in forward output."
    # Check that the latent variable 'z' has the proper shape
    assert outputs['z'].shape[0] == batch_size
    assert outputs['z'].shape[1] == latent_features


@pytest.mark.parametrize("batch_size", [4, 8])
def test_vae_forward_with_boson_sampler(input_shape, latent_features, batch_size, monkeypatch):
    """
    Test the forward pass of the VAE when a boson sampler is provided.
    We override the __init__ and forward methods of BosonLatentGenerator (from GenAI.BosonSamplerWrapper)
    so that no super() calls occur and a dummy behavior is used.
    """
    # Dummy __init__: bypass the original constructor.
    def dummy_init(self, latent_features, boson_sampler_params):
        # Directly initialize the nn.Module without invoking the problematic super call.
        nn.Module.__init__(self)
        self.latent_features = latent_features

    # Dummy forward: return a normal distribution with the proper shape.
    def dummy_forward(self, *args, **kwargs):
        bs = kwargs.get('batch_size', 1)
        return torch.distributions.Normal(
            torch.zeros(bs, self.latent_features),
            torch.ones(bs, self.latent_features)
        )
    
    # Monkey-patch the __init__ and forward of the BosonLatentGenerator in the GenAI.BosonSamplerWrapper module.
    monkeypatch.setattr("GenAI.BosonSamplerWrapper.BosonLatentGenerator.__init__", dummy_init)
    monkeypatch.setattr("GenAI.BosonSamplerWrapper.BosonLatentGenerator.forward", dummy_forward)

    # Provide dummy parameters (their content won’t affect our dummy).
    dummy_params = {"dummy_key": 1}

    # Instantiate the VAE with boson_sampler_params so that the branch is taken.
    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_features,
        boson_sampler_params=dummy_params
    )

    # Ensure the boson sampler was initialized.
    assert vae.boson_sampler is not None, "Boson sampler should be initialized when parameters are provided."

    # Create a dummy batch.
    x = torch.randn(batch_size, *input_shape)
    outputs = vae(x)

    # Verify that the output contains the expected keys.
    for key in ['px', 'pz', 'qz', 'z']:
        assert key in outputs, f"Missing key '{key}' in forward output with boson sampler."

    # Verify that the latent output 'z' has the expected shape.
    assert outputs['z'].shape == (batch_size, latent_features)

def test_observation_model(input_shape, latent_features):
    """
    Test that the observation model produces a Bernoulli distribution with logits of the correct shape.
    """
    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_features,
        boson_sampler_params=None
    )
    batch_size = 3
    z = torch.randn(batch_size, latent_features)
    obs_dist = vae.observation_model(z)
    # Check if the distribution has logits and that their shape matches the input shape.
    if hasattr(obs_dist, "logits") and obs_dist.logits is not None:
        assert obs_dist.logits.shape[0] == batch_size
        assert obs_dist.logits.shape[1:] == input_shape
    else:
        # Alternatively, sample from the distribution and verify the sample shape.
        samples = obs_dist.sample()
        assert samples.shape == (batch_size, *input_shape)

def test_prior_without_boson_sampler(input_shape, latent_features):
    """
    Test that the prior method returns a distribution with the expected interface when no boson sampler is used.
    """
    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_features,
        boson_sampler_params=None
    )
    batch_size = 5
    prior_dist = vae.prior(batch_size)
    # Check that the returned object is a distribution by verifying it has 'rsample' and 'log_prob' methods.
    assert hasattr(prior_dist, "rsample"), "Prior distribution lacks 'rsample' method."
    assert hasattr(prior_dist, "log_prob"), "Prior distribution lacks 'log_prob' method."

def test_posterior(input_shape, latent_features):
    """
    Test that the posterior method returns a distribution with the expected interface.
    """
    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_features,
        boson_sampler_params=None
    )
    batch_size = 4
    # The encoder expects flattened input.
    x = torch.randn(batch_size, int(np.prod(input_shape)))
    posterior_dist = vae.posterior(x)
    assert hasattr(posterior_dist, "rsample"), "Posterior distribution lacks 'rsample' method."
    assert hasattr(posterior_dist, "log_prob"), "Posterior distribution lacks 'log_prob' method."

def test_variational_inference_loss(input_shape, latent_features):
    """
    Test that the VariationalInference wrapper computes a loss and diagnostics and that the loss is differentiable.
    """
    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_features,
        boson_sampler_params=None
    )
    vi = VariationalInference(beta=1.0)
    batch_size = 2
    x = torch.randn(batch_size, *input_shape)
    # The encoder expects flattened inputs.
    x_flat = x.view(batch_size, -1)
    loss, diagnostics, outputs = vi(vae, x_flat)
    assert loss is not None, "VariationalInference did not return a loss."
    # Verify that the loss is differentiable.
    assert loss.requires_grad, "Loss should require gradients for backpropagation."
    # Optionally, check that diagnostics contains the required keys.
    for key in ['elbo', 'log_px', 'kl']:
        assert key in diagnostics, f"Missing '{key}' in diagnostics from VariationalInference."
