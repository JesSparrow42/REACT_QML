import pytest
import torch
import numpy as np
from torch import nn

from GenAI.models.graphvae import GraphVAE

@pytest.fixture
def latent_features():
    return 16

@pytest.fixture
def max_nodes():
    return 10

@pytest.fixture
def node_feature_dim():
    return 1

@pytest.fixture
def position_dim():
    return 3

@pytest.fixture
def hidden_dim():
    return 32

@pytest.fixture
def dummy_data(max_nodes, node_feature_dim, position_dim):
    batch_size = 4
    node_features = torch.randn(batch_size, max_nodes, node_feature_dim)
    node_positions = torch.randn(batch_size, max_nodes, position_dim)
    mask = torch.ones(batch_size, max_nodes)
    return node_features, node_positions, mask

@pytest.mark.parametrize("batch_size", [2, 4])
def test_graphvae_forward_without_boson_sampler(latent_features, max_nodes, node_feature_dim, position_dim, hidden_dim, batch_size):
    model = GraphVAE(
        latent_features=latent_features,
        max_nodes=max_nodes,
        node_feature_dim=node_feature_dim,
        position_dim=position_dim,
        hidden_dim=hidden_dim,
        boson_sampler_params=None
    )

    node_features = torch.randn(batch_size, max_nodes, node_feature_dim)
    node_positions = torch.randn(batch_size, max_nodes, position_dim)
    mask = torch.ones(batch_size, max_nodes)

    recon, mu, logvar = model(node_features, node_positions, mask)

    assert recon.shape == (batch_size, max_nodes, node_feature_dim + position_dim)
    assert mu.shape == (batch_size, latent_features)
    assert logvar.shape == (batch_size, latent_features)

def test_graphvae_prior_without_boson_sampler(latent_features, max_nodes):
    model = GraphVAE(
        latent_features=latent_features,
        max_nodes=max_nodes,
        boson_sampler_params=None
    )

    batch_size = 3
    prior_dist = model.prior(batch_size)

    assert hasattr(prior_dist, "rsample"), "Prior distribution lacks 'rsample' method."
    assert hasattr(prior_dist, "log_prob"), "Prior distribution lacks 'log_prob' method."

def test_graphvae_encode(latent_features, max_nodes, node_feature_dim, position_dim, hidden_dim, dummy_data):
    model = GraphVAE(
        latent_features=latent_features,
        max_nodes=max_nodes,
        node_feature_dim=node_feature_dim,
        position_dim=position_dim,
        hidden_dim=hidden_dim
    )

    node_features, node_positions, mask = dummy_data

    mu, logvar = model.encode(node_features, node_positions, mask)

    assert mu.shape == (node_features.size(0), latent_features)
    assert logvar.shape == (node_features.size(0), latent_features)

def test_graphvae_decode(latent_features, max_nodes, node_feature_dim, position_dim, hidden_dim):
    model = GraphVAE(
        latent_features=latent_features,
        max_nodes=max_nodes,
        node_feature_dim=node_feature_dim,
        position_dim=position_dim,
        hidden_dim=hidden_dim
    )

    batch_size = 5
    z = torch.randn(batch_size, latent_features)
    recon = model.decode(z)

    assert recon.shape == (batch_size, max_nodes, node_feature_dim + position_dim)

def test_graphvae_reparameterize(latent_features, max_nodes):
    model = GraphVAE(latent_features=latent_features, max_nodes=max_nodes)

    batch_size = 4
    mu = torch.zeros(batch_size, latent_features)
    logvar = torch.zeros(batch_size, latent_features)

    z = model.reparameterize(mu, logvar)

    assert z.shape == (batch_size, latent_features)
