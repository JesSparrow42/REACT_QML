import pytest
import torch
import warnings
from GenAI.model import VAE_Lightning, GraphVAE_Lightning, GAN_Lightning, UNetLightning, DiffusionLightning

@pytest.fixture
def boson_params():
    return None  # simplify for testing

@pytest.fixture
def latent_features():
    return 16

@pytest.fixture
def max_nodes():
    return 10

@pytest.fixture
def dummy_image_batch():
    batch_size = 2
    return torch.randn(batch_size, 1, 128, 128), torch.randn(batch_size, 1, 128, 128)

@pytest.fixture
def dummy_graph_batch(max_nodes):
    batch_size = 2
    node_features = torch.randn(batch_size, max_nodes, 1)
    node_positions = torch.randn(batch_size, max_nodes, 3)
    mask = torch.ones(batch_size, max_nodes)
    return {"node_features": node_features, "node_positions": node_positions, "mask": mask}

@pytest.mark.filterwarnings("ignore:.*self.trainer.*")
def test_vae_lightning(boson_params, latent_features, dummy_image_batch):
    model = VAE_Lightning(boson_params_to_use=boson_params, lr=1e-3,
                          latent_features=latent_features, output_dir_orig="orig", output_dir_reco="reco")
    assert model(dummy_image_batch[0].view(2, -1))
    loss = model.training_step(dummy_image_batch, 0)
    assert loss is not None
    assert model.configure_optimizers()

@pytest.mark.filterwarnings("ignore:.*self.trainer.*")
def test_graphvae_lightning(boson_params, latent_features, max_nodes, dummy_graph_batch):
    model = GraphVAE_Lightning(boson_params_to_use=boson_params,
                               latent_features=latent_features, max_nodes=max_nodes, lr=1e-3)
    recon, mu, logvar = model(dummy_graph_batch)
    assert recon.shape == (2, max_nodes, 4)
    loss = model.training_step(dummy_graph_batch, 0)
    assert loss is not None
    assert model.configure_optimizers()

@pytest.mark.filterwarnings("ignore:.*self.trainer.*")
def test_gan_lightning(boson_params, latent_features, dummy_image_batch):
    output_size = 64  # explicitly match GAN output size
    model = GAN_Lightning(
        boson_sampler_params=boson_params,
        gen_lr=1e-4, disc_lr=1e-4,
        latent_dim=latent_features, output_size=output_size
    )
    model.on_train_start()

    # Test optimizers configuration
    disc_optimizer, gen_optimizer = model.configure_optimizers()
    assert disc_optimizer is not None
    assert gen_optimizer is not None

    # Test forward pass
    latent = model.sample_latent(dummy_image_batch[0].size(0))
    fake_images = model(latent)
    assert fake_images.shape == (dummy_image_batch[0].size(0), 1, output_size, output_size)

@pytest.mark.filterwarnings("ignore:.*self.trainer.*")
def test_unet_lightning(dummy_image_batch):
    model = UNetLightning(lr=1e-3, output_dir="unet_images")
    pred_ct = model(dummy_image_batch[0])
    assert pred_ct.shape == dummy_image_batch[0].shape
    loss = model.training_step(dummy_image_batch, 0)
    assert loss is not None
    assert model.configure_optimizers()

@pytest.mark.filterwarnings("ignore:.*self.trainer.*")
def test_diffusion_lightning(boson_params, latent_features, dummy_image_batch):
    model = DiffusionLightning(boson_params_to_use=boson_params, lr=1e-3,
                               latent_features=latent_features,
                               output_dir_orig="diff_orig", output_dir_reco="diff_reco")
    loss = model.training_step(dummy_image_batch, 0)
    assert loss is not None
    assert model.configure_optimizers()
