#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-warning">
# This is an exercise to be handed in on Peergrade
# </div>

# In[1]:


import sys
sys.path.append(r'C:\Users\Jonat\Desktop\Deep Learning\Assignments\Week 7')


# In[2]:


from typing import *
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
get_ipython().run_line_magic('matplotlib', 'nbagg')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")

try:
    from plotting import make_vae_plots
except Exception as ex:
    print(f"If using Colab, you may need to upload `plotting.py`. \
          \nIn the left pannel, click `Files > upload to session storage` and select the file `plotting.py` from your computer \
          \n---------------------------------------------")
    print(ex)


# In[7]:


import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """Sample `z ~ N(z | mu, sigma)` (with the reparameterization trick)"""
        epsilon = self.sample_epsilon()  # Generate noise from N(0, 1)
        z = self.mu + self.sigma * epsilon  # Apply the reparameterization trick: z = mu + sigma * epsilon
        return z

    def log_prob(self, z: Tensor) -> Tensor:
        """Return the log probability: log `p(z)`"""
        log_scale = torch.log(self.sigma)
        return -0.5 * ((z - self.mu) ** 2 / self.sigma**2 + 2 * log_scale + torch.log(torch.tensor(2 * torch.pi)))


# In[8]:


# test your implementation

def test_normal_distribution():
    """a few safety checks for your implementation"""
    N = 1000000
    ones = torch.ones(torch.Size((N,)))
    mu = 1.224 * ones
    sigma = 0.689 * ones
    dist = ReparameterizedDiagonalGaussian(mu, sigma.log())
    z = dist.sample()
    
    # Expected value E[N(0, 1)] = 0
    expected_z = z.mean()
    diff = (expected_z - mu.mean())**2
    assert diff < 1e-3, f"diff = {diff}, expected_z = {expected_z}"
    
    # Variance E[z**2 - E[z]**2]
    var_z = (z**2 - expected_z**2).mean()
    diff = (var_z - sigma.pow(2).mean())**2
    assert diff < 1e-3, f"diff = {diff}, var_z = {var_z}"
    
    # log p(z)
    from torch.distributions import Normal
    base = Normal(loc=mu, scale=sigma)
    diff = ((base.log_prob(z) - dist.log_prob(z))**2).mean()
    assert diff < 1e-3, f"diff = {diff}"

test_normal_distribution()   

n_samples = 10000
mu = torch.tensor([[0, 1]])
sigma = torch.tensor([[0.5 , 3]])
ones = torch.ones((1000,2))
p = ReparameterizedDiagonalGaussian(mu=mu*ones, log_sigma=(sigma*ones).log())
samples = p.sample()
data = pd.DataFrame({"x": samples[:, 0], "y": samples[:, 1]})
g = sns.jointplot(
    data=data,
    x="x",y="y",
    kind="hex",
    ratio=10
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle(r"$\mathcal{N}(\mathbf{y} \mid \mu, \sigma)$")
plt.show()


# ### 2. Bernoulli Distribution
# 
# The Bernoulli distribution is a good fit when modelling binary outcomes (e.g. coin flipping). Given a binary random variable $X$ with outcomes $y \in \{0, 1\}$, the probability density of the Bernoulli model with a parameter $\theta$ is defined as
# $$\mathcal{B}( y \mid \theta) = \theta^{y} (1-\theta)^{1-y},\quad \theta \in [0,1]$$
# 
# **Exercise 1**: Import the `Bernoulli` from the [torch.distributions](https://pytorch.org/docs/stable/distributions.html) package.

# In[9]:


from torch.distributions import Bernoulli # <- your code

p = Bernoulli(logits=torch.zeros((1000,)))
plt.figure(figsize=(12, 3))
sns.distplot(p.sample())
plt.title(r"$\mathcal{B}(\mathbf{y} \mid \mathbf{\theta})$")
plt.show()


# In[14]:


import torch
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust logits for different probabilities
logit_value = 2.0  # Example logit value, leading to a high probability of '1'

# Create a Bernoulli distribution with the given logit value
p = Bernoulli(logits=torch.full((1000,), logit_value))
plt.figure(figsize=(12, 3))
sns.distplot(p.sample())
plt.title(r"$\mathcal{B}(\mathbf{y} \mid \mathbf{\theta})$")
plt.show()


# ## Dataset: MNIST
# 
# First let us load the MNIST dataset and plot a few examples. We only load a limited amount of classes, controlled through the `classes` variable, to speed up training.

# In[33]:


from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce

# Flatten the images into a vector
flatten = lambda x: ToTensor()(x).view(28**2)

# Define the train and test sets
dset_train = MNIST("./", train=True,  transform=flatten, download=True)
dset_test  = MNIST("./", train=False, transform=flatten)

# The digit classes to use
classes = [0, 1, 4, 9]

def stratified_sampler(labels):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)


batch_size = 64
eval_batch_size = 100
# The loaders perform the actual work
train_loader = DataLoader(dset_train, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.train_labels))
test_loader  = DataLoader(dset_test, batch_size=eval_batch_size, 
                          sampler=stratified_sampler(dset_test.test_labels))


# In[34]:


#plot a few MNIST examples
f, axarr = plt.subplots(4, 16, figsize=(16, 4))

# Load a batch of images into memory
images, labels = next(iter(train_loader))

for i, ax in enumerate(axarr.flat):
    ax.imshow(images[i].view(28, 28), cmap="binary_r")
    ax.axis('off')
    
plt.suptitle('MNIST handwritten digits')
plt.show()


# ## Building the model
# When defining the model the latent layer must act as a bottleneck of information, so that we ensure that we find a strong internal representation. We initialize the VAE with 1 hidden layer in the encoder and decoder using relu units as non-linearity.

# In[51]:


class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(0.2),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=128, out_features=2 * latent_features)
        )
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=self.observation_features)
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_logits = self.decoder(z)
        px_logits = px_logits.view(-1, *self.input_shape) # reshape the output
        return Bernoulli(logits=px_logits, validate_args=False)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}


latent_features = 2
vae = VariationalAutoencoder(images[0].shape, latent_features)
print(vae)


# ## Implement a module for Variational Inference
# 
# **Exercise 1**: implement `elbo` ($\mathcal{L}$) and `beta_elbo` ($\mathcal{L}^\beta$)

# In[52]:


def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta
        
    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) ] - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px - kl # <- your code here
        beta_elbo = log_px - self.beta * kl # <- your code here
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs
        


# In[53]:


vi = VariationalInference(beta=1.0)
loss, diagnostics, outputs = vi(vae, images)
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")


# ## Training and Evaluation
# 
# ### Initialize the model, evaluator and optimizer

# In[63]:


from collections import defaultdict
# define the models, evaluator and optimizer

# VAE
latent_features = 10
vae = VariationalAutoencoder(images[0].shape, latent_features)

# Evaluator: Variational Inference
beta = 0.1
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

epoch = 0


# ### Training Loop
# 
# **plotting guide**:
# 
# * 1st row: Reproducing the figure from the begining of the Notebook.
#     * (Left) Data. 
#     * (Middle) Latent space: the large gray disk reprensents the prior (radius = $2\sigma$), each point represents a latent sample $\mathbf{z}$. The smaller ellipses represent the distributions $q_\phi(\mathbf{z} | \mathbf{x})$  (radius = $2\sigma$). When using $\geq 2$ latent features, dimensionality reduction is applied using t-SNE and only samples $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$ are displayed. 
#     * (Right) samples from $p_\theta(\mathbf{x} | \mathbf{z})$.
# 
# * 2nd row: Training curves
# 
# * 3rd row: Latent samples. 
#     * (Left) Prior samples $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim p(\mathbf{z})$ 
#     * (Middle) Latent Interpolations. For each row: $\mathbf{x} \sim p_\theta(\mathbf{x} | t \cdot \mathbf{z}_1 + (1-t) \cdot \mathbf{z}_2), \mathbf{z}_1, \mathbf{z}_2 \sim p(\mathbf{z}), t=0 \dots 1$. 
#     * (Right): Sampling $\mathbf{z}$ from a grid [-3:3, -3:3] $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim \operatorname{grid}(-3:3, -3:3)$ (only available for 2d latent space).
# 
# **NOTE** this will take a while on CPU.

# In[64]:


num_epochs = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
vae = vae.to(device)

# training..
while epoch < num_epochs:
    epoch+= 1
    training_epoch_data = defaultdict(list)
    vae.train()
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y in train_loader:
        x = x.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # gather data for the current bach
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
            

    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()
        
        # Just load a single batch from the test loader
        x, y = next(iter(test_loader))
        x = x.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)
        
        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
    
    # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    make_vae_plots(vae, x, y, outputs, training_data, validation_data)
