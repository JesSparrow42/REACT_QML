import math
import numpy as np
import torch
import torch.nn as nn
from typing import Callable


class Parametrisation(nn.Module):
    def __init__(self):
        super(Parametrisation, self).__init__()

    def forward(self, x):
        return (1 + torch.cos(x)) / 2


class FlipModelNonQubo(nn.Module):
    """A Pytorch model to be utilized by the Binary Bosonic Solver algorithm for non-QUBO formulated problems.

    Handles the parametrized bit flipping of bit strings.
    """

    def __init__(self, dim: int, objective_function: Callable, sampling_factor: int = 1, entropy_penalty: float = 0.0):
        """Initialises the model.

        Args:
            dim (int): Problem dimension i.e. number of binary variables
            objective_function (callable): A function that defines the binary optimisation problem that accepts a bit
                                           string and returns a scalar value.
            sampling_factor (int, optional): Determines the number of classical samples. Defaults to 1.
            entropy_penalty (float): Entropy penalty factor that incentivises binary bit-flip probabilities
        """
        super().__init__()

        self.objective_function = objective_function
        self.dim = dim
        self.sampling_factor = sampling_factor
        self.entropy_penalty = entropy_penalty

        self.alphas = nn.Parameter(0.5 * np.pi * torch.ones((self.dim), requires_grad=True, dtype=torch.float32))

        self.parametrisation = Parametrisation()

    def get_probs(self):
        return self.parametrisation(self.alphas).detach()

    def forward(self, samples):

        return FlipModelAutograd.apply(
            self.alphas,
            samples,
            self.objective_function,
            self.sampling_factor,
            self.parametrisation,
            self.entropy_penalty,
        )

    @torch.no_grad()
    def sample(self, samples):
        n_samples = samples.shape[0] * self.sampling_factor
        probs = self.get_probs()

        flips = torch.bernoulli(probs.repeat(n_samples, 1))
        samples = samples.repeat((self.sampling_factor, 1))
        samples = flips * samples + (1 - flips) * (1 - samples)

        norm = samples.shape[1] * math.log(math.e) / math.e
        E_p = self.entropy_penalty * ((probs + 1e-14) * torch.log(1 / (probs + 1e-14))).sum().squeeze(0) / norm

        e = [self.objective_function(sample) + E_p for sample in samples.numpy()]

        return e, samples.cpu().numpy()


class FlipModelAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alphas, samples, objective_function, sampling_factor, parametrisation, entropy_penalty):
        probs = parametrisation(alphas)
        # Send the variables for the backward to context object (ctx)
        ctx.objective_function = objective_function
        ctx.sampling_factor = sampling_factor
        ctx.entropy_penalty = entropy_penalty
        ctx.parametrisation = parametrisation

        ctx.save_for_backward(alphas, samples)

        n_samples = samples.shape[0] * sampling_factor

        flips = torch.bernoulli(probs.repeat(n_samples, 1))
        samples = samples.repeat((sampling_factor, 1))
        samples = flips * samples + (1 - flips) * (1 - samples)

        samples_unique, samples_counts = np.unique(samples, axis=0, return_counts=True)

        E = sum(
            [objective_function(sample) * count / n_samples for sample, count in zip(samples_unique, samples_counts)]
        )

        norm = samples.shape[1] * math.log(math.e) / math.e
        E_p = entropy_penalty * ((probs + 1e-14) * torch.log(1 / (probs + 1e-14))).sum().squeeze(0) / norm

        return torch.tensor([E + E_p], dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        (alphas, samples) = ctx.saved_tensors

        sampling_factor = ctx.sampling_factor
        objective_function = ctx.objective_function
        parametrisation = ctx.parametrisation
        entropy_penalty = ctx.entropy_penalty

        parameter_shift = np.pi / 6

        derivatives_tensor = torch.zeros_like(alphas)

        for i in range(alphas.shape[0]):
            shifted_alphas = alphas.clone()
            shifted_alphas[i] = shifted_alphas[i] + parameter_shift

            upshifted_expectation = FlipModelAutograd.apply(
                shifted_alphas, samples, objective_function, sampling_factor, parametrisation, entropy_penalty
            )

            shifted_alphas[i] = shifted_alphas[i] - 2 * parameter_shift

            downshifted_expectation = FlipModelAutograd.apply(
                shifted_alphas, samples, objective_function, sampling_factor, parametrisation, entropy_penalty
            )

            # observable_wrt_theta_gradient = (upshifted_expectation - downshifted_expectation) / (2 * parameter_shift)
            observable_wrt_theta_gradient = (upshifted_expectation - downshifted_expectation) / (
                2 * np.sin(parameter_shift)
            )

            derivatives_tensor[i] = torch.sum(grad_output * observable_wrt_theta_gradient)

        return derivatives_tensor, None, None, None, None, None
