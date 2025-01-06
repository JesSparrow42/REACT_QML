import torch
import numpy as np
from typing import Callable

from ptseries.models.observables import Observable


class ObservableNonQubo(Observable):
    """A custom observable for the PTLayer that is utilised by the Binary Bosonic Solver algorithm.

    Used for non-QUBO formulated binary optimisation problems.
    This observable receives the samples from the boson sampler, flips and computes the average energy.
    Note that the flip probabilities of the observable are not trained inside the observable and they must be
    updated from the training code.
    """

    descr = "binary-non-qubo"
    sample_format = "array"

    def __init__(self, objective_function: Callable, sampling_factor: int = 1, padding: int = 0):
        """Initialises the observable.

        Args:
            objective_function (callable): A function that defines the binary optimisation problem that accepts a bit
                                           string and returns a scalar value.
            sampling_factor (int, optional): Determines the number of classical samples. Defaults to 1.
            padding (int, optional): Number of padding modes. Defaults to 0.
        """
        self.padding = padding
        self.objective_function = objective_function
        self.sampling_factor = sampling_factor
        self.samples = None

        self.tensor_ones = None
        self.samples_dim = None

        self.probs = None

    def estimate(self, measurements):

        if self.probs is None:
            raise ValueError("self.probs must be updated before calling the estimate method.")

        # Separate states (tuples) and values (counts)
        states = torch.tensor(list(measurements.keys()))  # Convert keys to a tensor
        counts = torch.tensor(list(measurements.values()), dtype=torch.int64)  # Convert counts to a tensor

        # Use repeat_interleave to replicate the states according to the counts
        samples = states.repeat_interleave(counts, dim=0)

        # samples = torch.tensor(measurements, dtype=torch.float64)

        if self.padding > 0:
            samples = samples[:, : -self.padding]

        if self.samples_dim is None or self.samples_dim != samples.shape[0]:
            self.tensor_ones = torch.ones_like(samples).to(samples.device)
            self.samples_dim = samples.shape[0]

        samples = torch.minimum(self.tensor_ones, samples)  # apply threshold detection mapping

        self.samples = samples

        n_samples = samples.shape[0] * self.sampling_factor

        flips = torch.bernoulli(self.probs.repeat(n_samples, 1))
        samples = samples.repeat((self.sampling_factor, 1))
        samples = flips * samples + (1 - flips) * (1 - samples)

        samples_unique, samples_counts = np.unique(samples, axis=0, return_counts=True)

        E = sum(
            [
                self.objective_function(sample) * count / n_samples
                for sample, count in zip(samples_unique, samples_counts)
            ]
        )

        return torch.tensor([E], dtype=torch.float32)
