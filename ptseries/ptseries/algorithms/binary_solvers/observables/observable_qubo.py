import torch
import numpy as np

from ptseries.models.observables import Observable


class ObservableQubo(Observable):
    """A custom observable for the PTLayer that is utilised by the Binary Bosonic Solver algorithm.

    Used for QUBO formulated binary optimisation problems.
    This observable receives the samples from the boson sampler and computes the following quantities:
        - m_i = p(n_i=0), i.e, probability to have 0 photons in the mode i.
        - m_ij = p(n_i=0 and n_j=0), i.e, probability to have 0 photons in the mode i and 0 photons in mode j.
    The observable returned is m_ij where m_i is obtained from the diagonal of m_ij.
    """

    descr = "binary-qubo"
    sample_format = "array"

    def __init__(self, padding: int = 0):
        """Initialises the observable.

        Args:
            padding (int, optional): Number of padding modes. Defaults to 0.
        """
        self.padding = padding
        self.samples = None

        self.tensor_ones = None
        self.samples_dim = None

    def estimate(self, measurements):

        # Separate states (tuples) and values (counts)
        states = torch.tensor(list(measurements.keys()))  # Convert keys to a tensor
        counts = torch.tensor(list(measurements.values()), dtype=torch.int64)  # Convert counts to a tensor

        # Use repeat_interleave to replicate the states according to the counts
        samples = states.repeat_interleave(counts, dim=0)

        if self.padding > 0:
            samples = samples[:, : -self.padding]

        if self.samples_dim is None or self.samples_dim != samples.shape[0]:
            self.tensor_ones = torch.ones_like(samples).to(samples.device)
            self.samples_dim = samples.shape[0]

        samples = torch.minimum(self.tensor_ones, samples)  # apply threshold detection mapping

        self.samples = samples  # to access the samples

        samples_inv = 1 - samples  # invert samples (n_samples x n_modes)

        m_ij = torch.matmul(samples_inv.T, samples_inv) / samples_inv.shape[0]  # (n_modes x n_modes)

        return m_ij.flatten()
