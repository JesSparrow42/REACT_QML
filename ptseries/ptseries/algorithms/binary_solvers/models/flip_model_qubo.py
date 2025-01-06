import math
import torch
from torch import nn
import numpy as np


class Parametrisation(nn.Module):
    def __init__(self):
        super(Parametrisation, self).__init__()

    def forward(self, x):
        return (1 + torch.cos(x)) / 2


class FlipModelQubo(nn.Module):
    """A Pytorch model to be utilized by the Binary Bosonic Solver algorithm for QUBO formulated problems.

    Handles the parametrized bit flipping of bit strings.
    """

    def __init__(self, Q: torch.tensor, device: str = "cpu", entropy_penalty: float = 0.0):
        """Initialises the model.

        Args:
            Q (torch.tensor): A matrix which defines a QUBO problem.
            device (str, optional): Defaults to "cpu".
            entropy_penalty (float): Entropy penalty factor that incentivises binary bit-flip probabilities
        """
        super().__init__()

        self.Q = Q
        self.device = device
        self.entropy_penalty = entropy_penalty

        self.alphas = nn.Parameter(0.5 * np.pi * torch.ones((1, Q.shape[0]), requires_grad=True, dtype=torch.float32))

        self.parametrisation = Parametrisation()

        self.m_ij = None

    def get_probs(self):
        return self.parametrisation(self.alphas).detach()

    def forward(self, x):

        m_ij = x[0].reshape(-1, self.Q.shape[0])
        m_i = torch.diag(m_ij)
        self.m_ij = m_ij.detach().clone()

        probs = self.parametrisation(self.alphas)

        term1 = m_ij * torch.matmul(1 - 2 * probs.T, 1 - 2 * probs)
        term2 = m_i[None, :].repeat(m_i.shape[0], 1) * torch.matmul(probs.T, 1 - 2 * probs)
        term3 = m_i[:, None].repeat(1, m_i.shape[0]) * torch.matmul(probs.T, 1 - 2 * probs).T
        term4 = torch.matmul(probs.T, probs)
        diag_offset = torch.diag(probs[0] - probs[0] ** 2)

        term = torch.sum((term1 + term2 + term3 + term4 + diag_offset) * self.Q)

        norm = self.Q.shape[0] * math.log(math.e) / math.e
        E_p = self.entropy_penalty * ((probs + 1e-14) * torch.log(1 / (probs + 1e-14))).sum() / norm

        return term.unsqueeze(0).unsqueeze(1) + E_p

    @torch.no_grad()
    def sample(self, samples):

        n_samples_infer = samples.shape[0]

        probs = self.parametrisation(self.alphas).detach()

        flips = torch.bernoulli(probs.repeat(samples.shape[0], 1))
        samples = flips * samples + (1 - flips) * (1 - samples)

        samples = samples[:, :, None]

        Q_repeat = self.Q[None, :, :].repeat(n_samples_infer, 1, 1)

        e = torch.bmm(torch.bmm(torch.transpose(samples, 2, 1), Q_repeat), samples)

        norm = self.Q.shape[0] * math.log(math.e) / math.e
        E_p = self.entropy_penalty * ((probs + 1e-14) * torch.log(1 / (probs + 1e-14))).sum().item() / norm

        return e.flatten().cpu().numpy(), samples.squeeze(2).cpu().numpy()
