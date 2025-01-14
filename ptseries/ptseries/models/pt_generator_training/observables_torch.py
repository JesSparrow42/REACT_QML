import torch
import torch.nn as nn


class AveragePhotons(nn.Module):
    def __init__(self, q_list, input_loss=0):
        super(AveragePhotons, self).__init__()

        self.q_list = q_list
        self.input_loss = input_loss

    def forward(self, unitary):
        """Given a unitary tensor, computes the <ni> as a PyTorch tensor of shape (n_modes)"""

        U_reindexed = unitary[:, self.q_list]
        U_reindexed_2 = torch.abs(U_reindexed) ** 2

        return torch.sum(U_reindexed_2, dim=1) * (1 - self.input_loss)
