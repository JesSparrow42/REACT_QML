import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


class Unitary(nn.Module):
    """Computes the unitary and its gradient using PyTorch in an efficient way."""

    def __init__(self, n_modes, tbi):
        super(Unitary, self).__init__()

        self.n_modes = n_modes
        self.tbi = tbi

    def forward(self, thetas):
        return UnitaryFun.apply(thetas, self)

    def extra_repr(self):
        """Prints essential information about this class instance."""
        return "Note: this represents an internal model of the PT Series that is tracked by PTGenerator"


class UnitaryFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, thetas, obj):
        n_modes = obj.n_modes
        tbi = obj.tbi
        device = thetas.device

        n_beam_splitters = tbi.calculate_n_beam_splitters(n_modes)
        n_loops = tbi.n_loops
        loop_lengths = [1] * n_loops if tbi.loop_lengths is None else tbi.loop_lengths

        arange = torch.arange(n_modes)

        c = torch.cos(thetas).unsqueeze(1)
        s = torch.sin(thetas).unsqueeze(1)

        small_unitaries = torch.cat((c - 1, s, -s, c - 1), dim=1)
        small_unitaries_2 = torch.cat((-s, c, -c, -s), dim=1)

        identity = torch.sparse_coo_tensor(
            indices=torch.stack((arange, arange)),
            values=[1.0] * n_modes,
            size=[n_modes, n_modes],
            device=device,
        )

        unitary = torch.clone(identity)

        unitaries = []

        current_idx = n_beam_splitters - 1

        G = torch.zeros(n_beam_splitters, n_modes, n_modes, device=device)

        indices = []

        for delay in reversed(loop_lengths):
            for i in reversed(range(n_modes - delay)):
                inds = torch.tensor([[i, i, i + delay, i + delay], [i, i + delay, i, i + delay]], device=device)
                u_i = torch.sparse_coo_tensor(
                    indices=inds,
                    values=small_unitaries[current_idx, :],
                    size=[n_modes, n_modes],
                    device=device,
                )

                unitaries.append(u_i)

                u_i_2 = torch.sparse_coo_tensor(
                    indices=inds,
                    values=small_unitaries_2[current_idx, :],
                    size=[n_modes, n_modes],
                    device=device,
                )

                G[current_idx, :, :] = torch.sparse.mm(unitary, u_i_2).to_dense()
                unitary = torch.sparse.mm(unitary, u_i + identity)

                indices.append(current_idx)
                current_idx -= 1

        unitary_back = torch.eye(n_modes, device=device)

        for p in indices[::-1]:
            G[p, :, :] = torch.mm(G[p, :, :], unitary_back)

            unitary_back = torch.mm((identity + unitaries[len(indices) - p - 1]).to_dense(), unitary_back)

        G = G[np.arange(n_beam_splitters), :, :]

        ctx.save_for_backward(G)

        return unitary.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        (G,) = ctx.saved_tensors

        derivatives_tensor = torch.tensordot(G, grad_output, dims=([1, 2], [0, 1]))

        return derivatives_tensor, None
