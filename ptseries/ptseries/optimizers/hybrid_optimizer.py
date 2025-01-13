import numpy as np
import torch

from ptseries.models.pt_layer import PTLayer


class HybridOptimizer:
    r"""A hybrid optimizer that defaults to SGD for quantum parameters and Adam for classical parameters.

    Unlike standard PyTorch optimizers, HybridOptimizer takes as argument a model instead of parameters.
    HybridOptimizer can be used to implement the popular SPSA optimizer for quantum processors.
    Please see https://arxiv.org/pdf/1704.05018v2.pdf#section*.11 for an explanation of the default SPSA arguments

    Args:
        model (nn.Module): model to be optimized.
        lr_classical (float, optional): learning rate of the classical parameters (default: 1e-2)
        lr_quantum (float, optional): learning rate of the classical parameters (default: 1e-2)
        optimizer_quantum (str, optional): 'SGD' or 'ADAM'. We recommend using 'SGD' for spsa.
        optimizer_classical (str, optional): 'SGD' or 'ADAM'
        betas (Tuple[float, float], optional): (beta1, beta2) of Adam.
        spsa_resamplings (int, optional): number of times the SPSA perturbations are resampled. Default is 1.
        spsa_gamma_decay (float, optional): exponent used for the perturbation strength decay in SPSA. Default is 0.101.
        spsa_alpha_decay (float, optional): exponent used for the learning rate decay in SPSA. Default is 0.602.
    """

    def __init__(
        self,
        model,
        lr_classical=1e-2,
        lr_quantum=1e-2,
        optimizer_quantum="SGD",
        optimizer_classical="Adam",
        betas=(0.9, 0.999),
        spsa_resamplings=1,
        spsa_gamma_decay=0.101,
        spsa_alpha_decay=0.602,
    ):
        self.model = model

        self.lr_classical = lr_classical
        self.lr_quantum = lr_quantum
        self.optimizer_quantum = optimizer_quantum
        self.optimizer_classical = optimizer_classical
        self.betas = betas
        self.spsa_resamplings = spsa_resamplings
        self.spsa_gamma_decay = spsa_gamma_decay
        self.spsa_alpha_decay = spsa_alpha_decay

        self.pt_layer = None
        param_groups_quantum = None
        self.update = 0

        with torch.no_grad():
            param_groups_quantum = []
            param_groups_classical = []

            for p in model.named_parameters():
                if "theta_trainable" in p[0]:
                    param_groups_quantum.append({"params": p[1], "lr": lr_quantum})
                else:
                    param_groups_classical.append({"params": p[1], "lr": lr_classical})

        # Find if there is a PTLayer in the model, and if spsa is used, update the PTLayer attributes
        for module in model.modules():
            if isinstance(module, PTLayer):
                self.pt_layer = module
                self.gradient_mode = self.pt_layer.gradient_mode
                self.gradient_delta = self.pt_layer.gradient_delta

                if self.gradient_mode not in ["spsa", "parameter-shift", "finite-difference"]:
                    raise ValueError(
                        "Gradient mode for HybridOptimizer must be 'spsa', 'parameter-shift' or 'finite-difference"
                    )

                if self.gradient_mode == "spsa":
                    self.pt_layer.spsa_resamplings = spsa_resamplings

        self.classical_optimizer = None
        self.quantum_optimizer = None

        if len(param_groups_classical) > 0:
            if self.optimizer_classical == "Adam":
                self.classical_optimizer = torch.optim.Adam(param_groups_classical, lr=lr_classical, betas=self.betas)
            elif self.optimizer_classical == "SGD":
                self.classical_optimizer = torch.optim.SGD(param_groups_classical, lr=lr_classical)

        if len(param_groups_quantum) > 0:
            if self.optimizer_quantum == "Adam":
                self.quantum_optimizer = torch.optim.Adam(param_groups_quantum, lr=lr_quantum, betas=self.betas)
            elif self.optimizer_quantum == "SGD":
                self.quantum_optimizer = torch.optim.SGD(param_groups_quantum, lr=lr_quantum)

    def step(self):
        """HybridOptimizer implementation of the PyTorch step method."""
        # update classical optimizer
        if self.classical_optimizer:
            self.classical_optimizer.step()

        # update quantum optimizer
        if self.gradient_mode == "spsa":
            current_learning_rate = self.lr_quantum / (self.update + 1) ** self.spsa_alpha_decay
            self._set_lr_quantum(current_learning_rate)

            if self.pt_layer is not None:
                self.pt_layer.gradient_delta = self.gradient_delta / (self.update + 1) ** self.spsa_gamma_decay
                self.update += 1

        if self.quantum_optimizer:
            self.quantum_optimizer.step()

    def zero_grad(self):
        """HybridOptimizer implementation of the PyTorch zero_grad method."""
        if self.classical_optimizer:
            self.classical_optimizer.zero_grad()
        if self.quantum_optimizer:
            self.quantum_optimizer.zero_grad()

    def set_initial_lr(self, lr):
        """Sets the learning rate.

        Args:
            lr (float): learning rate
        """
        self._set_lr_classical(lr)
        self.lr_classical = lr
        self._set_lr_quantum(lr)
        self.lr_quantum = lr

    def _set_lr_classical(self, lr):
        if self.classical_optimizer:
            for param in self.classical_optimizer.param_groups:
                param["lr"] = lr

    def _set_lr_quantum(self, lr):
        if self.quantum_optimizer:
            for param in self.quantum_optimizer.param_groups:
                param["lr"] = lr

    def reset(self):
        """Reset the update attribute."""
        self.update = 0
