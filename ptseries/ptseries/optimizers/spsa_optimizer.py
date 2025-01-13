from warnings import warn
import numpy as np
from ptseries.optimizers.hybrid_optimizer import HybridOptimizer
from ptseries.models.pt_layer import PTLayer


def HybridSPSA(
    model,
    lr_classical=1e-2,
    lr_quantum=1e-2,
    optimizer_quantum="SGD",
    optimizer_classical="Adam",
    betas=(0.9, 0.999),
    gradient_mode="spsa",
    spsa_resamplings=1,
    spsa_gamma_decay=0.101,
    spsa_alpha_decay=0.602,
    spsa_finite_difference=np.pi / 6,
):
    r"""A hybrid optimizer that defaults to SPSA for quantum parameters and Adam for classical parameters.

    Note:
        Deprecated in 2.6.0: `HybridSPSA` has been removed, it is been replaced by `HybridOptimizer`.
    """
    warn(
        "The 'HybridSPSA' class has been deprecated and removed. Please use 'HybridOptimizer' instead.",
        DeprecationWarning,
        2,
    )

    for module in model.modules():
        if isinstance(module, PTLayer):
            module.gradient_mode = gradient_mode
            module.gradient_delta = spsa_finite_difference

    return HybridOptimizer(
        model,
        lr_classical,
        lr_quantum,
        optimizer_quantum,
        optimizer_classical,
        betas,
        spsa_resamplings,
        spsa_gamma_decay,
        spsa_alpha_decay,
    )
