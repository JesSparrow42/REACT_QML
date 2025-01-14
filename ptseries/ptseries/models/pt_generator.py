import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from ptseries.models.pt_generator_training.observables_torch import AveragePhotons
from ptseries.models.pt_generator_training.unitary_torch import Unitary
from ptseries.models.utils import calculate_n_params
from ptseries.tbi import create_tbi


class PTGenerator(nn.Module):
    r"""A PyTorch module extension that uses the PT Series to generate probability distributions.

    Key differences with PTLayer are:

    - PTGenerator uses the `generate` method instead of a `forward` method. `generate` returns the specified number of
      samples from the PT Series' probability distribution
    - The PTGenerator distribution is trained by optimising the average photon number of each mode

    Args:
        input_state (list, tuple or array): input state for the TBI, for example :math:`(1,1,0)` for :math:`|1,1,0\rangle`.
        tbi_params (dict, optional): dict of optional parameters to instantiate the TBI.
        n_tiling (int, optional): uses n_tiling instances of PT Series and concatenates the results. Each tile uses
            different trainable parameters.
    """

    def __init__(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        tbi_params: dict | None = None,
        n_tiling: int = 1,
    ):
        super(PTGenerator, self).__init__()

        self.input_state = np.asarray(input_state)
        if tbi_params is None:
            tbi_params = {}
        self.tbi_params = tbi_params
        self.n_tiling = n_tiling
        self.order = 1

        self.n_modes = len(input_state)
        self.q_list = torch.where(torch.tensor(input_state) > 0)[0]

        self.tbi = create_tbi(**tbi_params)

        self.n_theta = calculate_n_params(self.input_state, tbi_params, n_tiling)
        self._set_thetas()

        self.input_loss = tbi_params.get("input_loss", 0)
        self.detector_efficiency = tbi_params.get("detector_efficiency", 1)
        self.input_loss_commuted = 1 - (1 - self.input_loss) * self.detector_efficiency

        # Create an observable for a classically simulable quantity that will be used for training
        self.observables = [AveragePhotons]
        self.unitary_model = Unitary(n_modes=self.n_modes, tbi=self.tbi)
        self._init_observable_models()

    def _set_thetas(self):
        theta_trainable = torch.rand(self.n_theta, requires_grad=True)
        self.theta_trainable = torch.nn.Parameter((theta_trainable - 0.5) * 2 * torch.pi)

    @torch.no_grad()
    def _init_observable_models(self):
        self.observable_models = [
            self.observables[o](self.q_list, input_loss=self.input_loss_commuted) for o in range(self.order)
        ]

    @torch.no_grad()
    def generate(self, batch_size):
        return self.forward(batch_size)

    @torch.no_grad()
    def forward(self, batch_size: int | torch.Tensor):
        """
        Args:
            batch_size (int): batch size to generate.
        """
        if isinstance(batch_size, torch.Tensor):
            batch_size = batch_size.item()

        samples_array = self.tbi.sample(
            self.input_state,
            self.theta_trainable.detach().cpu().numpy(),
            n_samples=batch_size,
            output_format="list",
            n_tiling=self.n_tiling,
        )
        return torch.tensor(np.array(samples_array), dtype=torch.float32, device=self.theta_trainable.device)

    def backward(self, samples, losses):
        """Compute the gradients of the averaged losses with respect to the PT Series trainable parameters (approximation).

        Args:
            samples (torch tensor).
            losses (list): list of the losses over a batch
        """

        self._check_tbi_compatibility()

        grad = self._get_gradient(samples, losses)
        self.theta_trainable.grad = grad

    def _check_tbi_compatibility(self):
        """Returns an exception if the backward method is called with an incompatible tbi"""
        if self.tbi_params is not None:
            if self.tbi_params.get("tbi_type", None) == "PT-1":
                raise Exception("PTGenerator does not yet support PT-1 training.")

            if not np.isclose(self.tbi_params.get("bs_loss", 0), 0):
                raise Exception("PTGenerator does not support non-zero bs_loss during training")

            if not np.isclose(self.tbi_params.get("bs_noise", 0), 0):
                raise Exception("PTGenerator does not support non-zero bs_noise during training")

    def _get_gradient(self, samples, losses):
        device = self.theta_trainable.device

        derivatives_tensor = torch.zeros_like(self.theta_trainable)

        self.u = [self.unitary_model(thetas) for thetas in self.theta_trainable.reshape(self.n_tiling, -1)]

        gradient_loss_wrt_obs = self._get_gradient_loss_wrt_obs(samples.detach(), losses.detach())

        if self.order == 1:
            gradient_loss_wrt_obs = torch.split(gradient_loss_wrt_obs, [self.n_modes * self.n_tiling])

        for order in range(self.order):
            self.observable_models[order].zero_grad()

            observable = torch.empty(0).to(device)
            for l in range(self.n_tiling):
                observable_l = self.observable_models[order](self.u[l])
                observable = torch.cat((observable, observable_l))

            grad_new = torch.autograd.grad(  # returns the jacobian x gradient_loss_wrt_obs
                outputs=observable,
                inputs=self.theta_trainable,
                grad_outputs=gradient_loss_wrt_obs[order].to(device),
                retain_graph=True,
            )[0]

            derivatives_tensor = derivatives_tensor + grad_new

        return derivatives_tensor

    def _get_gradient_loss_wrt_obs(self, latent, losses):
        order = self.order
        fit_method = "OLS"

        # fitting method
        if fit_method == "OLS":
            from ptseries.models.pt_generator_training.fit.ordinary_least_squares import fit
        else:
            raise NotImplementedError("Wrong fit_method")

        if order == 1:
            X = torch.clone(latent)

        f = fit(X, losses)
        self.last_fit = f

        return f

    def extra_repr(self):
        """Prints essential information about this class instance"""
        return "trainable_theta={}".format(self.n_theta)

    def print_info(self):
        """Prints all relevant information for this class instance"""

        if self.tbi_params is None:
            tbi_params = {}
        else:
            tbi_params = self.tbi_params

        HARDWARE_DEFINED = "Defined by hardware"
        if self.tbi.descr == "PT-1":
            tbi_params["bs_loss"] = HARDWARE_DEFINED
            tbi_params["bs_noise"] = HARDWARE_DEFINED
            tbi_params["input_loss"] = HARDWARE_DEFINED
            tbi_params["detector_efficiency"] = HARDWARE_DEFINED
            tbi_params["distinguishable"] = HARDWARE_DEFINED
            tbi_params["n_signal_detectors"] = HARDWARE_DEFINED
            tbi_params["g2"] = HARDWARE_DEFINED

        if hasattr(self.tbi, "loop_lengths"):
            if self.tbi.loop_lengths is None:
                loop_lengths = self.tbi.n_loops * [1]
            else:
                loop_lengths = self.tbi.loop_lengths
        else:
            loop_lengths = self.tbi.n_loops * [1]

        print("PTGenerator              |  ")
        print("------------------------------------------------")
        print("trainable_theta          |  " + str(self.n_theta))
        print("input_state              |  " + str(self.input_state))
        print("n_tiling                 |  " + str(self.n_tiling))
        print("n_loops                  |  " + str(self.tbi.n_loops))
        print("loop_lengths             |  " + str(loop_lengths))
        print("tbi_type                 |  " + self.tbi.descr)
        print("bs_loss                  |  " + str(tbi_params.get("bs_loss", 0)))
        print("bs_noise                 |  " + str(tbi_params.get("bs_noise", 0)))
        print("input_loss               |  " + str(tbi_params.get("input_loss", 0)))
        print("detector_efficiency      |  " + str(tbi_params.get("detector_efficiency", 1)))
        print("distinguishable          |  " + str(tbi_params.get("distinguishable", False)))
        print("n_signal_detectors       |  " + str(tbi_params.get("n_signal_detectors", 0)))
        print("g2                       |  " + str(tbi_params.get("g2", 0)))
