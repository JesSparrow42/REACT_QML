from typing import Callable
from collections import Counter
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from ptseries.models.observables import AvgPhotons, Correlations, Covariances, Observable, SingleSample
from ptseries.models.utils import calculate_n_params
from ptseries.tbi.tbi import create_tbi
from ptseries.tbi.tbi_abstract import TBISimulator, TBIDevice


class PTLayer(nn.Module):
    r"""A PyTorch module extension that describes the forward pass and backpropagation on a PT Series.

    Args:
        input_state: Input state for the TBI, for example :math:`(1,1,0)` for :math:`|1,1,0\rangle`.
        in_features: Input dimension of the layer. If smaller than the total number of beam splitters, the remaining
            beam splitters will be trainable parameters. Default is "default". Options include:

            - "default": No internal trained parameters, all the beam splitters encode input data
            - "half": Half of the parameters are trained. For a 2 loop layer, the first loop encodes the input, the
              second is trained.
        observable: Class that converts measurements to a tensor. Default is "avg-photons". Options include:

            - "correlations": The observable is the set of two-point correlators
            - "covariances": The observable is the set of two-point covariances
            - "avg-photons": The observable is the set of single-mode average photon numbers
            - "single-sample": One forward pass consists of a single sample from the TBI.
        n_samples: Number of samples to draw. Default is 100.
        tbi_params: Dictionary of optional parameters to instantiate the TBI. Default is None.
        gradient_mode: Method to compute the gradient. Default is "parameter-shift". Options include:

            - "parameter-shift": Parameter shift rule method
            - "finite-difference": Finite difference method
            - "spsa": Simultaneous perturbation stochastic approximation. Because this method also requires to adapt the
              learning rate and the delta as we iterate, it must be used with the ORCA optimiser HybridOptimizer.
        gradient_delta: Delta to use with the parameter shift rule or for the finite difference. Default is
            :math:`\pi/10`.
        n_tiling: Uses n_tiling instances of PT Series and concatenates the results. Input features are distributed
            between tiles, which each have different trainable params. Default is 1.
    """

    def __init__(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        in_features: str | int = "default",
        observable: str | Observable = "avg-photons",
        gradient_mode: str = "parameter-shift",
        gradient_delta: float = np.pi / 10,
        n_samples: int = 100,
        tbi_params: dict | None = None,
        n_tiling: int = 1,
    ):
        super(PTLayer, self).__init__()

        self.input_state = input_state
        self.in_features = in_features
        self.n_samples = n_samples
        self.tbi_params = tbi_params
        self.gradient_delta = gradient_delta
        self.n_tiling = n_tiling
        self.gradient_mode = gradient_mode

        # parameters for spsa, handled by the Hybrid optimizer
        self.spsa_resamplings = None

        # Handle observable entered as a string
        if isinstance(observable, str):
            observable_types = {
                "correlations": Correlations,
                "covariances": Covariances,
                "avg-photons": AvgPhotons,
                "single-sample": SingleSample,
            }
            if observable not in observable_types.keys():
                raise ValueError("Unknown observable.")
            self.observable_arg = observable  # for the __repr__
            self.observable = observable_types[observable]()

        # Handle observable entered as instance of an Observable class
        else:
            if not isinstance(observable, Observable):
                raise ValueError("Observable must be a string or an instance of the Observable class")
            self.observable_arg = observable.descr
            self.observable = observable

        self.tbi = create_tbi(**tbi_params) if tbi_params is not None else create_tbi()

        # check that there is no error when running the observable
        measurement_test = {(1,) * len(self.input_state) * n_tiling: 1.0}

        try:
            self.dim_observable = len(self.observable(measurement_test))
        except:
            raise Exception("Error in the observable function")

        # Add thetas that will be trained independently
        n_params = calculate_n_params(input_state, tbi_params, self.n_tiling)

        if self.in_features != "default":
            if self.in_features == "half":
                self.in_features = n_params // 2

            elif self.in_features == n_params:
                self.in_features = "default"

            # initialize free parameters to random values between -pi and pi
            elif 0 <= self.in_features < n_params:
                theta_values = torch.rand(1, n_params - self.in_features)
                theta_values = (theta_values - 0.5) * 2 * torch.pi
                self.set_thetas(theta_values)

            elif self.in_features > n_params:
                raise ValueError("PTLayer: in_features cannot be strictly greater than the number of beam splitters.")

            else:
                raise ValueError(
                    f"Invalid value. Valid arguments include 'default', 'half' or an integer between 0 and {n_params}."
                )

    def set_thetas(self, theta_values: torch.Tensor):
        """Set the beam splitter values.

        Args:
            theta_values: values of the beam splitter.
        """
        self.theta_trainable = nn.Parameter(theta_values, requires_grad=True)

    def forward(self, x: torch.Tensor | None = None, n_samples: int | None = None):
        """Defines inference performed with this model.

        Args:
            x: input PyTorch tensor. If in_features=0, then x can be None
                but can also be specified to return a batch of x.shape(0) datapoints.
            n_samples: number of samples to draw. If None, this instance's n_samples attribute is used

        Returns:
            the tensor containing the inference result
        """
        if x is None:
            if self.in_features == 0:
                x = torch.zeros(1)  # A dummy tensor
            else:
                raise ValueError("You must specify an input tensor for PTLayer inference")

        if n_samples is None:
            n_samples = self.n_samples

        parameters = [
            n_samples,
            self.input_state,
            self.gradient_delta,
            self.gradient_mode,
            self.spsa_resamplings,
            self.n_tiling,
            self.dim_observable,
            self.training,
        ]

        if self.in_features != "default":
            if self.in_features == 0:  # all the parameters of the tbi are trainable
                theta_trainable = self.theta_trainable.repeat(x.size(0), 1)
                return TBIAutograd.apply(theta_trainable, self.observable, self.tbi, parameters)

            elif x.size(1) != self.in_features:
                msg = "PTLayer error: in_features = {} but input tensor has size {}".format(self.in_features, x.size(1))
                raise ValueError(msg)

            else:
                theta_trainable = self.theta_trainable.repeat(x.size(0), 1)
                # split in_features equally (if possible) between the tiles
                x_subs = torch.tensor_split(x, self.n_tiling, 1)

                n_params_per_tile = (
                    calculate_n_params(self.input_state, self.tbi_params, self.n_tiling) // self.n_tiling
                )

                # split theta_trainable so that each tile has the same total number of parameters
                split_indices = torch.cumsum(
                    torch.tensor([n_params_per_tile - x_subs[i].size(1) for i in range(len(x_subs) - 1)]), 0
                ).long()

                theta_trainable_subs = torch.tensor_split(theta_trainable, split_indices, 1)

                # concatenate each ith subtensor of x with ith subtensor of theta_trainable
                tile_subs = []
                for i in range(self.n_tiling):
                    tile_subs.append(torch.cat((x_subs[i], theta_trainable_subs[i]), 1))

                # concatenate each of these new subtensors
                x = torch.cat(tile_subs, 1)

                return TBIAutograd.apply(x, self.observable, self.tbi, parameters)
        else:
            return TBIAutograd.apply(x, self.observable, self.tbi, parameters)

    def extra_repr(self):
        """Prints essential information about this class instance."""
        n_params = calculate_n_params(self.input_state, tbi_params=self.tbi_params, n_tiling=self.n_tiling)

        if self.in_features != "default" and 0 <= self.in_features < n_params:
            n_theta_trainable = n_params - self.in_features
            in_features = self.in_features
        else:
            n_theta_trainable = 0
            in_features = n_params

        return "in_features={}, output_observables={}, trainable_theta={}".format(
            in_features, self.dim_observable, n_theta_trainable
        )

    def print_info(self):
        """Prints all relevant information for this class instance."""
        n_params = calculate_n_params(self.input_state, tbi_params=self.tbi_params, n_tiling=self.n_tiling)
        try:
            n_theta_trainable = self.theta_trainable.shape[1]
        except AttributeError:
            n_theta_trainable = 0

        if self.in_features == "default":
            in_features = str(n_params) + " (default)"
        elif self.in_features == int(n_params / 2):
            in_features = str(int(n_params / 2)) + " (half)"
        else:
            in_features = str(self.in_features)

        if self.observable_arg in ["correlations", "covariances", "avg-photons", "single-sample"]:
            observable = self.observable_arg
        else:
            observable = "custom"

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

        print("PTLayer                  |  ")
        print("------------------------------------------------")
        print("in_features              |  " + in_features)
        print("output_observables       |  " + str(self.dim_observable))
        print("trainable_theta          |  " + str(n_theta_trainable))
        print("input_state              |  " + str(self.input_state))
        print("n_loops                  |  " + str(self.tbi.n_loops))
        print("loop_lengths             |  " + str(loop_lengths))
        print("n_tiling                 |  " + str(self.n_tiling))
        print("total number of angles   |  " + str(n_params))
        print("n_samples                |  " + str(self.n_samples))
        print("observable               |  " + observable)
        print("tbi_type                 |  " + self.tbi.descr)
        print("bs_loss                  |  " + str(tbi_params.get("bs_loss", "0")))
        print("bs_noise                 |  " + str(tbi_params.get("bs_noise", "0")))
        print("input_loss               |  " + str(tbi_params.get("input_loss", "0")))
        print("detector_efficiency      |  " + str(tbi_params.get("detector_efficiency", "1")))
        print("distinguishable          |  " + str(tbi_params.get("distinguishable", False)))
        print("n_signal_detectors       |  " + str(tbi_params.get("n_signal_detectors", 0)))
        print("g2                       |  " + str(tbi_params.get("g2", 0)))
        print("gradient_mode            |  " + str(self.gradient_mode))
        print("gradient_delta           |  " + str(self.gradient_delta))

    def gradient_info(self):
        """Prints information about the gradient computation."""
        print(
            "|------------------------------------------------------------------------------------------------------------------------------------------|"
        )
        print(
            "| gradient_mode       | PT Layer gradient arguments | requires custom optimizer (HyrbidSPSA) |            Gradient computation             |"
        )
        print(
            "|------------------------------------------------------------------------------------------------------------------------------------------| "
        )
        print(
            '| "parameter-shift"   |      "gradient_delta"       |                  no                    |   [f(x+shift) - f(x-shift)] / sin(2 shift)  |'
        )
        print(
            '| "finite-difference" |      "gradient_delta"       |                  no                    |   [f(x+shift) - f(x)] / shift               |'
        )
        print(
            '| "spsa"              |      "gradient_delta"       |                  yes                   |   [f(x+c shift) - f(x-c shift)] / (2 shift) |'
        )
        print(
            "|                     |                             |                                        |     where c and the learning rate           |"
        )
        print(
            "|                     |                             |                                        |     decrease  as we iterate.                |"
        )
        print(
            "|------------------------------------------------------------------------------------------------------------------------------------------| "
        )


class TBIAutograd(torch.autograd.Function):
    """An extension of PyTorch autograd that adapts the PyTorch backpropagation algorithm to the PT Series."""

    @staticmethod
    def forward(
        ctx, theta_angles: torch.Tensor, observable_function: Callable, tbi: TBISimulator | TBIDevice, parameters: list
    ):
        r"""Defines the forward pass.

        In training mode, we evaluate :math:`f(\theta + \Delta)` and :math:`f(\theta - \Delta)` and approximate
        :math:`f(\theta)` by :math:`(f(\theta + \Delta) + f(\theta - \Delta)) / 2`.
        In eval mode, we only calculate :math:`f(\theta )`.

        Args:
            ctx: The context object for autograd.
            theta_angles: tensor of angles with shape (batch_size, n_angles)
            observable_function: the observable to use
            tbi: an instance of a time bin interferometer
            parameters: parameters passed to the TBI

        Returns:
            a tensor containing the expectation values of the observables, with shape (n_batch, n_observables)
        """
        n_samples = parameters[0]
        input_state = parameters[1]
        gradient_delta = parameters[2]
        gradient_mode = parameters[3]
        spsa_resamplings = parameters[4]
        n_tiling = parameters[5]
        dim_observable = parameters[6]
        training = parameters[7]

        # Send the variables for the backward to context object (ctx)
        ctx.observable_function = observable_function
        ctx.tbi = tbi
        ctx.parameters = parameters

        batch_size = theta_angles.shape[0]
        n_thetas = theta_angles.shape[1]
        tile_size = len(input_state)

        if training:
            if gradient_mode == "spsa":
                upshifted_expectation = torch.zeros(spsa_resamplings, batch_size, dim_observable)
                downshifted_expectation = torch.zeros(spsa_resamplings, batch_size, dim_observable)

                noise_vectors = 2 * torch.bernoulli(torch.rand(spsa_resamplings, n_thetas)) - 1

                for i in range(batch_size):
                    thetas = theta_angles[i, :]

                    for idx in range(spsa_resamplings):
                        upshifted_theta = thetas + gradient_delta * noise_vectors[idx, :]

                        state_counts = tbi.sample(
                            input_state,
                            theta_list=upshifted_theta.detach().numpy(),
                            n_samples=n_samples,
                            n_tiling=n_tiling,
                        )
                        upshifted_observable = observable_function(state_counts)
                        upshifted_expectation[idx, i, :] = upshifted_observable

                        downshifted_theta = thetas - gradient_delta * noise_vectors[idx, :]

                        state_counts = tbi.sample(
                            input_state,
                            theta_list=downshifted_theta.detach().numpy(),
                            n_samples=n_samples,
                            n_tiling=n_tiling,
                        )
                        downshifted_observable = observable_function(state_counts)
                        downshifted_expectation[idx, i, :] = downshifted_observable

                    observable = (upshifted_expectation + downshifted_expectation) / 2
                    observable = torch.mean(observable, dim=0)

            elif gradient_mode == "parameter-shift" or gradient_mode == "finite-difference":
                noise_vectors = None

                params_per_tile = int(n_thetas / n_tiling)
                output_format = "array" if n_tiling > 1 else "dict"

                upshifted_expectation = torch.zeros(batch_size, n_thetas, dim_observable)
                downshifted_expectation = torch.zeros(batch_size, n_thetas, dim_observable)
                for i in range(batch_size):
                    if n_tiling > 1:  # Avoid resampling with same values later
                        samples_array = tbi.sample(
                            input_state,
                            theta_list=theta_angles[i, :].detach().numpy(),
                            n_samples=n_samples,
                            n_tiling=n_tiling,
                            output_format=output_format,
                        )

                    for tile in range(n_tiling):
                        start_idx = params_per_tile * tile
                        stop_idx = start_idx + params_per_tile
                        theta_list = theta_angles[i, start_idx:stop_idx].detach().numpy()

                        for j in range(params_per_tile):
                            theta_list[j] += gradient_delta

                            state_counts = tbi.sample(
                                input_state, theta_list=theta_list, n_samples=n_samples, output_format=output_format
                            )

                            if n_tiling > 1:
                                samples_copy = samples_array.copy()
                                samples_copy[:, tile_size * tile : tile_size * (tile + 1)] = state_counts
                                samples_copy = [tuple(state) for state in samples_copy]
                                state_counts = Counter(samples_copy)

                            upshifted_observable = observable_function(state_counts)
                            upshifted_expectation[i, start_idx + j, :] = upshifted_observable

                            theta_list[j] -= 2 * gradient_delta

                            state_counts = tbi.sample(
                                input_state, theta_list=theta_list, n_samples=n_samples, output_format=output_format
                            )

                            if n_tiling > 1:
                                samples_copy = samples_array.copy()
                                samples_copy[:, tile_size * tile : tile_size * (tile + 1)] = state_counts
                                samples_copy = [tuple(state) for state in samples_copy]
                                state_counts = Counter(samples_copy)

                            downshifted_observable = observable_function(state_counts)
                            downshifted_expectation[i, start_idx + j, :] = downshifted_observable

                            # Return angles to their original value
                            theta_list[j] += gradient_delta

                            observable = (upshifted_expectation + downshifted_expectation) / 2
                            observable = torch.mean(observable, dim=1)

            ctx.save_for_backward(theta_angles, noise_vectors, upshifted_expectation, downshifted_expectation)

        else:
            obs_expectation = torch.zeros(batch_size, dim_observable)
            for i in range(batch_size):
                state_counts = tbi.sample(
                    input_state,
                    theta_list=theta_angles[i, :].detach().numpy(),
                    n_samples=n_samples,
                    n_tiling=n_tiling,
                )

                obs_expectation[i, :] = observable_function(state_counts)

                observable = obs_expectation.clone()

        return observable

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Defines the backward pass.

        Takes the gradient of the observables and returns the gradient of the loss function with respect to the beam
        splitter angles.

        Args:
            ctx: The context object for autograd.
            grad_output: the gradient of the loss function with respect to the observables.
        """
        (theta_angles_tensor, noise_tensor, upshifted_expectation, downshifted_expectation) = ctx.saved_tensors

        parameters = ctx.parameters

        gradient_delta = parameters[2]
        gradient_mode = parameters[3]
        spsa_resamplings = parameters[4]

        if spsa_resamplings is None and gradient_mode == "spsa":
            raise Exception(
                'Setting gradient_mode to "spsa" requires the use of the ORCA optimizer to properly update the learning rate.'
            )

        if gradient_mode == "spsa":
            spsa_resamplings = upshifted_expectation.shape[0]
            n_thetas = theta_angles_tensor.shape[1]

            noise_tensor_expended = noise_tensor[:, None, None, :]
            diff = torch.mean(
                (upshifted_expectation - downshifted_expectation)[:, :, :, None].repeat(1, 1, 1, n_thetas)
                / (2 * gradient_delta * noise_tensor_expended),
                dim=0,
            )
            grad_output = grad_output[:, None, :]
            derivatives_tensor = torch.bmm(grad_output, diff).squeeze(1)
            return derivatives_tensor, None, None, None

        elif gradient_mode == "parameter-shift" or gradient_mode == "finite-difference":
            denom = np.sin(2 * gradient_delta) if gradient_mode == "parameter-shift" else 2 * gradient_delta

            diff = (upshifted_expectation - downshifted_expectation) / denom
            diff = torch.transpose(diff, 2, 1)
            grad_output = grad_output[:, None, :]
            derivatives_tensor = torch.bmm(grad_output, diff).squeeze(1)
            return derivatives_tensor, None, None, None

        else:
            raise ValueError('gradient_mode must be either "parameter-shift", "finite-difference" or "spsa"')
