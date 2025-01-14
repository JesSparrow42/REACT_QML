import torch
import numpy as np
from typing import Callable

from ptseries.common.logger import Logger
from ptseries.optimizers.hybrid_optimizer import HybridOptimizer
from ptseries.models.pt_layer import PTLayer
from ptseries.algorithms.binary_solvers.observables.observable_qubo import ObservableQubo
from ptseries.algorithms.binary_solvers.observables.observable_non_qubo import ObservableNonQubo
from ptseries.algorithms.binary_solvers.models.flip_model_qubo import FlipModelQubo
from ptseries.algorithms.binary_solvers.models.flip_model_non_qubo import FlipModelNonQubo


class BinaryBosonicSolver:
    """A class used to solve a binary optimization problem on a PT Series.

    This class supports both QUBO and non-QUBO formulations of the problem.
    This algorithm is an improved version of ORCA's previous binary optimization algorithm
    (https://arxiv.org/abs/2112.09766).
    """

    def __init__(
        self,
        pb_dim: int,
        objective: np.ndarray | Callable,
        input_state: list = None,
        tbi_params: dict = None,
        n_samples: int = 100,
        gradient_mode: str = "parameter-shift",
        gradient_delta: float = np.pi / 6,
        spsa_params: dict = None,
        device: str = "cpu",
        sampling_factor: int = 2,
        entropy_penalty: float = 0.1,
    ):
        """Initialises the Binary Bosonic Solver.

        Args:
            pb_dim (int): Dimension of the binary problem i.e. number of binary variables.
            objective (numpy array or function): matrix specifying the QUBO problem or function (in the case of non qubo formulation).
            input_state (list, optional): input state used by the boson sampler. If 'None', default to repeated 1,0 input state.
            tbi_params (dict, optional): dict of parameters used to instantiate a TBI.
            n_samples (int, optional): number of samples used to estimate expectation values.
            gradient_mode (str, optional): 'spsa' or 'parameter-shift' or 'finite-difference'. Type of method used to compute the gradient of the quantum parameters.
            gradient_delta (float, optional): delta to use with the parameter shift rule or for the finite difference.
            spsa_params (dict, optional): parameters of the spsa method.
            device (string, optional): default 'cpu'
            sampling_factor (int): Classical sample factor required for non-QUBO formulation only.
            entropy_penalty (float): Entropy penalty factor that incentivises
        """
        self.pb_dim = pb_dim
        self.objective = objective
        self.tbi_params = tbi_params
        self.n_samples = n_samples
        self.gradient_mode = gradient_mode
        self.gradient_delta = gradient_delta
        self.spsa_params = spsa_params
        self.device = device

        if input_state is None:
            input_state = [1, 0] * pb_dim
            input_state = input_state[:pb_dim]

        self.input_state = input_state
        self.entropy_penalty = entropy_penalty

        self.sampling_factor = sampling_factor

        self.qubo_formulation = False if callable(self.objective) else True

        if not self.qubo_formulation:
            # Tracks the lowest cost function value calculated, and stores the input bit string that achieved this

            self.min_encountered_cost_fn = float("inf")
            self.best_encountered_soln = None

            original_callable = self.objective

            def tracked_objective(bit_str):

                cost_fn_val = original_callable(bit_str)

                if cost_fn_val < self.min_encountered_cost_fn:
                    self.min_encountered_cost_fn = cost_fn_val
                    self.best_encountered_soln = bit_str

                return cost_fn_val

            self.objective = tracked_objective

        self.Q = np.array(self.objective, dtype=np.float64) if self.qubo_formulation else None

        self.n_tiling = 1
        self.padding = 0
        if self.pb_dim > len(input_state):
            self.n_tiling = self.pb_dim // len(input_state)

            if self.pb_dim % len(input_state) != 0:
                self.n_tiling += 1
                self.padding = len(input_state) - self.pb_dim % len(input_state)

            print(
                f"pb_dim larger than input_state length : will use a tiling of {self.n_tiling}{' and a padding of ' + str(self.padding) + '.' if self.padding>0 else '.'}"
            )

        self.res = {}

        self.E_min_encountered = np.inf
        self.E_min_encountered_list = []
        self.config_min_encountered = None
        self.logger = None

        self.logger = Logger(log_dir=None)

        if self.logger is not None:
            metadata = {
                "problem_dim": self.pb_dim,
                "n_samples": self.n_samples,
                "tbi_params": self.tbi_params,
                "gradient_delta": self.gradient_delta,
                "input_state": list(input_state),
                "n_tiling": self.n_tiling,
            }
            self.logger.register_metadata(metadata)

        # Flip model
        if self.qubo_formulation:
            self.flip_model = FlipModelQubo(
                Q=torch.tensor(self.Q, device=self.device),
                device=self.device,
                entropy_penalty=self.entropy_penalty,
            )
        else:
            self.flip_model = FlipModelNonQubo(
                dim=self.pb_dim,
                objective_function=self.objective,
                sampling_factor=sampling_factor,
                entropy_penalty=self.entropy_penalty,
            )

        self.flip_model_optimizer = torch.optim.SGD(self.flip_model.parameters(), lr=3e-2)

        # PTLayer
        if self.qubo_formulation:
            self.observable = ObservableQubo(padding=self.padding)

        else:
            self.observable = ObservableNonQubo(self.objective, sampling_factor=sampling_factor, padding=self.padding)
            self.observable.probs = self.flip_model.get_probs()

        self.pt_layer = PTLayer(
            input_state,
            in_features=0,
            observable=self.observable,
            gradient_mode=gradient_mode,
            gradient_delta=gradient_delta,
            n_samples=n_samples,
            tbi_params=tbi_params,
            n_tiling=self.n_tiling,
        ).to(self.device)

        self.n_beamsplitters = self.pt_layer.theta_trainable.shape[1]

        self.hybrid_optimizer = HybridOptimizer(
            self.pt_layer,
            **(spsa_params or {}),
        )

    def train(
        self,
        learning_rate: float = 5e-2,
        learning_rate_flip: float = 1e-1,
        updates: int = 100,
        logger: Logger = None,
        verbose: bool = True,
        print_frequency: int = 50,
    ) -> None:
        """Training of the QUBO algorithm.

        Args:
            learning_rate (float): learning rate of the beam splitter angles. Defaults to 5e-2.
            learning_rate_flip (float): learning rate of the bit flip probabilities. Defaults to 1e-1.
            updates (int): number of training iterations. Defaults to 100.
            logger (Logger, optional): object that handles and saves all the logs during the training. Defaults to None.
            verbose: whether to print loss information during training. Default is True.
            print_frequency (int): frequency at which we print to terminal. Defaults to 50.
        """
        self.logger = logger
        self.hybrid_optimizer.set_initial_lr(learning_rate)

        self.e_samples = None

        for param in self.flip_model_optimizer.param_groups:
            param["lr"] = learning_rate_flip

        for update in range(updates):
            e_avg_current = self.forward()

            self.save_best_results(update, e_avg_current.item())  # saved the best results encountered

            if (((update + 1) % print_frequency == 0) | (update == 0)) and verbose:
                print("Training loop {}: loss is {:.2f}".format(update + 1, e_avg_current.item()))

            self.hybrid_optimizer.zero_grad()
            self.flip_model_optimizer.zero_grad()

            e_avg_current.backward()

            self.hybrid_optimizer.step()

            self.flip_model_optimizer.step()

            self.E_min_encountered_list.append(self.E_min_encountered)

        if self.logger is not None:
            if self.logger.log_dir is not None:
                self.logger.save()

        if not self.qubo_formulation:
            # Saves best solution encountered by the tracked cost function
            self.config_min_encountered = self.best_encountered_soln

    def forward(self):
        if self.qubo_formulation:
            pt_layer_output = self.pt_layer()
            flip_output = self.flip_model(pt_layer_output)

            return flip_output

        else:
            self.pt_layer.observable.probs = self.flip_model.get_probs()

            E1 = self.pt_layer()
            samples = self.get_samples()
            E2 = self.flip_model(samples)

            return (
                (E1 + E2) / 2
            ).flatten()  # E1 leads to the gradients of PTLayer, E2 leads to the gradients of the flipping probs.

    def get_samples(self):
        return self.pt_layer.observable.samples.detach().to(torch.float64)

    @torch.no_grad()
    def save_best_results(self, update, current_loss):
        samples = self.get_samples()

        e_list, bit_strings = self.flip_model.sample(samples)

        if np.min(e_list) < self.E_min_encountered:
            self.E_min_encountered = np.min(e_list)
            self.config_min_encountered = bit_strings[np.argmin(e_list), :]

        if self.logger is not None:
            self.logger.log("energy_avg", update, np.mean(e_list))
            self.logger.log("energy_min", update, np.min(e_list))  # min energy found at the current iteration
            self.logger.log("energy_max", update, np.max(e_list))
            self.logger.log(
                "energy_min_all", update, self.E_min_encountered
            )  # min energy found since the begining of the training

            self.logger.log("loss", update, current_loss)

    @torch.no_grad()
    def sample(self, n_samples_infer: int):
        """Runs the PT Series to get samples and flip them using the optimised beam splitter angles and flip probabilities.

        Returns a list of bit strings with their associated energies.
        This function can be ran after training.

        Args:
            n_samples_infer (int): number of samples generated.
        """
        self.pt_layer.n_samples = n_samples_infer
        self.pt_layer()
        self.pt_layer.n_samples = self.n_samples

        samples = self.pt_layer.observable.samples.to(torch.float64)

        e_list, bit_string = self.flip_model.sample(samples)
        return bit_string, e_list
