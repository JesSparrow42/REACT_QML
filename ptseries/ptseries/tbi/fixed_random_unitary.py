import warnings

import numpy as np
import numpy.typing as npt

from ptseries.tbi.boson_sampler.boson_sampler import clifford_sample
from ptseries.tbi.boson_sampler.distinguishable_sampler import distinguishable_sample
from ptseries.tbi.tbi_abstract import TBISimulator


class FixedRandomUnitary(TBISimulator):
    """A class to simulate a boson sampler described by a Haar-random unitary matrix.

    Args:
        input_loss: The optical losses at the input of the PT Series, between 0 and 1. Defaults to 0.0.
        detector_efficiency: The detector efficiency of the detectors, between 0 and 1. Defaults to 1.0.
        distinguishable: Whether the photons interfere with each other or not. Defaults to False.
    """

    descr = "fixed-random-unitary"

    def __init__(self, input_loss: float = 0.0, detector_efficiency: float = 1.0, distinguishable: bool = False):
        super().__init__(
            n_loops=0,  # These make calculate_n_params and calculate_output_dim work
            input_loss=input_loss,
            detector_efficiency=detector_efficiency,
            distinguishable=distinguishable,
        )

        self.u = None

    def sample(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        n_samples: int = 1,
        output_format: str = "dict",
        n_tiling: int = 1,
    ) -> dict[tuple[int, ...], int] | tuple[tuple[int, ...], ...] | list[list[int]] | npt.NDArray[np.int_]:
        """Returns samples from the output of a boson sampler with a fixed random unitary.

        Calls a fast numba implementation to compute the permanents.

        Args:
            input_state: description of input modes. The left-most entry corresponds to the first mode entering the loop.
            n_samples: Number of samples to draw. Defaults to 1.
            output_format: Output format for the samples. "dict", "tuple", "list", "array". Defaults to "dict"
            n_tiling: Number of sample concatenation. Defaults to 1.

        Returns:
            a dictionary of the form ``{state: counts}``
        """
        input_state, output_format = self._validate_input(input_state, output_format, n_tiling)

        if self.u is None:
            self.u = self.random_unitary_haar(len(input_state))

        # We can commute detector inefficiency through the beam splitters and into input loss
        input_loss_commuted = 1 - (1 - self.input_loss) * self.detector_efficiency

        if self.distinguishable:
            samples = distinguishable_sample(self.u, input_state, n_samples, input_loss_commuted)
        else:
            samples = clifford_sample(self.u, input_state, n_samples, input_loss=input_loss_commuted)

        return self.format_samples(samples, output_format=output_format)

    def _validate_input(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        output_format: str,
        n_tiling: int,
    ) -> tuple[npt.NDArray[np.int_], str]:
        input_state = np.asarray(input_state, dtype=np.int_)

        if len(input_state) < 2:
            raise ValueError("The minimal number of input modes must be 2.")

        if n_tiling != 1:
            raise ValueError("The tiling number must be 1.")

        if output_format not in ["tuple", "list", "array", "dict"]:
            output_format = "dict"
            warnings.warn('output_format must be "tuple", "list", "array" or "dict". Attempting with "dict"...')

        return input_state, output_format

    @staticmethod
    def random_unitary_haar(n: int) -> npt.NDArray[np.float64]:
        """Returns a U(n) unitary of size n - random by the Haar measure.

        Code inspired by "How to generate random matrices from the classical compact groups" (Francesco Mezzadri)

        Args:
            n: Size of the square matrix to be returned

        Returns:
            Haar random unitary of size n x n
        """
        z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
        q, r = np.linalg.qr(z)
        d = np.diagonal(r)
        ph = d / np.absolute(d)
        q = np.multiply(q, ph, q)
        return q
