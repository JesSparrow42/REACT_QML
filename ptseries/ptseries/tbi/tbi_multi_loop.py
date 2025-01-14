import warnings

import numba as nb
import numpy as np
import numpy.typing as npt
import sympy as sym

from ptseries.tbi.boson_sampler.boson_sampler import clifford_sample
from ptseries.tbi.boson_sampler.distinguishable_sampler import distinguishable_sample
from ptseries.tbi.tbi_abstract import TBISimulator


class TBIMultiLoop(TBISimulator):
    """A class that is used to efficiently sample from a multi-loop PT Series time bin interferometer (TBI).

    Args:
        n_loops: Number of loops in the TBI system. It defaults to 2.
        loop_lengths: Lengths of the loops in the TBI system. If loop_lengths is specified then n_loops is not required,
            but if both are specified then they should be consistent. It defaults to None.
        postselected: Whether to perform postselection on the input and output state with equal number of photons.
            Defaults to False.
        distinguishable: Indicates whether the photons interfere with each other or not. It defaults to False.
        input_loss: Optical losses at the input of the PT Series, between 0 and 1. It defaults to 0.0.
        bs_loss: Fraction of light lost at each beam splitter (one at each input arm) in the TBI, between 0 and 1. It
            defaults to 0.0.
        detector_efficiency: Efficiency of the detectors, from 0 to 1. It defaults to 1.0.
    """

    descr = "multi-loop"

    def __init__(
        self,
        n_loops: int = 2,
        loop_lengths: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | None = None,
        postselected: bool = False,
        distinguishable: bool = False,
        input_loss: float = 0.0,
        bs_loss: float = 0.0,
        detector_efficiency: float = 1.0,
    ):
        super().__init__(
            n_loops=n_loops,
            loop_lengths=loop_lengths,
            postselected=postselected,
            distinguishable=distinguishable,
            input_loss=input_loss,
            bs_loss=bs_loss,
            detector_efficiency=detector_efficiency,
        )
        self.loop_lengths = np.asarray(self.loop_lengths, dtype=np.int_)

        self.n_modes: int = 0
        self.bs_loss = bs_loss

    def sample(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64] | None = None,
        n_samples: int = 1,
        calculate_symbolic: bool | None = None,
        output_format: str = "dict",
        n_tiling: int = 1,
    ) -> dict[tuple[int, ...], int] | tuple[tuple[int, ...], ...] | list[list[int]] | npt.NDArray[np.int_]:
        r"""Returns samples from the output of a multi-loop TBI.

        Calls a fast numba implementation to compute the permanents.

        Args:
            input_state: input state to be used, for example :math:`(1,1,0)` for input state :math:`|110\rangle`
            theta_list: List of beam splitter angles
            n_samples: Number of samples to draw. Defaults to 1.
            calculate_symbolic: Indicates if the unitary is computed once with symbolic values for the angles or
                recomputed numerically each time for new beam splitter angles. If None, default behavior is used with
                symbolic for small n_loops and numerical for large n_loops.
            output_format: Output format for the samples. "dict", "tuple", "list", "array". Defaults to "dict"
            n_tiling: Number of sample concatenation. Defaults to 1.

        Returns:
             a dictionary of the form ``{state: counts}``
        """
        input_state, theta_list, n_beam_splitters, calculate_symbolic, output_format = self._validate_input(
            input_state, theta_list, output_format, n_tiling, calculate_symbolic
        )

        if calculate_symbolic:
            self.u_sym = self._get_unitary_tbi_symbolic(self.n_modes, self.n_loops, self.loop_lengths, n_beam_splitters)

        # The samples are inserted to raw_samples
        raw_samples = np.empty((n_samples, n_tiling * self.n_modes), dtype=int)
        theta_list_len = len(theta_list) // n_tiling
        temp_theta_list_len = theta_list_len

        # Sampler is run for each tiling
        for i in range(n_tiling):
            if calculate_symbolic:
                self.u = self.u_sym(*theta_list[i * theta_list_len : temp_theta_list_len])  # type: ignore

            else:
                if self.bs_loss > 0:
                    bs_loss = np.arcsin(np.sqrt(self.bs_loss))
                    bs_loss = bs_loss * np.ones((n_beam_splitters, 2), dtype=np.float64)
                    self.u = _get_unitary_tbi_numerical_with_loss(
                        self.n_modes,
                        theta_list[i * theta_list_len : temp_theta_list_len],
                        self.n_loops,
                        self.loop_lengths,
                        n_beam_splitters,
                        bs_loss,
                    )

                else:
                    self.u = _get_unitary_tbi_numerical(
                        self.n_modes,
                        theta_list[i * theta_list_len : temp_theta_list_len],
                        self.n_loops,
                        self.loop_lengths,
                        n_beam_splitters,
                    )

            # We can commute detector inefficiency through the beam splitters and into input loss
            input_loss_commuted = 1 - (1 - self.input_loss) * self.detector_efficiency

            if self.distinguishable:
                samples = distinguishable_sample(
                    self.u, input_state, n_samples, input_loss=input_loss_commuted, postselected=self.postselected
                )
            else:
                samples = clifford_sample(
                    self.u, input_state, n_samples, input_loss=input_loss_commuted, postselected=self.postselected
                )

            if self.bs_loss > 0:
                samples = samples[:, : self.n_modes]

            temp_theta_list_len += theta_list_len
            raw_samples[:, i * self.n_modes : (i + 1) * self.n_modes] = samples

        samples = raw_samples

        return self.format_samples(samples, output_format=output_format)

    def _validate_input(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64] | None,
        output_format: str,
        n_tiling: int,
        calculate_symbolic: bool | None,
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], int, bool, str]:
        """Validates the input parameters for the sample method.

        Args:
            input_state: Input state.
            theta_list: List of beam splitter angles.
            output_format: Output format for the samples.
            n_tiling: Number of sample concatenation.
            calculate_symbolic: Whether the unitary is computed symbolically or numerically.

        Returns:
            Tuple of validated input parameters: input_state, theta_list, n_beam_splitters, calculate_symbolic,
                output_format.

        Raises:
            ValueError: If the input parameters are invalid.
        """
        input_state = self._validate_input_state(input_state)
        theta_list, n_beam_splitters = self._validate_theta_list(theta_list, n_tiling)
        calculate_symbolic = self._validate_calculate_symbolic(calculate_symbolic)
        self._validate_n_tiling(n_tiling)
        output_format = self._validate_output_format(output_format)

        return input_state, theta_list, n_beam_splitters, calculate_symbolic, output_format

    def _validate_input_state(
        self, input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_]
    ) -> npt.NDArray[np.int_]:
        input_state = np.asarray(input_state, dtype=np.int_)
        self.n_modes: int = len(input_state)
        if len(input_state) < max(self.loop_lengths) + 1:
            raise ValueError(
                f"The input_state must have at least as many modes as the longest {max(self.loop_lengths)  + 1}."
            )
        if np.all(input_state == 0):
            raise ValueError(
                "The input state must have at least one mode with a finite number of photons i.e. not vacuum on all modes."
            )
        return input_state

    def _validate_theta_list(
        self, theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64] | None, n_tiling: int
    ) -> tuple[npt.NDArray[np.float64], int]:
        if theta_list is None:
            raise ValueError("theta_list must be provided.")
        theta_list = np.asarray(theta_list, dtype=np.float64)
        n_beam_splitters = self.calculate_n_beam_splitters(self.n_modes)
        if len(theta_list) != n_beam_splitters * n_tiling:
            raise ValueError(
                f"List of thetas of length {n_beam_splitters * n_tiling} expected, received {len(theta_list)}."
            )
        return theta_list, n_beam_splitters

    def _validate_calculate_symbolic(self, calculate_symbolic: bool | None) -> bool:
        if calculate_symbolic is None:
            if self.n_loops > 5 or self.n_modes > 20 or self.bs_loss > 0:
                calculate_symbolic = False
            else:
                calculate_symbolic = True
        else:
            if isinstance(calculate_symbolic, bool):
                if not np.isclose(self.bs_loss, 0):
                    raise ValueError("calculate_symbolic cannot be set to True when bs_loss > 0")
            else:
                raise ValueError("calculate_symbolic must be a boolean or None")
        return calculate_symbolic

    def _validate_n_tiling(self, n_tiling):
        if n_tiling < 1:
            raise ValueError("The minimal tiling number must be 1.")

    def _validate_output_format(self, output_format):
        if output_format not in ["tuple", "list", "array", "dict"]:
            output_format = "dict"
            warnings.warn('output_format must be "tuple", "list", "array" or "dict". Attempting with "dict"...')
        return output_format

    def _get_unitary_tbi_symbolic(
        self,
        n_modes: int,
        n_loops: int,
        loop_lengths: npt.NDArray[np.int_],
        n_beam_splitters: int,
    ) -> sym.Function:
        """Builds the unitary matrix from 2x2 building blocks using symbolic computation.

        Args:
            n_modes: Number of input modes.
            n_loops: Number of loops in the TBI system.
            loop_lengths: Array of loop lengths.
            n_beam_splitters: Number of beam splitters.

        Returns:
            Symbolic expression for the unitary matrix.
        """
        theta_arr = np.array([sym.Symbol(f"th{i}") for i in range(n_beam_splitters)])

        unitary = sym.eye(n_modes, dtype=np.complex128)
        current_idx = n_beam_splitters - 1

        for loop in reversed(range(n_loops)):
            delay = loop_lengths[loop]
            for i in reversed(range(n_modes - delay)):
                u = sym.eye(n_modes, dtype=np.complex128)
                u_2_2 = self._get_unitary_single_symbolic(theta_arr[current_idx])
                u[i, i] = u_2_2[0, 0]
                u[i, i + delay] = u_2_2[0, 1]
                u[i + delay, i] = u_2_2[1, 0]
                u[i + delay, i + delay] = u_2_2[1, 1]
                unitary = unitary * u
                current_idx -= 1

        unitary_sym = sym.lambdify(theta_arr.reshape(1, n_beam_splitters, order="A")[0], unitary, "numpy")

        return unitary_sym

    @staticmethod
    def _get_unitary_single_symbolic(theta: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
        """Returns the unitary matrix of a beam splitter with an angle theta using symbolic computation.

        Args:
            theta: Angle of the beam splitter.

        Returns:
            Unitary matrix of the beam splitter.
        """
        return np.array([[sym.cos(theta), sym.sin(theta)], [-sym.sin(theta), sym.cos(theta)]])


@nb.njit(
    nb.complex128[:, :](
        nb.int32,
        nb.float64[:],
        nb.int32,
        nb.int_[:],
        nb.int32,
    ),
)
def _get_unitary_tbi_numerical(
    n_modes: int,
    theta_list: npt.NDArray[np.float64],
    n_loops: int,
    loop_lengths: npt.NDArray[np.int_],
    n_beam_splitters: int,
) -> npt.NDArray[np.complex128]:
    """Build the unitary matrix from 2x2 building blocks using numerical computation."""
    dty = np.complex128  # np.float64 #
    unitary = np.eye(n_modes, dtype=dty)
    current_idx = n_beam_splitters - 1

    for loop in range(n_loops):
        loop = n_loops - 1 - loop
        delay = loop_lengths[loop]
        for i in range(n_modes - delay):
            i = n_modes - delay - 1 - i
            u = np.eye(n_modes, dtype=dty)
            u[i, i] = np.cos(theta_list[current_idx])
            u[i, i + delay] = np.sin(theta_list[current_idx])
            u[i + delay, i] = -np.sin(theta_list[current_idx])
            u[i + delay, i + delay] = np.cos(theta_list[current_idx])
            unitary = np.dot(unitary, u)
            current_idx -= 1

    return unitary


@nb.njit(
    nb.complex128[:, :](nb.int32, nb.float64[:], nb.int32, nb.int_[:], nb.int32, nb.float64[:, :]),
)
def _get_unitary_tbi_numerical_with_loss(
    n_modes: int,
    theta_list: npt.NDArray[np.float64],
    n_loops: int,
    loop_lengths: npt.NDArray[np.int_],
    n_beam_splitters: int,
    bs_loss: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Build the unitary matrix from 2x2 building blocks using numerical computation.

    Each beam splitter has a loss on both of its input arms. Each beam splitter is coupled to 2 vacuum modes (one per input arm).
    """
    dty = np.complex128
    unitary = np.eye(n_modes + 2 * n_beam_splitters, dtype=dty)
    current_idx = n_beam_splitters - 1

    s1 = n_modes
    s2 = n_modes + n_beam_splitters

    for loop in range(n_loops):
        loop = n_loops - 1 - loop
        delay = loop_lengths[loop]
        for i in range(n_modes - delay):
            i = n_modes - delay - 1 - i

            u1 = np.eye(n_modes + 2 * n_beam_splitters, dtype=dty)
            u2 = np.eye(n_modes + 2 * n_beam_splitters, dtype=dty)
            u3 = np.eye(n_modes + 2 * n_beam_splitters, dtype=dty)

            u1[i, i] = np.cos(theta_list[current_idx])
            u1[i, i + delay] = np.sin(theta_list[current_idx])
            u1[i + delay, i] = -np.sin(theta_list[current_idx])
            u1[i + delay, i + delay] = np.cos(theta_list[current_idx])
            unitary = np.dot(unitary, u1)

            u2[s1, s1] = np.cos(bs_loss[current_idx, 0])
            u2[s1, i] = -np.sin(bs_loss[current_idx, 0])
            u2[i, s1] = np.sin(bs_loss[current_idx, 0])
            u2[i, i] = np.cos(bs_loss[current_idx, 0])
            unitary = np.dot(unitary, u2)

            u3[i + delay, i + delay] = np.cos(bs_loss[current_idx, 1])
            u3[i + delay, s2] = np.sin(bs_loss[current_idx, 1])
            u3[s2, i + delay] = -np.sin(bs_loss[current_idx, 1])
            u3[s2, s2] = np.cos(bs_loss[current_idx, 1])
            unitary = np.dot(unitary, u3)

            current_idx -= 1
            s1 += 1
            s2 += 1

    return unitary
