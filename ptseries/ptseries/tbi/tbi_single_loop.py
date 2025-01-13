import math
import warnings

import numba as nb
import numpy as np
import numpy.typing as npt

from ptseries.tbi.tbi_abstract import TBISimulator
from ptseries.tbi.boson_sampler import utils

# Table of values of log(n!) for n up to 50
LOG_FACTORIAL_TABLE = np.array(
    [
        0.0,
        0.0,
        0.6931471805599453,
        1.791759469228055,
        3.1780538303479453,
        4.787491742782046,
        6.579251212010101,
        8.525161361065415,
        10.60460290274525,
        12.80182748008147,
        15.104412573075518,
        17.502307845873887,
        19.98721449566189,
        22.552163853123425,
        25.191221182738683,
        27.899271383840894,
        30.671860106080675,
        33.50507345013689,
        36.39544520803305,
        39.339884187199495,
        42.335616460753485,
        45.38013889847691,
        48.47118135183523,
        51.60667556776438,
        54.784729398112326,
        58.003605222980525,
        61.26170176100201,
        64.55753862700634,
        67.88974313718154,
        71.25703896716801,
        74.65823634883017,
        78.09222355331532,
        81.55795945611504,
        85.05446701758153,
        88.5808275421977,
        92.13617560368711,
        95.71969454214322,
        99.33061245478744,
        102.96819861451382,
        106.63176026064347,
        110.3206397147574,
        114.0342117814617,
        117.77188139974507,
        121.53308151543864,
        125.3172711493569,
        129.12393363912722,
        132.95257503561632,
        136.80272263732638,
        140.67392364823428,
        144.5657439463449,
    ],
    dtype=np.float64,
)


@nb.njit(nb.float64(nb.uint16))
def _compute_log_factorial(n: int) -> float:
    s = 0.0
    for i in range(1, n + 1):
        s += np.log(i)
    return s


@nb.njit(nb.float64(nb.uint16))
def _fast_log_factorial(n: int) -> float:
    if n > 49:
        return _compute_log_factorial(n)
    else:
        return LOG_FACTORIAL_TABLE[n]


@nb.njit()
def _isclose(a: float, b: float, rtol: float = 1e-08, atol: float = 1e-9) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


@nb.njit(nb.intp(nb.float64[:]))
def _rand_choice_nb(prob):
    return np.random.multinomial(1, prob).argmax()


@nb.njit(nb.float64(nb.uint16, nb.uint16, nb.uint16, nb.uint16, nb.float64, nb.boolean))
def _calculate_probabilities(N1: int, N2: int, n1: int, n2: int, theta: float, distinguishable: bool) -> float:
    t = np.cos(theta)  # reflection and transmission coefficients
    r = np.sin(theta)
    total_sum = 0
    for k in range(n1 + 1):  # evaluate sum
        l = N1 - k
        if l >= 0 and l <= n2:
            term1 = (r ** (n1 - k + l)) * (t ** (n2 + k - l))
            if term1 == 0:
                continue

            factorial_term = _fast_log_factorial(n1) + _fast_log_factorial(n2)

            if not distinguishable:
                factorial_term *= 0.5
                factorial_term += 0.5 * _fast_log_factorial(N1) + 0.5 * _fast_log_factorial(N2)
            factorial_term -= (
                _fast_log_factorial(k)
                + _fast_log_factorial(n1 - k)
                + _fast_log_factorial(l)
                + _fast_log_factorial(n2 - l)
            )

            factorial_term = np.exp(factorial_term)

            if distinguishable:
                term = term1**2 * factorial_term
            else:
                term = (-1) ** (n1 - k) * term1 * factorial_term

            total_sum += term

        else:
            continue
    if not distinguishable:
        total_sum **= 2
    return total_sum


@nb.njit(nb.float64(nb.uint16, nb.uint16, nb.float64))
def _calculate_probabilities_with_vacuum(n: int, k: int, theta: float):
    """Calculates probabilities at output of a beam splitter with input |n,0>."""
    if k > n:
        raise ValueError("Incorrect inputs in probabilities calculation.")

    term1 = (np.cos(theta) ** (2 * (k))) * (np.sin(theta) ** (2 * (n - k)))

    factorial_term = _fast_log_factorial(n) - _fast_log_factorial(k) - _fast_log_factorial(n - k)
    term = term1 * np.exp(factorial_term)
    return term


@nb.njit(nb.float64[:](nb.uint16, nb.uint16, nb.float64, nb.boolean))
def _calculate_probs_after_bs(n: int, m: int, theta: float, distinguishable: bool) -> npt.NDArray[np.float64]:
    probabilities = np.zeros(n + m + 1, dtype=np.float64)
    for N in range(n + m + 1):
        M = n + m - N
        val = _calculate_probabilities(N, M, n, m, theta, distinguishable)
        probabilities[N] = val if val > 1e-8 else 0.0  # This fixes a numerical issue with np.random.multinomial

    # even with floating point errors, the sum cannot be greater than 1
    probabilities = (1 - 1e-8) * probabilities / np.sum(probabilities)

    if not _isclose(np.sum(probabilities, dtype=np.float64), 1.0):
        raise ValueError("Probabilities do not sum to one")

    return probabilities


@nb.njit(nb.intp(nb.uint16, nb.float64))
def _n_photon_lost(n_photon: int, loss_angle: float):
    probs_loss = np.array(
        [(_calculate_probabilities_with_vacuum(n_photon, k, loss_angle)) for k in range(n_photon + 1)]
    )

    if not _isclose(np.sum(probs_loss), 1):
        raise ValueError("Probabilities do not sum to one")

    return _rand_choice_nb(probs_loss)


@nb.njit(
    nb.int32[:](nb.int_[:], nb.float64[:], nb.float64, nb.float64, nb.boolean),
)
def _get_one_sample(
    input_list: npt.NDArray[np.int_],
    theta_list: npt.NDArray[np.float64],
    bs_loss_angle: float,
    input_loss_angle: float,
    distinguishable: bool,
) -> npt.NDArray[np.int32]:
    """Returns a single sample from the output of a TBI."""
    N = len(theta_list)
    sample = np.empty(N + 1, dtype=np.int32)
    n_photon_loop = input_list[0]

    # Adding loss to the first input
    fl = not _isclose(input_loss_angle, math.pi / 2)
    if fl and (n_photon_loop > 0):
        n_photon_lost = _n_photon_lost(n_photon_loop, input_loss_angle)
        n_photon_loop -= n_photon_lost

    for i in range(N):
        # First, deal with input loss
        n_in = input_list[i + 1]
        if fl and (n_in > 0):  # ie if there is loss and photons in this mode
            n_photon_lost = _n_photon_lost(n_in, input_loss_angle)
            n_in -= n_photon_lost

        # Send this input state through a beam splitter
        probabilities = _calculate_probs_after_bs(n_photon_loop, n_in, theta_list[i], distinguishable)

        # Photons that go to the output
        output_photons = _rand_choice_nb(probabilities)

        # Photons that stay in the loop
        n_photon_loop = n_photon_loop + n_in - output_photons

        if not _isclose(bs_loss_angle, math.pi / 2):  # i.e if there is loss
            # We sample a number of lost photons using the probabilities probs_loss_1
            # Losing k1 photons is equivalent to having a beam splitter
            # that generates the state |k, n-k>
            # where the first mode is the environment

            # Loss in the measurement mode
            n_photon_lost = _n_photon_lost(int(output_photons), bs_loss_angle)
            output_photons -= n_photon_lost

            # Loss in the loop mode
            n_photon_lost = _n_photon_lost(n_photon_loop, bs_loss_angle)
            n_photon_loop -= n_photon_lost

        sample[i] = output_photons

    sample[N] = n_photon_loop

    return sample


class TBISingleLoop(TBISimulator):
    """A class that is used to efficiently sample from a single loop PT Series time bin interferometer (TBI).

    Args:
        postselected: Whether to perform postselection on the input and output state with equal number of photons.
            Defaults to False.
        distinguishable: Whether the photons interfere with each other or not. Defaults to False.
        input_loss: Optical losses at the input of the PT Series, between 0 and 1. Defaults to 0.0.
        bs_loss: Fraction of light lost at each beam splitter in the TBI, between 0 and 1. Defaults to 0.0.
        bs_noise: Standard deviation of Gaussian noise on a beam splitter angle, it can take any positive value.
            Defaults to 0.0.
        detector_efficiency: Detection efficiency of the detectors, from 0 to 1. Defaults to 1.0.
        n_signal_detectors: the number of detectors used for pseudo PNR detection. The default sets this to 0, which means true PNR is used rather than pseudo
        PNR.
        g2: the g2 of the source. The default is 0.0.
    """

    descr = "single-loop"

    def __init__(
        self,
        postselected: bool = False,
        distinguishable: bool = False,
        input_loss: float = 0.0,
        bs_loss: float = 0.0,
        bs_noise: float = 0.0,
        detector_efficiency: float = 1.0,
        n_signal_detectors: int = 0,
        g2: float = 0.0,
    ):
        super().__init__(
            n_loops=1,  # This simulator is specific to single-loop PT Series
            postselected=postselected,
            distinguishable=distinguishable,
            bs_loss=bs_loss,
            bs_noise=bs_noise,
            input_loss=input_loss,
            detector_efficiency=detector_efficiency,
        )

        self.g2 = g2
        self.n_signal_detectors = n_signal_detectors
        self.bs_loss_angle = math.acos(math.sqrt(bs_loss))
        # We can commute detector inefficiency through the beam splitters and into input loss
        input_loss_commuted = 1 - (1 - self.input_loss) * self.detector_efficiency
        self.input_loss_commuted_angle = math.acos(math.sqrt(input_loss_commuted))

        self._conditional_photon_distribution = utils.conditional_photon_distribution(self.g2, self.detector_efficiency)

    def sample(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64] | None = None,
        n_samples: int = 1,
        output_format: str = "dict",
        n_tiling: int = 1,
    ) -> dict[tuple[int, ...], int] | tuple[tuple[int, ...], ...] | list[list[int]] | npt.NDArray[np.int_]:
        r"""Returns samples from the output of a single loop TBI. Calls a fast numba implementation.

        Args:
            input_state: Input state to be used, for example :math:`(1,1,0)` for input state :math:`|110\rangle`
            theta_list: List of beam splitter angles
            n_samples: Number of samples to draw. Defaults to 1.
            output_format: Output format for the samples. "dict", "tuple", "list", "array". Defaults to "dict"
            n_tiling: Number of sample concatenations. Defaults to 1.
            heralded_auto_correlation: Auto-correlation value for heralded photons. Defaults to 0.0.
            quasi_number_resolution: Whether to use quasi-number resolution. Defaults to False.

        Returns:
            a dictionary of the form ``{state: counts}``
        """
        _input_state, theta_list, output_format = self._validate_input(input_state, theta_list, output_format, n_tiling)
        n_modes = len(_input_state)

        # The samples are inserted to raw_samples
        raw_samples = np.empty((n_samples, n_tiling * n_modes), dtype=int)
        theta_list_len = len(theta_list) // n_tiling
        temp_theta_list_len = theta_list_len

        for i in range(n_tiling):
            samples = self._jit_sampler(
                # Theta list is split by the tiling number
                theta_list[i * theta_list_len : temp_theta_list_len],
                _input_state,
                n_samples,
                self.bs_noise,
                self.bs_loss_angle,
                self.input_loss_commuted_angle,
                self._conditional_photon_distribution,
                distinguishable=self.distinguishable,
                postselected=self.postselected,
                n_signal_detectors=self.n_signal_detectors,
                detector_efficiency=self.detector_efficiency,
                g2=self.g2,
            )

            temp_theta_list_len += theta_list_len
            raw_samples[:, i * n_modes : (i + 1) * n_modes] = samples

        samples = raw_samples

        return self.format_samples(samples, output_format=output_format)

    def _validate_input(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64] | None,
        output_format: str,
        n_tiling: int,
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], str]:
        """Validates the input parameters for the TBI single loop calculation.

        Args:
            input_state: The input state of the system.
            theta_list: The list of theta values for the beam splitters.
            output_format: The desired output format.
            n_tiling: The number of tilings.

        Returns:
            Tuple of validated input parameters: input_state, theta_list, output_format.

        Raises:
            ValueError: If the input state is all zeros.
            ValueError: If the n_tiling value is less than 1.
            ValueError: If the length of theta_list does not match the number of beam splitters.
        """
        if theta_list is None:
            raise ValueError("theta_list must be provided.")

        input_state = np.asarray(input_state, dtype=np.int_)
        theta_list = np.asarray(theta_list, dtype=np.float64)
        n_modes = len(input_state)

        n_beam_splitters = self.calculate_n_beam_splitters(n_modes)

        if n_modes < 2:
            raise ValueError("The minimal number of input modes must be 2.")

        if np.all(input_state == 0):
            raise ValueError(
                "The input state must have at least one mode with a finite number of photons i.e. not vacuum on all modes."
            )

        if n_tiling < 1:
            raise ValueError("The minimal tiling number must be 1.")

        if len(theta_list) != n_beam_splitters * n_tiling:
            raise ValueError(
                f"List of thetas of length {n_beam_splitters * n_tiling} expected, received {len(theta_list)}."
            )

        if output_format not in ["tuple", "list", "array", "dict"]:
            output_format = "dict"
            warnings.warn('output_format must be "tuple", "list", "array" or "dict". Attempting with "dict"...')

        return input_state, theta_list, output_format

    @staticmethod
    @nb.njit(
        nb.int32[:, :](
            nb.float64[:],
            nb.int_[:],
            nb.int32,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64[:],
            nb.boolean,
            nb.boolean,
            nb.float64,
            nb.int32,
            nb.float64,
        ),
    )
    def _jit_sampler(
        theta_list: npt.NDArray[np.float64],
        input_state: npt.NDArray[np.int_],
        n_samples: int,
        bs_noise: float,
        bs_loss_angle: float,
        input_loss_angle: float,
        conditional_photon_distribution: npt.NDArray[np.float64],
        distinguishable: bool,
        postselected: bool,
        detector_efficiency: float = 1.0,
        n_signal_detectors: int = 0,
        g2: float = 0.0,
    ) -> npt.NDArray[np.int32]:
        """Fast numba implementation of sampling from the output of a single loop TBI."""
        n_modes = len(input_state)
        samples = np.empty((n_samples, n_modes), dtype=np.int32)

        if n_signal_detectors != 0:
            detection_matrix = utils.get_conditional_prob_matrix(n_signal_detectors, detector_efficiency)

        for k in nb.prange(n_samples):

            if _isclose(bs_noise, 0.0):
                theta_list_real = theta_list
            else:
                theta_list_real = np.empty(len(theta_list), dtype=np.float64)
                for i, theta in enumerate(theta_list):
                    theta_list_real[i] = np.random.normal(theta, bs_noise)

            if postselected:
                n_output_photons = 0
                n_input_photons = input_state.sum()
                # keep drawing samples until one with the correct number of photons is obtained.
                while n_output_photons != n_input_photons:

                    # only sample input state if g2>0
                    if g2 != 0.0:
                        multi_photon_input = utils.generate_photon_input(input_state, conditional_photon_distribution)
                    else:
                        multi_photon_input = input_state

                    new_state = _get_one_sample(
                        multi_photon_input, theta_list_real, bs_loss_angle, input_loss_angle, distinguishable
                    )
                    if n_signal_detectors != 0:
                        detected_state = utils.transform_sample_number_res(
                            new_state, detection_matrix, n_signal_detectors
                        )

                    else:
                        detected_state = new_state

                    n_output_photons = detected_state.sum()

                samples[k, :] = detected_state

            else:

                # only sample input state if g2>0
                if g2 != 0.0:
                    multi_photon_input = utils.generate_photon_input(input_state, conditional_photon_distribution)
                else:
                    multi_photon_input = input_state

                new_state = _get_one_sample(
                    multi_photon_input, theta_list_real, bs_loss_angle, input_loss_angle, distinguishable
                )

                if n_signal_detectors != 0:
                    detected_state = utils.transform_sample_number_res(new_state, detection_matrix, n_signal_detectors)

                else:
                    detected_state = new_state

                samples[k, :] = detected_state

        return samples
