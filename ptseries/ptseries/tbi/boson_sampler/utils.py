import numba as nb
import numpy as np
import numpy.typing as npt

from ptseries.tbi.boson_sampler import source_statistics

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


@nb.njit
def add_loss(input_state: np.ndarray, input_loss: float):
    """Randomly removes photons from the input state with statistics given by the loss."""
    lossy_input_state = np.zeros(len(input_state), dtype=np.int32)
    for i in range(len(input_state)):
        n_photons = input_state[i]
        if n_photons > 0:
            probs_loss = np.array(
                [
                    (
                        _calculate_prefactor_with_vacuum(n_photons, k)
                        * (np.cos(input_loss) ** (k))
                        * (np.sin(input_loss) ** (n_photons - k))
                    )
                    ** 2
                    for k in range(n_photons + 1)
                ]
            )
            n_photon_lost_choices = list(range(n_photons + 1))
            n_photon_lost = rand_choice_nb(n_photon_lost_choices, probs_loss)
            lossy_input_state[i] = n_photons - n_photon_lost
    return lossy_input_state


@nb.njit
def rand_choice_nb(arr, prob):
    """Randomly selects an element from an array with probabilities."""
    cumsum = np.cumsum(prob)
    r = np.random.random()
    r *= cumsum[-1]  # Fix for rare normalisation issue
    return arr[np.searchsorted(cumsum, r, side="right")]


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
        return LOG_FACTORIAL_TABLE[n].item()


@nb.njit
def _calculate_prefactor(n: int, m: int, k: int, p: int):
    """Calculates numerical factor used in expression of amplitudes at output of a beam splitter with input |n,m>."""
    if k > n or p > m:
        raise Exception("Incorrect inputs in amplitude calculation.")

    term1 = 0.5 * _fast_log_factorial(k + p) + 0.5 * _fast_log_factorial(n + m - p - k)
    term2 = 0.5 * _fast_log_factorial(n) + 0.5 * _fast_log_factorial(m)
    term3 = (
        _fast_log_factorial(n)
        + _fast_log_factorial(m)
        - (_fast_log_factorial(k) + _fast_log_factorial(n - k) + _fast_log_factorial(p) + _fast_log_factorial(m - p))
    )
    return np.exp(term1 + term3 - term2)


@nb.njit
def _calculate_prefactor_with_vacuum(n: int, k: int):
    """Calculates numerical factor used in expression of amplitudes at output of a beam splitter with input |n,0>."""
    if k > n:
        raise Exception("Incorrect inputs in amplitude calculation.")

    return _calculate_prefactor(n, 0, k, 0)


def conditional_photon_distribution(
    heralded_autocorrelation: float, herald_eff: float, number_cutoff: int = 5
) -> npt.NDArray[np.float64]:
    """Return the conditional photon number distribution of a heralded photon source."""
    herald_detector_number = 1
    p_coeff_init = 0.1

    photon_source = source_statistics.HeraldedSource(
        herald_detector_number, herald_eff, p_coeff_init
    )  # initialise with 1 herald detector
    photon_source.set_heralded_auto_correlation(heralded_autocorrelation)

    # generate an array representing the conditional photon number distribution
    conditional_distribution = np.array([photon_source.prob_n_given_herald(n) for n in range(number_cutoff + 1)])

    norm_dist = conditional_distribution / (sum(conditional_distribution))

    return norm_dist


@nb.njit
def generate_photon_input(
    sample_request: npt.NDArray[np.int_], conditional_distribution: npt.NDArray[np.float64]
) -> npt.NDArray[np.int_]:
    """Transforms a requested sample pattern into a photon input, based on the source statistics.

    These are parameterized by the second order heralded auto correlation at zero time delay (g2) and the total loss on
    the herald arm
    """
    n_modes = len(sample_request)

    photon_number_array = np.arange(0, len(conditional_distribution))

    input_sample = np.empty(n_modes, dtype=np.int_)

    for idx, sample in enumerate(sample_request):
        if sample == 1:
            # Draw from the conditional photon number dist. This has the same functionality as np.random.choice(photon_number_array, p = conditional distribution). but this is not supported by numba.

            input_sample[idx] = photon_number_array[
                np.searchsorted(np.cumsum(conditional_distribution), np.random.random(), side="right")
            ]

        else:
            input_sample[idx] = 0

    return input_sample


@nb.njit(nb.int64[:](nb.int64, nb.float64, nb.int64), fastmath=True)
def _gen_binomial(n_photons: int, pairwise_indist: float, n_samples: int = 1):
    """Generate the number of indistinguishable photons for each sample.

    Draw from the binomial distribution based on the pairwise indistinguishability
    to generate the number of indistinguishable photons for each sample.

    Args:
        n_photons: Number of photons
        pairwise_indist: Value between 0 (distinguishable) and 1 (indistinguishable)
        n_samples: Number of samples

    Returns:
        Array with the number of indistinguishable photons for each sample.
    """
    if pairwise_indist < 0.0 or pairwise_indist > 1.0:
        raise ValueError("pairwise_indist must be a value between 0.0 and 1.0")

    # Use a manual binomial generator
    binomial_samples = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        count = 0
        for _ in range(n_photons):
            if np.random.random() <= pairwise_indist:
                count += 1
        binomial_samples[i] = count

    return binomial_samples


@nb.njit(nb.int64[:](nb.int64[:], nb.int64), fastmath=True)
def _gen_indist_sample(input_state: npt.NDArray[np.int_], n_photons_dist: int) -> npt.NDArray[np.int_]:
    """Generate a single sample with only indistinguishable photons.

    Given an arbitrary input state and number of distinguishable photons,
    generate a single sample with only indistinguishable photons.

    Args:
        input_state: Input Fock state
        n_photons_dist: Number of distinguishable photons

    Returns:
        A sample containing only indistinguishable photons.
    """
    n = input_state.size
    array_index = np.empty(np.sum(input_state), dtype=np.int64)
    idx_count = 0

    # Flatten the input_state into array_index
    for i in range(n):
        for idx_count, _ in enumerate(range(input_state[i])):
            array_index[idx_count] = i

    # Manual random choice without replacement
    chosen_indices = np.empty(n_photons_dist, dtype=np.int64)
    remaining = array_index.size
    for i in range(n_photons_dist):
        rand_idx = np.random.randint(remaining)
        chosen_indices[i] = array_index[rand_idx]
        array_index[rand_idx] = array_index[remaining - 1]
        remaining -= 1

    # Create a copy of the input_state and update with the chosen sample
    temp_state = np.copy(input_state)
    for idx in chosen_indices:
        temp_state[idx] -= 1

    return temp_state


@nb.njit((nb.int64[:], nb.float64, nb.int64), fastmath=True)
def get_input_distribution(input_state, pairwise_indist=1.0, n_samples=1):
    """Generate samples of indistinguishable and distinguishable Fock states.

    Given an arbitrary input state and pairwise indistinguishability, generate n_samples
    of indistinguishable and distinguishable Fock states.

    Args:
        input_state: Input Fock state (numpy array)
        pairwise_indist: Value between 0 (distinguishable) and 1 (indistinguishable)
        n_samples: Number of samples

    Returns:
        A tuple of Fock states (numpy arrays) for indistinguishable and distinguishable samples.
    """
    n_photons = np.sum(input_state)
    n_photons_indist = _gen_binomial(n_photons, pairwise_indist, n_samples)

    indist_samples = np.empty((n_samples, input_state.size), dtype=np.int64)
    dist_samples = np.empty((n_samples, input_state.size), dtype=np.int64)

    for i in range(n_samples):
        n_indist = n_photons_indist[i]
        indist_sample = _gen_indist_sample(input_state, int(n_photons - n_indist))
        indist_samples[i] = indist_sample
        dist_samples[i] = input_state - indist_sample

    return indist_samples, dist_samples


@nb.njit
def _binomial(n: int, k: int) -> float:
    """A fast way to calculate binomial coefficients."""
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


@nb.njit
def _prob_m_clicks_given_n(
    m: int, n: int, detector_number: int, detector_efficiency: float, dark_count: float = 0.0
) -> float:
    """Conditional probability of m detection events given n incident photons with multiplexed detectors (with a balanced splitting network).

    Args:
        m: Number of detection events.
        n: Number of incident photons.
        detector_number: Number of multiplexed detectors.
        detector_efficiency: Detector efficiency (between 0 and 1).
        dark_count: Dark count rate (counts per second).

    Returns:
        the conditional probability
    """
    front_matter = _binomial(detector_number, m) * (1 - dark_count) ** (detector_number - m)

    back_matter = [
        _binomial(m, r)
        * (dark_count - 1) ** (m - r)
        * ((1 - detector_efficiency) + r * detector_efficiency / detector_number) ** n
        for r in range(0, m + 1)
    ]

    prob = front_matter * sum(back_matter)

    if prob < 0.0:  # check floating point error
        prob = 0.0

    return prob


@nb.njit(parallel=True)
def get_conditional_prob_matrix(
    detector_number: int, detector_efficiency: float, dark_count: float = 0.0, sample_max: int = 10
) -> npt.NDArray[np.float64]:
    """Generate a matrix of conditional probability distributions with elements p_{m,n} = p(m|n).

    Args:
        detector_number: Number of multiplexed detectors.
        detector_efficiency: Detector efficiency (between 0 and 1).
        dark_count: Dark count rate (counts per second).
        sample_max: Maximum number of samples. Defaults to 10.
    """
    # generate a matrix of conditional probability distributions with elements p_{m,n} = p(m|n)

    prob_array = np.empty((sample_max + 1, detector_number + 1), dtype=np.float64)

    for n in nb.prange(0, sample_max + 1):
        for m in nb.prange(0, detector_number + 1):
            prob_array[n][m] = _prob_m_clicks_given_n(m, n, detector_number, detector_efficiency, dark_count=dark_count)

    return prob_array


@nb.njit
def transform_sample_number_res(
    input_sample: npt.NDArray[np.int_], conditional_probability: npt.NDArray[np.float64], detector_number: int
) -> npt.NDArray[np.int_]:
    """Apply conditional probability distribution to an input sample.

    Args:
        input_sample: A single input sample
        conditional_probability: Array of conditional probabilities
        detector_number: The number of multiplexed detectors

    Returns:
        transformed sample
    """
    output_sample = np.zeros(np.shape((input_sample))[0], dtype=np.int32)

    # Loop through each element of the input sample
    for idx, number in enumerate(input_sample):
        # Get the corresponding conditional probability distribution
        probs = conditional_probability[number]

        # Perform manual sampling based on the probabilities
        rand_val = np.random.rand()  # Generate a random number between 0 and 1
        cumulative_prob = 0.0

        # Loop through detector choices and select based on the cumulative probability
        for i in range(detector_number + 1):
            cumulative_prob += probs[i]
            if rand_val < cumulative_prob:
                output_sample[idx] = i
                break

    return output_sample
