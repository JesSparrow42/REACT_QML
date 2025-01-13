import numba as nb
import numpy as np
import numpy.typing as npt

from ptseries.tbi.boson_sampler.utils import add_loss, rand_choice_nb


def distinguishable_sample(
    u: npt.NDArray[np.complex128 | np.float64],
    input_state: npt.NDArray[np.int_],
    n_samples: int = 1,
    input_loss: float = 0,
    postselected: bool = False,
) -> npt.NDArray[np.int_]:
    """Samples from the output state of a boson sampler where the photons do not interfere with each other.

    Returns a list of numpy arrays corresponding to output samples
    """
    samples = samples_distinguishable(u, input_state, n_samples, input_loss, postselected)

    return samples


@nb.njit
def samples_distinguishable(
    u: npt.NDArray[np.complex128 | np.float64],
    input_state: npt.NDArray[np.int_],
    n_samples: int = 1,
    input_loss: float = 0,
    postselected: bool = False,
) -> npt.NDArray[np.int_]:
    """Returns samples from a boson sampler where the photons are not interfering."""
    n_modes = u.shape[0]
    samples = np.empty((n_samples, n_modes), dtype=np.int_)
    input_loss_angle = np.arccos(np.sqrt(input_loss))

    for k in range(n_samples):
        if postselected:
            n_output_photons = 0
            n_input_photons = input_state.sum()
            while n_output_photons != n_input_photons:
                sample = distinguishable_one_sample(u, input_state)
                n_output_photons = sample[: len(input_state)].sum()

        else:
            if input_loss > 0:
                lossy_inputs = add_loss(input_state, input_loss_angle)
                sample = distinguishable_one_sample(u, lossy_inputs)
            else:
                sample = distinguishable_one_sample(u, input_state)

        samples[k, :] = sample

    return samples


@nb.njit
def distinguishable_one_sample(
    u: npt.NDArray[np.complex128 | np.float64], input_state: npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    """Returns one sample of a boson sampler where the photons are not interfering."""
    c_U = np.square(np.abs(u))

    n_modes = c_U.shape[0]

    output_modes = np.arange(n_modes)
    sample = np.zeros(n_modes, dtype=np.int_)

    for i in range(len(input_state)):
        ni = input_state[i]
        probs = c_U[:, i]
        probs = probs / np.sum(probs)

        for k in range(ni):
            output_mode = rand_choice_nb(output_modes, probs)
            output_mode = int(output_mode)
            sample[output_mode] += 1

    return sample
