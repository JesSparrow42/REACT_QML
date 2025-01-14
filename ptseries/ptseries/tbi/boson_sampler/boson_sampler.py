import numba as nb
import numpy as np
import numpy.typing as npt

from ptseries.tbi.boson_sampler.permanent_functions import subperms1, subperms2
from ptseries.tbi.boson_sampler.utils import add_loss, rand_choice_nb


@nb.njit
def clifford_sample(
    U: npt.NDArray[np.complex128 | np.float64],
    input_state: npt.NDArray[np.int_],
    n_samples: int = 1,
    input_loss: float = 0,
    postselected: bool = False,
) -> npt.NDArray[np.int_]:
    """Samples from the output state of a boson sampler.

    Returns a list of numpy arrays corresponding to output samples
    """
    input_loss_angle = np.arccos(np.sqrt(input_loss))

    n_modes = U.shape[0]
    output_samples = np.empty((n_samples, n_modes), dtype=np.int_)
    in_modelist = _input_tf(input_state)
    n_photons_in = len(in_modelist)

    A = _numba_ix(np.asarray(U, dtype=np.complex128), np.arange(n_modes), in_modelist)

    for k in range(n_samples):
        if postselected:
            n_photons_out = 0
            # keep drawing samples until one with the correct number of photons is obtained in the main modes.
            while n_photons_out != n_photons_in:
                output_state = _clifford_single_sample(
                    U, A, input_state, input_loss, input_loss_angle, n_photons_in, n_modes, postselected
                )
                n_photons_out = output_state[: len(input_state)].sum()

        else:
            output_state = _clifford_single_sample(
                U, A, input_state, input_loss, input_loss_angle, n_photons_in, n_modes, postselected
            )

        output_samples[k, :] = output_state

    return output_samples


@nb.njit
def _clifford_single_sample(
    U: npt.NDArray[np.complex128 | np.float64],
    A: npt.NDArray[np.complex128 | np.float64],
    input_state: npt.NDArray[np.int_],
    input_loss: float,
    input_loss_angle: float,
    n_photons_in: int,
    n_modes: int,
    postselected: bool,
) -> npt.NDArray[np.int_]:
    """Returns one sample of a boson sampler."""
    if input_loss > 0 and not postselected:
        lossy_inputs = add_loss(input_state, input_loss_angle)
        if np.all(lossy_inputs == 0):
            output_state = np.zeros(n_modes, dtype=np.int_)
            return output_state

        in_modelist = _input_tf(lossy_inputs)
        n_photons_in = len(in_modelist)
        A = _numba_ix(np.asarray(U, dtype=np.complex128), np.arange(n_modes), in_modelist)

    # Uniformly randomly permute the columns by in-place shuffling of the transpose of A
    np.random.shuffle(A.T)

    if n_photons_in == 1:
        firstcol = A[:, 0]
        # Get the first pmf (from the first column) and sample from it to get the first output mode
        pmf = np.square(np.abs(firstcol))
        pmf = pmf / np.sum(pmf)

        s = [rand_choice_nb(np.arange(n_modes), pmf)]

        output_state = _output_tf(s, n_modes)
        return output_state

    else:
        out_modelist = np.array([0])
        permlist = np.array([0])

        firstcol = A[:, 0]
        # Get the first pmf (from the first column) and sample from it to get the first output mode
        pmf = np.square(np.abs(firstcol))
        pmf = pmf / np.sum(pmf)

        s = rand_choice_nb(np.arange(n_modes), pmf)

        out_modelist[0] = s

        for modenum in range(1, n_photons_in):
            # Calculate subpermanents using first Clifford and Clifford method: https://arxiv.org/abs/1706.01260
            # if total number of photons is less than 10
            if n_photons_in < 10:
                # Pick out the correct rows and columns
                M = _numba_ix(A, out_modelist, np.arange(modenum + 1))

                # We need to compute all of the submatrix permanents in the Laplace expansion.
                # These get used, in combination with matrix elements,
                # to form the conditional pmf for sampling the next output mode
                sp = subperms1(M)

            # Calculate subpermanents using second Clifford and Clifford method: https://arxiv.org/abs/1706.01260
            # if total number of photons is 10 or more
            else:
                # Pick out the correct rows and columns
                M = _numba_ix(A, np.arange(A.shape[0]), np.arange(modenum + 1))

                # We need to compute all of the submatrix permanents in the Laplace expansion.
                # These get used, in combination with matrix elements,
                # to form the conditional pmf for sampling the next output mode
                sp = subperms2(M, _output_tf(out_modelist, n_modes))

            # Multiply the sub-permanents by the correct matrix elements
            permvector = np.dot(A[:, np.arange(modenum + 1)], sp.T).T

            # Get the pmf
            pmf = np.square(np.abs(permvector))
            pmf /= np.sum(pmf)

            # Sample the next output mode from the pmf
            nextmode = rand_choice_nb(np.arange(n_modes), pmf)

            out_modelist = np.append(out_modelist, nextmode)

        final_perm = permvector[nextmode]
        np.append(permlist, final_perm)

        output_state = _output_tf(np.sort(out_modelist), n_modes)

        return output_state


@nb.njit
def _numba_ix(
    arr: npt.NDArray[np.complex128 | np.float64], rows: npt.NDArray[np.int_], cols: npt.NDArray[np.int_]
) -> npt.NDArray[np.complex128 | np.float64]:
    """Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays."""
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int64)

    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))


@nb.njit
def _input_tf(vector: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Map the input modes to a different convention (used in boson sampler).

    Transform the input [0,1,1,1] into [1,2,3].
    """
    input_ = [0] * np.sum(vector)
    s = 0
    for i in range(len(vector)):
        for _ in range(vector[i]):
            input_[s] = i
            s += 1
    return np.array(input_, dtype=np.int_)


@nb.njit
def _output_tf(vector: npt.NDArray[np.int_] | list[int], n_modes: int) -> npt.NDArray[np.int_]:
    """Performs the opposite task as input_tf."""
    output = np.zeros(n_modes, dtype=np.int_)
    for vi in vector:
        output[int(vi)] += 1
    return output
