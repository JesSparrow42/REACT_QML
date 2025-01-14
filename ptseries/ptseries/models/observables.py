import torch
import numpy as np
import numba as nb
from abc import ABC, abstractmethod


class Observable(ABC):
    """Base class for an observable or ensemble of observables."""

    # String describing the observable
    descr = ""

    def __call__(self, measurements: dict):
        return self.estimate(measurements)

    @abstractmethod
    def estimate(self, measurements: dict):
        """Returns an estimate of the observable based on the measurements.

        Args:
            measurements: dict of state: counts

        Returns:
            tensor with values of the observable
        """
        pass


class Correlations(Observable):
    """Observable corresponding to all 2-point output correlators.

    This observable exhibits interesting quantum properties. First,
    2-point correlators can be estimated in quadratic time classically, whereas
    the PT Series can be used to estimate them in linear time. Second,
    interfering photons in a PT Series have richer output correlations
    than non-interfering photons.

    For n output modes, this observable returns n*(n-1)/2 values. The parameter
    shift rule does not work for this observable, but a small value of about pi/10
    (which is the default) works well in practice
    """

    descr = "correlations"

    def estimate(self, measurements: dict):
        """Returns tensor of all two-point correlators in output state given a dict of measurements.

        If the measured states are dimension n, then the output is of dimension n(n-1)/2

        Args:
            measurements: dict of state: counts

        Returns:
            1d tensor with values of the correlators
        """
        corrs_numpy = self._correlations_numba(
            np.array(list(measurements.keys())), np.array(list(measurements.values()))
        )
        return torch.tensor(corrs_numpy)

    @staticmethod
    @nb.njit
    def _correlations_numba(list_states, list_counts):
        total_counts = np.sum(list_counts)
        state_length = len(list_states[0])

        avg_photons = _avg_photons_numba(list_states, list_counts)
        std_vector = _std_photons_numba(list_states, list_counts, avg_photons)

        correlation_vector = np.zeros(int((state_length - 1) * state_length / 2))
        for i in range(state_length):
            for j in range(i):
                idx = int(i * (i - 1) / 2 + j)
                if std_vector[i] == 0 or std_vector[j] == 0:
                    correlation_vector[idx] = 0
                else:
                    s = 0
                    for ii in range(len(list_states)):
                        s += list_counts[ii] * list_states[ii][i] * list_states[ii][j]
                    correlation_vector[idx] = s / total_counts
                    correlation_vector[idx] -= avg_photons[i] * avg_photons[j]
                    correlation_vector[idx] /= std_vector[i] * std_vector[j]

        return correlation_vector


class Covariances(Observable):
    """Observable corresponding to all 2-point output covariances.

    This is a non-rescaled version of the Correlations observable

    For n output modes, this observable returns n*(n-1)/2 values. The parameter
    shift rule does not work for this observable, but a small value of about pi/10
    (which is the default) works well in practice
    """

    descr = "covariances"

    def estimate(self, measurements: dict):
        """Returns tensor of all two-point covariances in output state given a dict of measurements.

        If the measured states are dimension n, then the output is of dimension n(n-1)/2

        Args:
            measurements: dict of state: counts

        Returns:
            1d tensor with values of the covariances
        """
        covs_numpy = self._covariances_numba(np.array(list(measurements.keys())), np.array(list(measurements.values())))
        return torch.tensor(covs_numpy)

    @staticmethod
    @nb.njit
    def _covariances_numba(list_states, list_counts):
        total_counts = np.sum(list_counts)
        state_length = len(list_states[0])

        avg_photons = _avg_photons_numba(list_states, list_counts)

        covariance_vector = np.zeros(int((state_length - 1) * state_length / 2))
        for i in range(state_length):
            for j in range(i):
                idx = int(i * (i - 1) / 2 + j)
                s = 0
                for ii in range(len(list_states)):
                    s += list_counts[ii] * list_states[ii][i] * list_states[ii][j]
                covariance_vector[idx] = s / total_counts
                covariance_vector[idx] -= avg_photons[i] * avg_photons[j]

        return covariance_vector


class AvgPhotons(Observable):
    """Observable corresponding to the average number of photons per mode.

    This is a simple observable that can be used for proof of concept testing.
    It can be estimated in linear time classically and its value is the same whether
    the photons interfere or not, but the parameter shift rule conveniently applies.

    For n output modes, this observable returns n output values.
    """

    descr = "avg-photons"

    def estimate(self, measurements: dict):
        """Returns tensor of average photons per mode.

        Args:
            measurements: dict of state: counts

        Returns:
            1d tensor with values of the average photon numbers
        """
        photons_per_mode = np.sum([np.asarray(state) * counts for state, counts in measurements.items()], axis=0)
        avg_photons_per_mode = torch.tensor(photons_per_mode) / sum(counts for counts in measurements.values())
        return avg_photons_per_mode


class SingleSample(Observable):
    """Directly returns output samples from the TBI.

    This is technically not an observable, and also one output
    sample is not a continuous function of the trainable
    parameters so this observable is not conducive to training.
    """

    descr = "single-sample"

    def estimate(self, measurements: dict):
        """Directly returns an output sample.

        The measurements dict must consist of single samples

        Args:
            measurements: dict of state: counts
        """
        output_list = list(list(measurements.keys())[0])
        return torch.tensor(output_list).to(torch.float32)


@nb.njit
def _avg_photons_numba(list_states, list_counts):
    total_counts = np.sum(list_counts)
    state_length = len(list_states[0])

    avg_photons = np.zeros(state_length)
    for i in range(len(list_states)):
        avg_photons += list_counts[i] * list_states[i]
    avg_photons /= total_counts

    return avg_photons


@nb.njit
def _std_photons_numba(list_states, list_counts, avg_photons):
    total_counts = np.sum(list_counts)
    state_length = len(list_states[0])

    std_photons = np.zeros(state_length)
    for i in range(len(list_states)):
        std_photons += list_counts[i] * (list_states[i] - avg_photons) ** 2
    std_photons /= total_counts
    std_photons = np.sqrt(std_photons)

    return std_photons
