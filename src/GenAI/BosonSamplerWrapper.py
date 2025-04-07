import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# Boson Sampling Simulation Functions
# =============================================================================

def generate_unitary(m, bs_loss, bs_jitter, phase_noise_std=0.0, systematic_phase_offset=0.0, mode_loss=None):
    """
    Generate a modified random unitary matrix for boson sampling simulations.

    This function creates a random complex matrix, performs QR decomposition
    to obtain a unitary matrix, and then applies various modifications such as loss,
    jitter, phase noise, and a systematic phase offset. Optionally, mode-dependent losses
    can be applied.

    Parameters:
        m (int): The dimension of the unitary matrix.
        bs_loss (float): Overall loss factor applied to the matrix.
        bs_jitter (float): Jitter added to the matrix elements.
        phase_noise_std (float, optional): Standard deviation for phase noise. Defaults to 0.0.
        systematic_phase_offset (float, optional): Systematic phase offset to apply. Defaults to 0.0.
        mode_loss (array-like, optional): Array of loss factors for each mode. Defaults to None.

    Returns:
        np.ndarray: The modified unitary matrix of shape (m, m).
    """
    random_matrix = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    Q, _ = np.linalg.qr(random_matrix)
    Q *= np.sqrt(bs_loss)
    Q += bs_jitter * np.random.randn(m, m)
    if phase_noise_std > 0:
        phase_noise = np.exp(1j * np.random.normal(0, phase_noise_std, size=(m, m)))
        Q *= phase_noise
    if systematic_phase_offset != 0:
        Q *= np.exp(1j * systematic_phase_offset)
    if mode_loss is not None:
        Q = np.array([Q[i, :] * np.sqrt(mode_loss[i]) for i in range(m)])
    return Q


def compute_permanent(A: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the permanent of a square matrix A (dtype should be torch.cfloat or torch.cdouble)
    by brute-force enumeration of permutations.

    Parameters:
        A (torch.Tensor): A 2D tensor (n x n) with a complex dtype.

    Returns:
        torch.Tensor: A single complex scalar (the permanent).
    """
    # Number of modes
    n = A.shape[0]

    # We'll accumulate the permanent in perm_sum
    perm_sum = torch.zeros([], dtype=A.dtype, device=A.device)

    # For each permutation sigma of {0,1,...,n-1}, multiply together A[i, sigma[i]] for i in [0..n-1].
    for sigma in itertools.permutations(range(n)):
        product_term = torch.tensor(1.0, dtype=A.dtype, device=A.device)
        for i in range(n):
            product_term = product_term * A[i, sigma[i]]
        perm_sum = perm_sum + product_term

    return perm_sum

def get_submatrix(U, input_modes, output_modes):
    """
    Extract a submatrix from a given matrix U.

    This function returns the submatrix defined by selecting the rows corresponding
    to input_modes and the columns corresponding to output_modes.

    Parameters:
        U (np.ndarray): The original matrix.
        input_modes (array-like): Indices for the rows.
        output_modes (array-like): Indices for the columns.

    Returns:
        np.ndarray: The extracted submatrix.
    """
    return U[np.ix_(input_modes, output_modes)]

def kerr_phase_factor(output_modes, m, chi):
    """
    Compute a Kerr phase factor based on the photon counts in output modes.
    
    Each mode contributes a factor exp(-1j * chi * n * (n-1)),
    where n is the photon number in that mode.
    """
    counts = np.bincount(output_modes, minlength=m)
    factors = [np.exp(-1j * chi * n * (n - 1)) for n in counts]
    return factors

def boson_sampling_probability(U, input_modes, output_modes, effective_mu, m):
    if len(input_modes) == 0 or len(output_modes) == 0:
        return 0
    U_sub = get_submatrix(U, input_modes, output_modes)

    A_sub_torch = torch.tensor(U_sub, dtype=torch.complex64)
    permanent_val = compute_permanent(A_sub_torch)

    classical_probability = np.prod(np.abs(U_sub)**2)

    # Convert permanent to numpy complex
    permanent_val_np = permanent_val.detach().cpu().numpy()  # shape (), complex

    # Compute probs
    probability = effective_mu * np.abs(permanent_val_np)**2 + (1 - effective_mu) * classical_probability

    return probability

def boson_sampling_probability_advanced(U, input_modes, output_modes, effective_mu, m, chi):
    if len(input_modes) == 0 or len(output_modes) == 0:
        return 0
    U_sub = get_submatrix(U, input_modes, output_modes)

    A_sub_torch = torch.tensor(U_sub, dtype=torch.complex64)
    permanent_val = compute_permanent(A_sub_torch)

    classical_probability = np.prod(np.abs(U_sub)**2)

    # Kerr Phase Factors
    nl_factor_list = kerr_phase_factor(output_modes, m, chi)  
    nl_factor_val = np.prod(nl_factor_list)  # single scalar

    # Convert permanent to numpy complex
    permanent_val_np = permanent_val.detach().cpu().numpy()  # shape (), complex

    # Compute probs
    combined_val = nl_factor_val * permanent_val_np
    probability = effective_mu * np.abs(combined_val)**2 + (1 - effective_mu) * classical_probability

    return probability

def sample_input_state(m, num_sources, input_loss, coupling_efficiency, detector_inefficiency, multi_photon_prob):
    """
    Simulate the sampling of input states for a boson sampling experiment.

    Each source is considered for photon emission based on a combined efficiency parameter.
    If a photon is emitted, a mode is randomly selected (with possible multi-photon events).

    Parameters:
        m (int): Total number of modes.
        num_sources (int): Number of photon sources.
        input_loss (float): Loss factor at the input.
        coupling_efficiency (float): Efficiency of photon coupling.
        detector_inefficiency (float): Detector inefficiency factor.
        multi_photon_prob (float): Probability of multi-photon emission from a source.

    Returns:
        tuple: A tuple (interfering_input_modes, total_photons) where:
            interfering_input_modes (np.ndarray): Array of mode indices where photons are present.
            total_photons (int): Total number of photons emitted.
    """
    interfering_input_modes = []
    total_photons = 0
    p_eff = input_loss * coupling_efficiency * detector_inefficiency
    for _ in range(num_sources):
        if np.random.rand() < p_eff:
            n_emit = 2 if np.random.rand() < multi_photon_prob else 1
            total_photons += n_emit
            modes = np.random.choice(m, n_emit, replace=True)
            interfering_input_modes.extend(modes.tolist())
    interfering_input_modes = np.array(interfering_input_modes)
    return interfering_input_modes, total_photons

def apply_dark_counts(probability, m, dark_count_rate):
    """
    Apply a dark count correction to a computed probability.

    The correction is simply an additive term proportional to the dark count rate and the number of modes.

    Parameters:
        probability (float): The original computed probability.
        m (int): Total number of modes.
        dark_count_rate (float): The dark count rate per mode.

    Returns:
        float: The adjusted probability.
    """
    dark_counts = dark_count_rate * m
    return probability + dark_counts

def boson_sampling_simulation(
    m, num_sources, num_loops,
    input_loss, coupling_efficiency, detector_inefficiency, multi_photon_prob,
    mu, temporal_mismatch, spectral_mismatch, arrival_time_jitter,
    bs_loss, bs_jitter, phase_noise_std, systematic_phase_offset, mode_loss,
    dark_count_rate,
    use_advanced_nonlinearity=False, chi=0
):
    """
    Run multiple boson sampling simulation loops and return the average transition probability.

    For each loop, the function:
      1. Generates a unitary matrix with the specified losses, jitter, and noise.
      2. Samples an input state.
      3. Randomly selects output modes.
      4. Computes the transition probability using either the standard or an advanced nonlinearity model.
      5. Applies a dark count correction.

    Parameters:
        m (int): Total number of modes.
        num_sources (int): Number of photon sources.
        num_loops (int): Number of simulation loops.
        input_loss (float): Input loss factor.
        coupling_efficiency (float): Photon coupling efficiency.
        detector_inefficiency (float): Detector inefficiency.
        multi_photon_prob (float): Probability of multi-photon emission.
        mu (float): Weighting parameter between quantum and classical contributions.
        temporal_mismatch (float): Factor representing temporal mismatches.
        spectral_mismatch (float): Factor representing spectral mismatches.
        arrival_time_jitter (float): Factor for photon arrival time jitter.
        bs_loss (float): Loss factor for the boson sampler unitary.
        bs_jitter (float): Jitter factor for the boson sampler unitary.
        phase_noise_std (float): Standard deviation for phase noise.
        systematic_phase_offset (float): Systematic phase offset to apply.
        mode_loss (np.ndarray): Mode-dependent loss factors.
        dark_count_rate (float): Dark count rate per mode.
        use_advanced_nonlinearity (bool, optional): If True, use the advanced nonlinearity model. Defaults to False.
        chi (float): Entanglement strength. Default 0.0

    Returns:
        float: The average transition probability over all simulation loops.
    """
    probabilities = []
    effective_mu = mu * (1 - temporal_mismatch) * (1 - spectral_mismatch) * (1 - arrival_time_jitter)
    for _ in range(num_loops):
        U = generate_unitary(m, bs_loss, bs_jitter, phase_noise_std, systematic_phase_offset, mode_loss)
        input_modes, interfering_count = sample_input_state(
            m, num_sources, input_loss, coupling_efficiency, detector_inefficiency, multi_photon_prob
        )
        if interfering_count == 0:
            probabilities.append(0)
            continue
        output_modes = np.random.choice(m, interfering_count, replace=True)
        if use_advanced_nonlinearity:
            probability = boson_sampling_probability_advanced(U, input_modes, output_modes,
                                                               effective_mu, m, chi)
        else:
            probability = boson_sampling_probability(U, input_modes, output_modes, effective_mu)
        probability = apply_dark_counts(probability, m, dark_count_rate)
        probabilities.append(probability)
    return np.mean(probabilities)

# =============================================================================
# PyTorch Module: Wrap Boson Sampling Simulation
# =============================================================================

class BosonSamplerTorch(nn.Module):
    r"""A PyTorch module that wraps the boson sampling simulation.

    This module runs the boson sampling simulation (using NumPy operations) and returns a scalar probability,
    replicated over a batch. The systematic phase offset is a trainable parameter.

    Attributes:
        m (int): Total number of modes.
        num_sources (int): Number of photon sources.
        num_loops (int): Number of simulation loops.
        input_loss (float): Input loss factor.
        coupling_efficiency (float): Coupling efficiency.
        detector_inefficiency (float): Detector inefficiency.
        multi_photon_prob (float): Probability of multi-photon emission.
        mu (float): Weighting parameter for quantum versus classical contributions.
        temporal_mismatch (float): Temporal mismatch factor.
        spectral_mismatch (float): Spectral mismatch factor.
        arrival_time_jitter (float): Photon arrival time jitter factor.
        bs_loss (float): Loss factor for the boson sampler unitary.
        bs_jitter (float): Jitter factor for the boson sampler unitary.
        phase_noise_std (float): Standard deviation for phase noise.
        systematic_phase_offset (float): Systematic phase offset (trainable).
        mode_loss (np.ndarray): Array of mode-dependent loss factors.
        dark_count_rate (float): Dark count rate per mode.
        use_advanced_nonlinearity (bool): Flag to use the advanced nonlinearity model.
        chi (float): Entanglement Strength. Default 0.0
    """
    def __init__(self,
                 m: int,
                 num_sources: int,
                 num_loops: int,
                 input_loss: float,
                 coupling_efficiency: float,
                 detector_inefficiency: float,
                 multi_photon_prob: float,
                 mu: float,
                 temporal_mismatch: float,
                 spectral_mismatch: float,
                 arrival_time_jitter: float,
                 bs_loss: float,
                 bs_jitter: float,
                 phase_noise_std: float,
                 systematic_phase_offset: float,
                 mode_loss: np.ndarray,
                 dark_count_rate: float,
                 use_advanced_nonlinearity: bool = False,
                 chi: float = 0.0):
        super(BosonSamplerTorch, self).__init__()
        self.m = m
        self.num_sources = num_sources
        self.num_loops = num_loops
        self.input_loss = input_loss
        self.coupling_efficiency = coupling_efficiency
        self.detector_inefficiency = detector_inefficiency
        self.multi_photon_prob = multi_photon_prob
        self.mu = mu
        self.temporal_mismatch = temporal_mismatch
        self.spectral_mismatch = spectral_mismatch
        self.arrival_time_jitter = arrival_time_jitter
        self.bs_loss = bs_loss
        self.bs_jitter = bs_jitter
        self.phase_noise_std = phase_noise_std
        # Make the systematic phase offset trainable.
        self.systematic_phase_offset = nn.Parameter(torch.tensor(systematic_phase_offset, dtype=torch.float32))
        self.mode_loss = mode_loss
        self.dark_count_rate = dark_count_rate
        self.use_advanced_nonlinearity = use_advanced_nonlinearity
        self.chi = chi

    def forward(self, batch_size: int):
        """
        Run the boson sampling simulation and return a tensor of probabilities replicated over the batch.

        Parameters:
            batch_size (int): The number of samples in the batch.

        Returns:
            torch.Tensor: A tensor of shape (batch_size,) where each element is the simulated probability.
        """
        # Use the current (trainable) phase offset.
        phase_offset = self.systematic_phase_offset.detach().cpu().numpy().item()
        avg_probability = boson_sampling_simulation(
            self.m, self.num_sources, self.num_loops,
            self.input_loss, self.coupling_efficiency, self.detector_inefficiency, self.multi_photon_prob,
            self.mu, self.temporal_mismatch, self.spectral_mismatch, self.arrival_time_jitter,
            self.bs_loss, self.bs_jitter, self.phase_noise_std, phase_offset, self.mode_loss,
            self.dark_count_rate,
            self.use_advanced_nonlinearity, self.chi
        )
        # Return the scalar probability as a tensor replicated over the batch dimension.
        result = torch.tensor(avg_probability, dtype=torch.float32, device=self.systematic_phase_offset.device)
        return result.expand(batch_size)

# =============================================================================
# Boson Latent Generator: Wrap the boson sampler to produce a latent vector.
# =============================================================================

class BosonLatentGenerator(nn.Module):
    r"""A wrapper that converts the scalar output of BosonSamplerTorch into a latent vector.

    For demonstration, the scalar output is replicated to create a latent vector of a specified dimension.

    Attributes:
        latent_dim (int): Desired dimension of the latent vector.
        boson_sampler_module (BosonSamplerTorch): An instance of the boson sampler module.
    """
    def __init__(self, latent_dim: int, boson_sampler_module: BosonSamplerTorch):
        super(BosonLatentGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.boson_sampler_module = boson_sampler_module

    def forward(self, batch_size: int):
        """
        Convert the scalar output of the boson sampler into a latent vector.

        Parameters:
            batch_size (int): Number of latent vectors to generate.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, latent_dim) representing the latent vectors.
        """
        latent_scalar = self.boson_sampler_module(batch_size)  # shape: (batch_size,)
        latent_vector = latent_scalar.unsqueeze(1).repeat(1, self.latent_dim)
        return latent_vector