# Back-ends for the PT Series Time Bin Interferometer

The `ptseries.tbi` module contains both contains both a programmatic interface to the quantum processor and state of the
art simulators for ORCA's PT Series quantum processor. Simulators or the hardware interface are typically instantiated
with the `create_tbi` function located in `tbi.py` . Parameters for the create_tbi function are:

- `tbi_type`: this can be either "multi-loop", "single-loop", "fixed-random-unitary", or "PT-1"
- `n_loops`: number of loops for the simulator
- `bs_loss`: simulated loss at the beam splitters, as a fraction of the light lost
- `input_loss`: simulated loss at the input to the PT Series, as a fraction of the light lost
- `bs_noise`: amplitude of the simulated relative noise in each beam splitter angle
- `detector_efficiency`: simulated photon detection efficiency
- `distinguishable`: if set to True, then simulated photons are fully distinguishable and do not interfere

## Using a real quantum hardware back-end

The PT-1 can be instantiated using `create_tbi(tbi_type="PT-1")` . Further usage instructions can be found in the
documentation online and in the user manual.

## Simulators

The different types of simulators currently available are:

### Multi-loop sampler

This simulator in `tbi_multi_loop.py` uses the method from [this paper](https://arxiv.org/abs/2005.04214) to sample from
a multi-loop PT Series by calculating the permanents of the matrices that describe the interference between photons.
The time taken to run this simulator scales exponentially with the number of photons and we recommend not exceeding 10
photons.

### Single-loop sampler

This simulator in `tbi_single_loop.py` implements an efficient sampling algorithm for single-loop PT Series. Since a
single-loop PT Series can be efficiently simulated with very little overhead, this simulator can be used with thousands
of input modes and photons. This is helpful for performing extrapolation studies to a large number of photons, athough
for most applications results are improved by having more loops.

### Fixed random unitary

The simulator in `fixed_random_unitary.py` implements boson sampling with a Haar-random unitary matrix. In practice, an
NxN matrix can be implemented with a PT Series with N inputs and N loops. However, in this back-end the matrix is
selected once at random the first time a sample is requested and is then fixed. The absence of tunable parameters makes
this back-end incompatible with most machine learning applications in this SDK.
