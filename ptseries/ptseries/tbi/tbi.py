from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ptseries.tbi.fixed_random_unitary import FixedRandomUnitary
    from ptseries.tbi.pt1 import PT1
    from ptseries.tbi.tbi_multi_loop import TBIMultiLoop
    from ptseries.tbi.tbi_single_loop import TBISingleLoop


def create_tbi(
    tbi_type: str | None = None,
    n_loops: int | None = None,
    loop_lengths: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | None = None,
    postselected: bool | None = None,
    distinguishable: bool = False,
    bs_loss: float = 0.0,
    bs_noise: float = 0.0,
    input_loss: float = 0.0,
    detector_efficiency: float = 1.0,
    n_signal_detectors: int = 0,
    g2: float = 0.0,
    ip_address: str | None = None,
    url: str | None = None,
    machine: str | None = None,
    **kwargs,
) -> TBISingleLoop | TBIMultiLoop | FixedRandomUnitary | PT1:
    """Returns an instance of a PT Series time bin interferometer (TBI).

    Args:
        tbi_type: Type of TBI to return. Can be 'multi-loop', 'single-loop', 'fixed-random-unitary' or
            'PT-1'.
        n_loops: Number of loops in the TBI.
        loop_lengths: Lengths of the loops in the PT Series.
            If loop_lengths is specified then n_loops is not required, but if both are specified then they
            should be consistent.
        postselected: If True, postselects on the input and output state with equal number of photons.
        distinguishable: If True, the photons do not interfere with each other.
        bs_loss: Fraction of light lost at each beam splitter in the TBI, between 0 and 1.
        bs_noise: Standard deviation of Gaussian noise on beam splitter angle. Can take any positive
            value.
        input_loss: Optical losses at the input of the PT Series, between 0 and 1.
        detector_efficiency: Detection efficiency of the detectors, from 0 to 1.
        n_signal_detectors: Number of threshold detectors at the signal. If set to 0, then we assume the
            detectors are fully photon number resolving
        g2: Heralded autocorrelation coefficient. The default value of 0 means an ideal signal photon.
        ip_address: Deprecated: The IP address of the PT-1 hardware, for example "0.0.0.0".
        url: The URL of the PT-1 hardware, for example "http://<orca_api_address>".
        machine: Machine name of device to be used. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        An instance of a PT Series time-bin interferometer.

    Raises:
        ValueError: If the TBI type is unrecognized.

    Note:
        - If `tbi_type` is not specified, the function selects the best type of simulator by default.
        - If `n_loops` is not specified, it defaults to 1 if `loop_lengths` is None, otherwise it is set to the length
          of `loop_lengths`.
        - If `loop_lengths` is provided, `n_loops` is set to its length.
        - Supports different types of TBIs: 'multi-loop', 'single-loop', 'fixed-random-unitary', and 'PT-1'.
        - For 'single-loop' TBIs, the function calls `_create_single_loop` internally.
        - For 'multi-loop' TBIs, the function calls `_create_multi_loop` internally.
        - For 'fixed-random-unitary' TBIs, the function calls `_create_fixed_random_unitary` internally.
        - For 'PT-1' TBIs, the function calls `_create_pt1` internally.

    Example:
        >>> tbi = create_tbi(tbi_type="multi-loop", n_loops=3, loop_lengths=[2, 3, 4])
    """
    n_loops = n_loops or (1 if loop_lengths is None else len(loop_lengths))

    if loop_lengths is not None:
        if n_loops != len(loop_lengths):
            warnings.warn("n_loops is inconsistent with loop_lengths. Defaulting to loop_lengths...")
        n_loops = len(loop_lengths)

    # We select the best type of simulator by default if none is specified
    tbi_type = tbi_type or ("single-loop" if n_loops == 1 else "multi-loop")

    postselected = postselected if postselected is not None else (tbi_type == "PT-1")

    if tbi_type == "single-loop":
        return _create_single_loop(
            n_loops,
            postselected,
            distinguishable,
            bs_loss,
            bs_noise,
            input_loss,
            detector_efficiency,
            n_signal_detectors,
            g2,
        )
    elif tbi_type == "multi-loop":
        return _create_multi_loop(
            n_loops,
            loop_lengths,
            postselected,
            distinguishable,
            input_loss,
            detector_efficiency,
            bs_loss,
            bs_noise,
            n_signal_detectors,
            g2,
        )
    elif tbi_type == "fixed-random-unitary":
        return _create_fixed_random_unitary(
            distinguishable,
            input_loss,
            detector_efficiency,
            bs_loss,
            bs_noise,
            n_signal_detectors,
            g2,
        )
    elif tbi_type == "PT-1":
        return _create_pt1(
            ip_address,
            url,
            machine,
            n_loops,
            loop_lengths,
            postselected,
            bs_loss,
            bs_noise,
            input_loss,
            detector_efficiency,
            distinguishable,
            n_signal_detectors,
            g2,
            **kwargs,
        )
    else:
        raise ValueError(
            f"TBI type {tbi_type} unrecognized! \
                Please select 'multi-loop', 'fixed-random-unitary', 'single-loop' \
                or 'PT-1'."
        )


def _create_single_loop(
    n_loops: int,
    postselected: bool,
    distinguishable: bool,
    bs_loss: float,
    bs_noise: float,
    input_loss: float,
    detector_efficiency: float,
    n_signal_detectors: int,
    g2: float,
) -> TBISingleLoop:
    """Create a single-loop TBI object.

    Args:
        n_loops: The number of loops. This should be 1 for a single-loop TBI.
        postselected: If True, postselects on the input and output state with equal number of photons.
        distinguishable: Whether the photons are distinguishable.
        bs_loss: The loss in the beam splitter.
        bs_noise: The noise in the beam splitter.
        input_loss: The loss in the input.
        detector_efficiency: The efficiency of the detector.
        n_signal_detectors: Number of threshold detectors at the signal. If 0, then default is full PNR.
        g2: Heralded autocorrelation coefficient. The default value of 0 means an ideal signal photon.

    Returns:
        TBISingleLoop: The created single-loop TBI object.

    Raises:
        Warning: If `n_loops` is not equal to 1, a warning is raised.
    """
    from ptseries.tbi.tbi_single_loop import TBISingleLoop

    if n_loops != 1:
        warnings.warn("TBI type 'single-loop' only has a single loop of length one. Attempting with a single loop...")

    return TBISingleLoop(
        postselected=postselected,
        distinguishable=distinguishable,
        bs_loss=bs_loss,
        bs_noise=bs_noise,
        input_loss=input_loss,
        detector_efficiency=detector_efficiency,
        n_signal_detectors=n_signal_detectors,
        g2=g2,
    )


def _create_multi_loop(
    n_loops: int,
    loop_lengths: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | None,
    postselected: bool,
    distinguishable: bool,
    input_loss: float,
    detector_efficiency: float,
    bs_loss: float,
    bs_noise: float,
    n_signal_detectors: int,
    g2: float,
) -> TBIMultiLoop:
    """Create a multi-loop TBI object.

    Args:
        n_loops: The number of loops in the TBI.
        loop_lengths: The lengths of the loops.
        postselected: If True, postselects on the input and output state with equal number of photons.
        distinguishable: Whether the photons in the loops are distinguishable or not.
        input_loss: The input loss of the TBI.
        detector_efficiency: The efficiency of the detectors.
        bs_loss: The loss of the beam splitter.
        bs_noise: The noise of the beam splitter. Not supported for this TBI type.
        n_signal_detectors: Number of threshold detectors at the signal. Not supported for this TBI type.
        g2: Heralded autocorrelation coefficient. Not supported for this TBI type.

    Returns:
        TBIMultiLoop: The created multi-loop TBI object.

    Raises:
        Warning: If `bs_noise` is not equal to 0, a warning is raised.
    """
    from ptseries.tbi.tbi_multi_loop import TBIMultiLoop

    if not np.isclose(bs_noise, 0.0):
        warnings.warn("TBI type 'multi-loop' does not support beam splitter noise. Attempting without noise...")

    if not n_signal_detectors == 0:
        warnings.warn("TBI type 'multi-loop' does not support partial photon number resolution. Attempting without...")

    if not np.isclose(g2, 0.0):
        warnings.warn("TBI type 'multi-loop' does not support non-zero g2 values. Attempting with g2 set to 0...")

    return TBIMultiLoop(
        n_loops=n_loops,
        loop_lengths=loop_lengths,
        postselected=postselected,
        distinguishable=distinguishable,
        input_loss=input_loss,
        detector_efficiency=detector_efficiency,
        bs_loss=bs_loss,
    )


def _create_fixed_random_unitary(
    distinguishable: bool,
    input_loss: float,
    detector_efficiency: float,
    bs_loss: float,
    bs_noise: float,
    n_signal_detectors: int,
    g2: float,
) -> FixedRandomUnitary:
    """Create a FixedRandomUnitary object.

    Args:
        distinguishable: Whether the particles are distinguishable or not.
        input_loss: The input loss of the system.
        detector_efficiency: The efficiency of the detectors.
        bs_loss: The beam splitter loss. Not supported for this TBI type.
        bs_noise: The beam splitter noise. Not supported for this TBI type.
        n_signal_detectors: Number of threshold detectors at the signal. Not supported for this TBI type.
        g2: Heralded autocorrelation coefficient. Not supported for this TBI type.

    Returns:
        FixedRandomUnitary: The created FixedRandomUnitary object.

    Raises:
        None

    Warnings:
        If `bs_loss` is not equal to 0, a warning is raised indicating that TBI type 'fixed-random-unitary' is not
            compatible with beam splitter loss.
        If `bs_noise` is not equal to 0, a warning is raised indicating that TBI type 'fixed-random-unitary' does not
            support beam splitter noise.
    """
    from ptseries.tbi.fixed_random_unitary import FixedRandomUnitary

    if not np.isclose(bs_loss, 0.0):
        warnings.warn(
            "TBI type 'fixed-random-unitary' is not compatible with beam splitter loss. Attempting without loss..."
        )
    if not np.isclose(bs_noise, 0.0):
        warnings.warn(
            "TBI type 'fixed-random-unitary' does not support beam splitter noise. Attempting without noise..."
        )

    if not n_signal_detectors == 0:
        warnings.warn(
            "TBI type 'fixed-random-unitary' does not support partial photon number resolution. Attempting without..."
        )

    if not np.isclose(g2, 0.0):
        warnings.warn(
            "TBI type 'fixed-random-unitary' does not support non-zero g2 values. Attempting with g2 set to 0..."
        )

    return FixedRandomUnitary(
        distinguishable=distinguishable,
        input_loss=input_loss,
        detector_efficiency=detector_efficiency,
    )


def _create_pt1(
    ip_address: str | None,
    url: str | None,
    machine: str | None,
    n_loops: int,
    loop_lengths: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | None,
    postselected: bool,
    bs_loss: float,
    bs_noise: float,
    input_loss: float,
    detector_efficiency: float,
    distinguishable: bool,
    n_signal_detectors: int,
    g2: float,
    **kwargs,
) -> PT1:
    """Create a PT1 object.

    Args:
        ip_address: The IP address of the PT-1 device.
        url: The URL of the PT-1 device.
        machine: The machine name of the PT-1 device.
        n_loops: The number of loops.
        loop_lengths: The lengths of the loops.
        postselected: Whether postselection is enabled.
        bs_loss: The beam splitter loss. Must be 0.0 for TBI type 'PT-1'.
        bs_noise: The beam splitter noise. Must be 0.0 for TBI type 'PT-1'.
        input_loss: The input loss. Must be 0.0 for TBI type 'PT-1'.
        detector_efficiency: The detector efficiency. Must be 1.0 for TBI type 'PT-1'.
        distinguishable: Whether the photons are distinguishable. Must be False for TBI type 'PT-1'.
        n_signal_detectors: Number of threshold detectors at the signal. Must be 0 for TBI type 'PT-1'.
        g2: Heralded autocorrelation coefficient. Must be 0 for TBI type 'PT-1'.
        **kwargs: Additional keyword arguments.

    Returns:
        PT1: The created PT1 object.

    Raises:
        None

    Warnings:
        If `bs_loss` is not equal to 0, a warning is raised indicating that TBI type 'PT-1' does not support beam
            splitter loss.
        If `bs_noise` is not equal to 0, a warning is raised indicating that TBI type 'PT-1' does not support beam
            splitter noise.
        If `input_loss` is not equal to 0, a warning is raised indicating that TBI type 'PT-1' does not support input
            loss.
        If `detector_efficiency` is not equal to 1, a warning is raised indicating that TBI type 'PT-1' does not
            support detector efficiency.
        If `distinguishable` is True, a warning is raised indicating that TBI type 'PT-1' does not support
            distinguishable photons
    """
    from ptseries.tbi.pt1 import PT1

    if not np.isclose(bs_loss, 0.0):
        warnings.warn("bs_loss cannot be set for TBI type 'PT-1'")

    if not np.isclose(bs_noise, 0.0):
        warnings.warn("bs_noise cannot be set for TBI type 'PT-1'")

    if not np.isclose(input_loss, 0.0):
        warnings.warn("input_loss cannot be set for TBI type 'PT-1'")

    if not np.isclose(detector_efficiency, 1.0):
        warnings.warn("detector_efficiency cannot be set for TBI type 'PT-1'")

    if distinguishable:
        warnings.warn("distinguishable cannot be set to True for TBI type 'PT-1'")

    if not n_signal_detectors == 0:
        warnings.warn("n_signal_detectors cannot be set for TBI type 'PT-1'")

    if not np.isclose(g2, 0.0):
        warnings.warn("g2 cannot be set for TBI type 'PT-1'")

    return PT1(
        ip_address=ip_address,
        url=url,
        machine=machine,
        n_loops=n_loops,
        loop_lengths=loop_lengths,
        postselected=postselected,
        **kwargs,
    )
