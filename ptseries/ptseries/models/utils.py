from ptseries.tbi import create_tbi


def calculate_n_params(input_state, tbi_params=None, n_tiling=1):
    r"""Calculates the number of variational parameters in a PTLayer or PTGenerator.

    Args:
        input_state (tuple): a tuple corresponding to an input state, for example :math:`(0,1,1)` is :math:`|0,1,1\rangle`.
        tbi_params (dict, optional): dict of optional parameters to instantiate the TBI.
        n_tiling (int, optional): number of tiles used to generate output samples.

    Returns:
        the number of variational parameters.
    """
    # We get the correct number of parameters by instantiating a TBI instance
    tbi = create_tbi(**tbi_params) if tbi_params is not None else create_tbi()

    n_modes = len(input_state)
    n_beam_splitters = tbi.calculate_n_beam_splitters(n_modes)

    return n_beam_splitters * n_tiling


def calculate_output_dim(input_state, tbi_params=None, observable=None, n_tiling=1):
    r"""Calculates the output dimension of a PTLayer.

    Args:
        input_state (tuple): a tuple corresponding to an input state, for example :math:`(0,1,1)` is :math:`|0,1,1\rangle`.
        tbi_params (dict, optional): dict of optional parameters to instantiate the TBI.
        observable (str or Observable instance, optional): function that converts measurements to a tensor.
        n_tiling (int, optional): number of tiles used to generate output samples.

    Returns:
        the output dimension of a PTLayer.
    """
    # Avoid a circular import
    from ptseries.models import PTLayer

    # Filter out the Nones so that we use the PTLayer defaults
    params = {"input_state": input_state, "tbi_params": tbi_params, "observable": observable, "n_tiling": n_tiling}
    not_none_params = {k: v for k, v in params.items() if v is not None}

    ptlayer_test = PTLayer(**not_none_params)
    return ptlayer_test.dim_observable
