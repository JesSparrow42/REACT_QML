# PyTorch nn.Module extensions for the PT Series

The `ptseries.models` module contains PyTorch nn.Module extensions allowing users to include a simulated PT Series device into any PyTorch-based machine learning workflow.

## PTLayer extension

The PTLayer nn.Module extension allows users to use the PT Series as a swap-in replacement for a classical neural
network layer. Upon instantiating a PTLayer, the user must define the input state and other parameters such as the
number of loops.

As a nn.Module PyTorch extension, the PTLayer defines both a forward function for inference and a custom backpropagation
function for learning. In the forward pass, inputs and learnable parameters are encoded into the beam splitter coupling
ratios, output samples are collected by simulating the device, and these samples are then used to calculate the
expectation values of a user-specified observable. These expectation values can then be sent to subsequent layers. In
the backwards pass, gradients are calculated using finite difference methods such as the parameter shift rule.

Important arguments when instantiating the PTLayer are:

- `input_state`: the input quantum state for the device, for example (1,1,1) for one photon in each of the first three
  modes
- `in_features` (optional): the number of features for the input data to the PTLayer, which must be smaller than or
  equal to the total number of variable parameters of the PTLayer. Any remaining parameters are the PTLayer's internal
  trainable parameters. Defaults to the total number of variable parameters.
- `observable` (optional): this defines the output of the PT Series, and is some function of the set of output samples.
  Defaults to 2-point correlators between the outputs, so the output dimension is `n_outputs*(n_outputs-1)/2`.
- `gradient_mode` (optional): method to compute the gradient. Default is "parameter-shift". Options include:

  - "parameter-shift": parameter shift rule method
  - "finite-difference": finite difference method
  - "spsa": simultaneous perturbation stochastic approximation. Because this method also requires to adapt the learning
    rate and the delta as we iterate, it must be used with the ORCA optimiser HybridOptimizer.

- `gradient_delta` (optional): delta to use with the parameter shift rule or for the finite difference.

- `n_samples` (optional): this is the number of samples used to measure the expectation value of the observable.
  Defaults to 100.
- `tbi_params` (optional): a dictionary containing the parameters that define the underlying PT Series device, for
  example `n_loops: 2`. Defaults to the default parameters of the `create_tbi` function in `ptseries.tbi`
- `n_tiling` (optional): uses n_tiling instances of PT Series and concatenates the results.Input features are
  distributed between tiles, which each have different trainable params. Default is 1.

The number of variable parameters in the PTLayer is the size of the input state, minus one, multiplied by the number of
loops. For example, with input state |1110> and 3 loops, there are 9 variable parameters. With the default observable
which is 2-point correlators, there are 6 possible combinations of 4 output modes so this PTLayer has 6 outputs.

## PTGenerator extension

The PTLayer nn.Module extension allows users to use the PT Series to generate complex probability distributions that can
then be fed into downstream models such as classical neural networks for algorithms such as GANs. Upon instantiating a
PTLayer, the user must define the input state and other parameters such as the number of loops.

This extension does not have a forward method. Instead, it has a generate method that produces samples from the PT
Series using the current values of its beam splitter parameters.

Important arguments when instantiating the PTLayer are:

- `input_state`: the input quantum state for the device, for example (1,1,1) for one photon in each of the first three
  modes
- `tbi_params` (optional): a dictionary containing the parameters that define the underlying PT Series device, for
  example `n_loops: 2`. Defaults to the default parameters of the `create_tbi` function in `ptseries.tbi`
