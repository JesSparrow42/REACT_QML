# Machine learning algorithms for the PT Series

The `ptseries.algorithms` module contains examples of hybrid machine learning algorithms that use the PT Series.
Further algorithms will be added in future versions of this software development kit.

## Variational Classification

ORCA has developed a hybrid quantum/classical classification algorithm that combines neural networks with the PT Series.
A classical neural network can be used to encode higher-dimensional data such as images into the parameters of the PT
Series, or to transform the expectation values of output observables into some other quantity used in the objective
function.

## Generative Adversarial Networks

ORCA has developed a hybrid quantum/classical generative adversarial network (GAN) that uses the PT Series to generate
the probability distribution that is transformed by a GAN into new data.  Specifically, the algorithm implemented here
is a WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty), which is known to yield particularly
good performance.

## Binary Optimization Problems

ORCA has developed a hybrid quantum/classical algorithm for solving binary optimization problems where the PT series is
trained to produce photonic states which map to binary strings that minimize a cost function which defines the binary
optimization problem.  This algorithm can solve standard binary optimization formulations such as the quadratic
unconstrained binary optimization (QUBO) problem but crucially it can handle these problems in a more general and
efficient way.
