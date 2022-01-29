#  Differential Privacy of Dirichlet Posterior Sampling

Repository to perform a private Dirichlet posterior sampling, and to reproduce the results in the following paper.

> Anonymous. Differential Privacy of Dirichlet Posterior Sampling. 2022.

Run the script `plots.py` to generate Figure 1-4 in the paper. The code is implemented in Python 3.  

The scripts were tested using Numpy 1.19.5, Scipy 1.4.1 and Matplotlib 3.3.4.

## Differential Privacy

The file `utils.py` contains many auxiliary functions for computing the privacy guarantees stated in the paper. 
* `alpha2epsilon(min_alphas, gamma, Delta_2sq, Delta_inf)` computes `epsilon` from the minimum of `alpha`'s.
* `epsilon2alpha(epsilon, gamma, Delta_2sq, Delta_inf)` computes the minimum of `alpha`'s, given a value of `epsilon`.
* `alpha2adp(epsilon, alpha, Delta_2sq, Delta_inf)` computes `delta` from the minimum of `alpha`'s, given a value of `epsilon`.

In all of the functions above, `Delta_2sq` is the squared l2-sensitivity and `Delta_inf` is the l^{infty}-sensitivity of the sample statistics.


## Private sampling

The file `PrivDPS.py` provides the `DirichletPosteriorSampling` class that tracks the privacy parameters and sample from a Dirichlet posterior. The class `GaussianMechanism` and `LaplaceMechanism` are also provided to reproduce the results in the paper.


## Disclaimer

This is research code.

## Licence

[Apache](https://www.apache.org/licenses/LICENSE-2.0)
