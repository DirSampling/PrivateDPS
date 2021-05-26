#  Differential Privacy of Dirichlet Posterior Sampling

Repository to perform a private Dirichlet posterior sampling, and to reproduce the results in the following paper.

> Anonymous. Differential Privacy of Dirichlet Posterior Sampling. 2021.

Run the script `plots.py` to generate all three plots in the paper. The code is implemented in Python 3.  
The scripts were tested using Numpy 1.19.5, Scipy 1.4.1 and Matplotlib 3.3.4.

## Differential Privacy

The file `utils.py` contains many auxiliary functions for compting the privacy guarantees stated in the paper. 
* `alpha2rho(min_alphas, gamma, Delta_2sq)` computes `rho` from the minimum of `alpha`'s.
* `rho2alpha(rho, gamma, Delta_2sq)` computes the minimum of `alpha`'s, given a value of `rho`.
* `tcdp2adp(delta, alpha, Delta_2sq, Delta_inf)` converts from tCDP to approximate DP at given `alpha` and `delta`.

In all of the functions above, `Delta_2sq` is the squared l2-sensitivity and `Delta_inf` is the l^{infty}-sensitivity of the sample statistics.


## Private sampling

The file `PrivDPS.py` provides the `DirichletPosteriorSampling` class that tracks the privacy parameters and sample from a Dirichlet posterior. The class `GaussianMechanism` is also provided to reproduce the results in the paper.


## Disclaimer

This is research code.

## Licence

[Apache](https://www.apache.org/licenses/LICENSE-2.0)
