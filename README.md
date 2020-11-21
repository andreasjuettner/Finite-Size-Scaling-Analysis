# Finite-Size-Scaling-Analysis

## License: GPL v2

## Authors: 
 - Andreas Juettner    <juettner@soton.ac.uk>
 - Ben Kitching-Morley <bkm1n18@soton.ac.uk>

## Introduction:
These scripts were used for the data analysis underlying the paper

   *"Nonperturbative infrared finiteness in super-renormalisable scalar quantum field theory"*

   [https://arxiv.org/abs/2009.14768]

by: Guido Cossu, Luigi Del Debbio, Andreas Juttner, Ben Kitching-Morley, Joseph K. L. Lee, Antonin Portelli, Henrique Bergallo Rocha, and Kostas Skenderis

**Abstract:** We present a study of the IR behaviour of a three-dimensional super-renormalisable quantum field theory (QFT) consisting of a scalar field in the adjoint of SU(N) with a phi^4 interaction. A bare mass is required for the theory to be massless at the quantum level. In perturbation theory the critical mass is ambiguous due to infrared (IR) divergences and we indeed find that at two-loops in lattice perturbation theory the critical mass diverges logarithmically. It was conjectured long ago in [1, 2] that super-renormalisable theories are nonperturbatively IR finite, with the coupling constant playing the role of an IR regulator. Using a combination of Markov-Chain-Monte-Carlo simulations of the lattice-regularised theory, both frequentist and Bayesian data analysis, and considerations of a corresponding effective theory we gather evidence that this is indeed the case.

[1] Jackiw and S. Templeton, How Superrenormalizable Interactions Cure their Infrared Divergences, Phys. Rev. D23, 2291 (1981).
[2] T. Appelquist and R. D. Pisarski, High-Temperature Yang-Mills Theories and Three-Dimensional Quantum Chromodynamics, Phys. Rev. D23, 2305 (1981).


## Description
The suite of scripts here carry out the entire data-analysis and produces the results provided in the paper.

In particular:
- Construction of the reweighted Binder cumulant under boostrap resampling and the determination of its crossing
points with user-determined values (`Binderanalysis.py`)
- Frequentist finite-size-scaling analysis under bootsrap
- Bayesian finite-size-scaling analysis

## Usage
### Binderanalysis
`python3 Binderanalysis.py <N> <ag> <Bbar> <L/a>`
where `N` as in `SU(N)`, `ag` the gauge coupling, `Bbar` the desired crossing point, `L/a` the desired lattice size.

The required MCMC data can be donwloaded from zenodo.org under DOI 10.5281/zenodo.4266114 and needs to be copied into a local subdirectory
called `h5data/`. The data file contains data for

 - `N`=2,4
 - `ag`=0.1,0.2,0.3,0.5,0.6
 - `L/a`=8,16,32,48,64,96,128

The shell script `do_Binderanalysis.sh` loops over all available data. Note the `&` in the shell script which will trigger many instances of the code running in parallel. This might need to be adapted for smaller compute nodes.

The output will be stored in `h5data/Bindercrossings.h5` and is required by the following codes.

### Reproducing Plot data
`python3 -i publication_results.py`

This script contains 4 functions which produce 4 distint results from the paper. All 4 functions readin data from `h5data/Bindercrossings.h5`. All functions take a single argument of `N` as in `SU(N)`. The functions are as follows:

   - `get_pvalues_central_fit(N)`: This function returns the p-values used in figure 4. (e.g. the p-values of the proposed fit anzatz with the central fit across a range of gL_<sub>min</sub> values)
   - `get_statistical_errors_central_fit(N, model_name="model1")`: This
   function returns the central values and statistical errors on the model parameters under the bootstrap in the central fit. The variable "model_name"
   determines whether the central fit of the finite anzatz (model1) or the
   IR divergent anzatz (model2) is investigated.
   - `get_systematic_errors(N, model_name="model1")`: This function returns the central values of the model parameters in the central fit and the systematic error found by considering all fits with more than 15 degrees of freedom and
   a p-value between 0.05 and 0.95. The variable "model_name" determines
   whether the fits of the finite anzatz (model1) or the IR divergent anzatz (model2) is investigated.
   - `get_Bayes_factors(N, points=1000)`: This function returns log_<sub>10</sub> of the Bayes factors used in figure 4. (e.g. the Bayes factors of the proposed fit anzatz with the central fit across a range of gL_<sub>min</sub> values). The input parameter "points" describes the number of live points used in the MULTINEST algorithm. The higher this number, the higher the accuracy in the result and the higher the computational demand. To produce the plot of the Bayes factor against gL_min 5000 points were used. For the posterior plots 1000 points were used.

The exact form of the function return values is detailed in the function doc-strings. After running `python3 -i publication_results.py`, simply call the function you wish to run with the N value you wish to use. To change the specifics, e.g. to try a non-central fit, simply edit the relevent parameters, which are defined at the start of each function.

## Required libraries
- Python v3 [www.python.org]
- Scipy [www.scipy.org]
- h5py [www.h5py.org]
- tqdm [tqdm.github.io]
- Parse [pypi.org/project/parse]
- other python standard libraries: `random`, `os`,`sys`, `warnings`




