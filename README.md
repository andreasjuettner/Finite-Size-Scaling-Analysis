# Finite-Size-Scaling-Analysis

## License: GPL v2

## Authors: 
 - Andreas Juettner    <juettner@soton.ac.uk>
 - Ben Kitching-Morley <bkm1n18@soton.ac.uk>

## Introduction:
These scripts were used for the data analysis underlying the paper
   "Nonperturbative infrared finiteness in super-renormalisable scalar quantum field theory"
   [https://arxiv.org/abs/2009.14768]
by: Guido Cossu, Luigi Del Debbio, Andreas Juttner, Ben Kitching-Morley, Joseph K. L. Lee, Antonin Portelli, Henrique Bergallo Rocha, and Kostas Skenderis

**Abstract:** We present a study of the IR behaviour of a three-dimensional super-renormalisable quantum field theory (QFT) consisting of a scalar field in the adjoint of SU(N) with a phi^4 interaction. A bare mass is required for the theory to be massless at the quantum level. In perturbation theory the critical mass is ambiguous due to infrared (IR) divergences and we indeed find that at two-loops in lattice perturbation theory the critical mass diverges logarithmically. It was conjectured long ago in [1, 2] that super-renormalisable theories are nonperturbatively IR finite, with the coupling constant playing the role of an IR regulator. Using a combination of Markov-Chain-Monte-Carlo simulations of the lattice-regularised theory, both frequentist and Bayesian data analysis, and considerations of a corresponding effective theory we gather evidence that this is indeed the case.

[1] Jackiw and S. Templeton, How Superrenormalizable Interactions Cure their Infrared Divergences, Phys. Rev. D23, 2291 (1981).
[2] T. Appelquist and R. D. Pisarski, High-Temperature Yang-Mills Theories and Three-Dimensional Quantum Chromodynamics, Phys. Rev. D23, 2305 (1981).


## Description
The suite of scripts here carry out the entire data-analysis underlying the paper. 
In particular:
- Construction of the reweighted Binder cumulant under boostrap resampling and the determination of its crossing
points with user-determined values (`Binderanalysis.py`)
- Frequentist finite-size-scaling analysis under bootsrap
- Bayesian finite-size-scaling analysis

## Usage:


## Required libraries:
- Python v3 [www.python.org]
- Scipy [www.scipy.org]
- other python standard libraries: `random`, `h5py`, `os`,`sys`, `parse`, `warnings`




