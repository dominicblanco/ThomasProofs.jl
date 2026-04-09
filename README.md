# Proving the existence of localized patterns, periodic solutions, and branches of periodic solutions in the 1D Thomas model



Table of contents:


* [Introduction](#introduction)
* [The Thomas model](#the-thomas-model)
   * [Proving Localized Patterns](#proving-localized-patterns)
   * [Proving Periodic Solutions](#proving-periodic-solutions)
   * [Proving Branches of Solutions](#proving-branches-of-solutions)
* [Utilisation and References](#utilisation-and-references)
* [License and Citation](#license-and-citation)
* [Contact](#contact)



# Introduction

This Julia code is a complement to the article 

#### [[1]](To appear) : "Proving the existence of localized patterns, periodic solutions, and branches of periodic solutions in the 1D Thomas model, [ArXiv Link](To appear).

It provides the necessary rigorous computations of the bounds presented along the paper. The computations are performed using the package [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl). The mathematical objects (spaces, sequences, operators,...) are built using the package [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl). Detailed instructions are available in the code to help the user run it.


# The Thomas model

$$\partial_t u = \nu\Delta u + \nu_4 - u - \frac{\nu_1 uv}{1+u+\nu_2 u^2},$$

$$\partial_t v = \Delta v + \nu_3(\nu_5 - v) -  \frac{\nu_1 uv}{1+u+\nu_2 u^2},$$

is known to have localized patterns and periodic solutions.

## Proving Localized Patterns 

The Thomas model has solutions which are believed to be localized. We perform the analysis to prove these patterns in Section 2 of [[1]](To appear). We provide the candidate solutions, which are given in the files Ubar_Th_2_j.jl for j = 14,15,16,17, and 18. These correspond to the approximate solution $\overline{u}$ in Section 2.3.2. The approximate solution is computed in the code using a Newton method.

Given these approximate solution, the file Thomas_Localized_Proof.jl provides the explicit computation of the bounds presented in Section 2.4. It also provides a value for $r_0$ where the proof is successful. 

## Proving Periodic Solutions

The Thomas model has solutions which are believed to be periodic as well. We wish to provide the theory needed to prove these solutions. We perform a similar, yet different analysis to prove these solutions in Section 2 of [[1]](To appear). We provide the candidate solutions, which are given in the files Ubar_Th_3_j.jl for j = 7,8, and 9. These correspond to the approximate solution $\overline{U}$ in Section 3.1. The approximate solution is computed in the code using a Newton method.

Given these approximate solution, the file Thomas_Periodic_Proof.jl provides the explicit computation of the bounds presented in Section 3.2. It also provides a value for $r_0$ where the proof is successful. 

## Proving Branches of Solutions

We then demonstrate our approach for proving branches of solutions in Section 4 of [[1]](To appear). We provide a starting solution and the code needed to compute the branch of solution we prove. It is given in the file Ubar_Thomas_Branch_start.jl. After performing some computations in the code, we have an approximate solution $\overline{W}(s)$ as defined in Section 4. The approximate solution is computed in the code using a Newton method.

Given the starting point and the computations done in the code, the file Thomas_Branch_Proof.jl provides the explicit computation of the bounds presented in Section 4.1. It also provides a value for $r_0$ where the proof is successful. 
 
 # Utilisation and References

 The codes in Thomas_Proofs.jl can serve to prove other localized patterns and periodic solutions/branches other than the ones provided as illustration should one have the numerical candidates. 
 
 The code is build using the following packages :
 - [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl) 
 - [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl)
 - [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
 - [JLD2](https://github.com/JuliaIO/JLD2.jl)
 
 
 # License and Citation
 
This code is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).
  
If you wish to use this code in your publication, research, teaching, or other activities, please cite it using the following BibTeX template:

```
@software{Thomas_Proofs.jl,
  author = {Dominic Blanco},
  title  = {Thomas_Proofs.jl},
  url    = {https://github.com/dominicblanco/Thomas_Proofs.jl},
  note = {\url{ https://github.com/dominicblanco/Thomas_Proofs.jl},
  year   = {2026},
  doi = {10.5281/zenodo.}
}}
```
DOI : [10.5281/zenodo.](https://doi.org/10.5281/zenodo.) 


# Contact

You can contact me at :

dominic.blanco@mail.mcgill.ca
