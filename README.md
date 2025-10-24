<!-- Title -->
<h1 align="center">
  Breeze.jl
</h1>

<!-- description -->
<p align="center">
  <strong>⛈️ Fast and friendly Julia software for atmosphere simulations on CPUs and GPUs. https://numericalearth.github.io/Breeze.jl/stable/</strong>
</p>

<p align="center">
    <a href="https://numericalearth.github.io/Breeze.jl/dev/">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-in%20development-orange?style=flat-square">
    </a>
    <a href="https://github.com/NumericalEarth/Breeze.jl/discussions">
    <img alt="Ask us anything" src="https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square">
  </a>
  <a href="https://github.com/SciML/ColPrac">
    <img alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet?style=flat-square">
  </a>
  <a href="https://codecov.io/gh/NumericalEarth/Breeze.jl" >
    <img src="https://codecov.io/gh/NumericalEarth/Breeze.jl/graph/badge.svg?token=09TZGWKUPV"/>
  </a>
</p>

Breeze is a software package for atmosphere simulations using finite volume methods on CPUs and GPUs.
Breeze currently supports simulations based on the Boussinesq and anelastic approximations in Cartesian domains.
Breeze's power flows from [Oceananigans](https://github.com/CliMA/Oceananigans.jl), which provides user interface design, grids, fields, solvers, advection schemes, and more.
Watch this space for the crystallization of Breeze's roadmap.

### Installing and using Breeze

First [install Julia](https://julialang.org/downloads/); suggested version 1.10. See [juliaup](https://github.com/JuliaLang/juliaup) README for how to install 1.10 and make that version the default.

Then clone this repository

```bash
git clone git@github.com:NumericalEarth/Breeze.jl.git
```

Open Julia from within the local directory of the repo via:

```bash
julia --project
```

The first time, you need to install any dependencies:

```julia
julia> using Pkg; Pkg.instantiate()
```

Now you are ready to run any of the examples!

For instance,

```julia
julia> include("examples/free_convection.jl")
```

produces

https://github.com/user-attachments/assets/dc45d188-6c61-4eb5-95fb-9a51c6f99013
