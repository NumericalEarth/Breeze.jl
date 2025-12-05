<!-- Title -->
<h1 align="center">
  Breeze.jl
</h1>

<!-- description -->
<p align="center">
  <strong>🌪 Fast and friendly Julia software for atmospheric fluid dynamics on CPUs and GPUs. https://numericalearth.github.io/BreezeDocumentation/dev/</strong>
</p>

<p align="center">
    <a href="https://numericalearth.github.io/BreezeDocumentation/dev/">
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
  <a href="https://github.com/JuliaTesting/Aqua.jl" >
    <img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg"/>
  </a>
</p>

Breeze is a library for simulating atmospheric flows, convection, clouds, weather, and hurricanes on CPUs and GPUs.
Much of Breeze's power flows from [Oceananigans](https://github.com/CliMA/Oceananigans.jl), which provides a user interface, grids, fields, solvers, advection schemes, Lagrangian particles, physics, and more.
Right now, `Breeze.AtmosphereModel` is in an early stage of development, and supports simple simulations that use the anelastic formulation of the Euler equations on `RectilinearGrid`.
But we're working feverishly towards a future with bulk, bin and superdroplet microphysics, radiation, and a fully compressible formulation with acoustic substepping (and note, the roadmap and vision for Breeze is still something of a work in progress).
Check out [the documentation](https://numericalearth.github.io/BreezeDocumentation/dev/) to see what we can do now, and watch this space (or get in touch to discuss!) its crystallization.

### Installing and using Breeze

First [install Julia](https://julialang.org/install/); suggested version 1.12. See [juliaup](https://github.com/JuliaLang/juliaup) README for how to install 1.12 and make that version the default.

Then clone this repository

```bash
git clone git@github.com:NumericalEarth/Breeze.jl.git
```

Open Julia from within the local directory of the repo via:

```bash
julia --project
```

The first time, we need to install any dependencies:

```julia
julia> using Pkg; Pkg.instantiate()
```

Now we are ready to run any of the examples!

For instance, by increasing the resolution of the cloudy Kelvin-Helmholtz instability
to `Nx=1536` and `Nz=1024` and running

```julia
julia> include("examples/cloudy_kelvin_helmholtz.jl")
```

to get

https://github.com/user-attachments/assets/f47ff268-b2e4-401c-a114-a0aaf0c7ead3

Or cranking up the spatial resolution of the thermal bubble example to to `size = (1024, 512)` and running

```julia
julia> include("examples/thermal_bubble.jl")
```

we get

https://github.com/user-attachments/assets/c9a0c9c3-c199-48b8-9850-f967bdcc4bed
