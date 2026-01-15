# Breeze.jl

Fast, friendly atmosphere simulations on CPUs and GPUs.

Breeze is a library for simulating atmospheric flows, convection, clouds, weather, and hurricanes on CPUs and GPUs.
Much of Breeze's power flows from [Oceananigans](https://github.com/CliMA/Oceananigans.jl), which provides a user interface, grids, fields, solvers, advection schemes, Lagrangian particles, physics, and more.

Breeze's [`AtmosphereModel`](@ref Breeze.AtmosphereModels.AtmosphereModel) provides anelastic, compressible, and prescribed (kinematic) dynamics, closures for large eddy simulation, WENO advection schemes, strong stability preserving (SSP) RK3 time-stepping, saturation adjustment microphysics, Kessler microphysics, and one- and two-moment bulk microphysics schemes via an extension to the [Climate Modeling Alliance's](https://clima.caltech.edu/) excellent [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) package.
An extension to [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl) provides solvers for gray, clear-sky, and all-sky radiative transfer.
Breeze's examples include single column radiation, idealized thermal bubbles and inertia-gravity waves and Kelvin-Helmholtz, [BOMEX](@cite Siebesma2003) shallow convection, [RICO](@cite vanZanten2011) trade-wind cumulus, [supercells](@cite Klemp2015), mountain waves, and more.

Don't hesitate to get in touch on the [NumericalEarth slack](https://join.slack.com/t/numericalearth/shared_invite/zt-3kjcowmpg-B0s3nalWkvZg8IBc~BIJEA) or by opening a new [discussion](https://github.com/NumericalEarth/Breeze.jl/discussions)!

## Roadmap and a call to action

Our goal is to build a very fast, easy to learn, productive tool for atmospheric research, teaching, and forecasting, as well as a platform for the development of algorithms, numerical methods, parameterizations, microphysical schemes, and atmosphere model components.
This won't be the effort of a single group, project, or even a single community.
Such a lofty aim can only be realized by a wide-ranging and sustained collaboration of passionate people.
Maybe that includes you --- consider it!
Model development is hard but rewarding, and builds useful skills for a myriad of pursuits.

The goals of the current group of model developers include developing

- ⛈️ **Advanced microphysics**: Predicted Particle Property (P3) bulk microphysics, spectral bin schemes, and Lagrangian superdroplet methods for high-fidelity cloud and precipitation modeling
- ️🏔 **Acoustic substepping and terrain-following coordinates**: A compressible dynamical core with horizontally explicit, vertically-implicit acoustic substepping that efficiently resolves sound waves in flow over complex topography with smooth [sigma coordinates](https://en.wikipedia.org/wiki/Sigma_coordinate_system)
- 🔬 **Open boundaries and nesting**: Two-way nesting to support multi-level nested simulations embedded in global atmosphere simulations
- 🌀 **Coupled atmosphere-ocean simulations**: Support for high-resolution coupled atmosphere-ocean simulations via [ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl)

If you have ideas, dreams, or criticisms that can make Breeze and its future better, don't hesitate to speak up by [opening issues](https://github.com/NumericalEarth/Breeze.jl/issues/new/choose) and contributing pull requests.

## Installation

Breeze is a registered Julia package. First [install Julia](https://julialang.org/install/); suggested version 1.12. See [juliaup](https://github.com/JuliaLang/juliaup) README for how to install 1.12 and make that version the default.

Then launch Julia and type

```julia
julia> using Pkg

julia> Pkg.add("Breeze")
```

which will install the latest stable version of Breeze that's compatible with your current environment.

You can check which version of Breeze you got via

```julia
Pkg.status("Breeze")
```

If you want to live on the cutting edge, you can use, e.g.,
`Pkg.add(; url="https://github.com/NumericalEarth/Breeze.jl.git", rev="main")` to install the latest version of
Breeze from `main` branch. For more information, see the
[Pkg.jl documentation](https://pkgdocs.julialang.org).

## Quick Start

A basic free convection simulation with an `AtmosphereModel`:

```@example
using Breeze
using Oceananigans.Units
using CairoMakie
using Random: seed!

# Fix the seed to generate the noise, for reproducible simulations.
# You can try different seeds to explore different noise patterns.
seed!(42)

Nx = Nz = 64
Lz = 4 * 1024
grid = RectilinearGrid(size=(Nx, Nz), x=(0, 2Lz), z=(0, Lz), topology=(Periodic, Flat, Bounded))

p₀, θ₀ = 1e5, 288 # reference state parameters
reference_state = ReferenceState(grid, surface_pressure=p₀, potential_temperature=θ₀)
dynamics = AnelasticDynamics(reference_state)

Q₀ = 1000 # heat flux in W / m²
thermodynamic_constants = ThermodynamicConstants()
cᵖᵈ = thermodynamic_constants.dry_air.heat_capacity
ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Q₀ / cᵖᵈ))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(1e-2))

advection = WENO()
model = AtmosphereModel(grid; advection, dynamics, thermodynamic_constants,
                              boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs))

Δθ = 2 # ᵒK
Tₛ = reference_state.potential_temperature # K
θᵢ(x, z) = Tₛ + Δθ * z / grid.Lz + 2e-2 * Δθ * (rand() - 0.5)
set!(model, θ=θᵢ)

simulation = Simulation(model, Δt=10, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

run!(simulation)

heatmap(PotentialTemperature(model), colormap=:thermal)
```

!!! note "Note about reproducibility"

    Due to their [chaotic nature](https://en.wikipedia.org/wiki/Chaos_theory), even the smallest numerical differences can cause nonlinear systems, such as atmospheric models, not to be reproducible on different systems, therefore the figures you will get by running the simulations in this manual may not match the figures shown here.
    For more information about this, see the [section about reproducibility](@ref reproducibility).

## Relationship to Oceananigans

Breeze is built on [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl), an ocean modeling package that provides grids, fields, operators, advection schemes, time-steppers, turbulence closures, and output infrastructure.
Breeze extends Oceananigans with atmospheric dynamics, thermodynamics, microphysics, and radiation to create a complete atmosphere simulation capability.
The two packages share a common philosophy: fast, flexible, GPU-native Julia code with a user interface designed for productivity and experimentation.
To learn these foundational components of Breeze, please see the [Oceananigans documentation](https://clima.github.io/OceananigansDocumentation/stable/).

If you're familiar with Oceananigans, you'll feel right at home with Breeze.
If you're new to both, Breeze is a great entry point—and the skills you develop transfer directly to ocean and climate modeling with Oceananigans and [ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl).
