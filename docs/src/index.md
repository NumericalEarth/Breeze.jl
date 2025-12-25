# Breeze.jl

Fast, friendly atmosphere simulations on CPUs and GPUs.

Breeze provides software for flexible software package for finite-volume atmosphere simulations on CPUs and GPUs, based on [Oceananigans](https://github.com/CliMA/Oceananigans.jl).
Like Oceananigans, it provides a radically productive user interface that makes simple simulations easy, and complex, creative simulations possible.

## Features

Breeze provides two ways to simulate moist atmospheres:

1. An [`AtmosphereModel`](@ref Breeze.AtmosphereModels.AtmosphereModel) which currently supports anelastic approximation following [Pauluis2008](@citet):
    * `AtmosphereModel` has simple warm-phase saturation adjustment microphysics
    * `AtmosphereModel` is being rapidly developed and changes day-to-day!
    * A roadmap is coming soon, and will include radiation, bulk, bin, and superdroplet microphysics, a fully compressible formulation, and more

2. A [`MoistAirBuoyancy`](@ref) buoyancy implementation that can be used with [Oceananigans](https://clima.github.io/OceananigansDocumentation/stable/)' [`NonhydrostaticModel`](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Models.NonhydrostaticModels.NonhydrostaticModel-Tuple{}) to simulate atmospheric flows with the [Boussinesq approximation](https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy)):
    * `MoistAirBuoyancy` includes a warm-phase saturation adjustment implementation
    * Note that our attention is focused on `AtmosphereModel`!


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

simulation = Simulation(model, Δt=10, stop_time=1hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

run!(simulation)

heatmap(PotentialTemperature(model), colormap=:thermal)
```
