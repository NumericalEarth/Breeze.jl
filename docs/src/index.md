# Breeze.jl

Fast, friendly atmosphere simulations on CPUs and GPUs.

Breeze provides software for flexible software package for finite-volume atmosphere simulations on CPUs and GPUs, based on [Oceananigans](https://github.com/CliMA/Oceananigans.jl).
Like Oceananigans, it provides a radically productive user interface that makes simple simulations easy, and complex, creative simulations possible.

## Features

Breeze provides two ways to simulate moist atmospheres:

* A [`MoistAirBuoyancy`](@ref) that can be used with [Oceananigans](https://clima.github.io/OceananigansDocumentation/stable/)' [`NonhydrostaticModel`](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Models.NonhydrostaticModels.NonhydrostaticModel-Tuple{}) to simulate atmospheric flows with the Boussinesq approximation.

* A prototype [`AtmosphereModel`](@ref Breeze.AtmosphereModels.AtmosphereModel) that uses the anelastic approximation following [Pauluis2008](@citet).

## Installation

To use Breeze, install directly from GitHub:

```julia
using Pkg
Pkg.add("https://github.com/NumericalEarth/Breeze.jl.git")
```

## Quick Start

A basic free convection simulation:

```julia
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Breeze
using Random: seed!

# Fix the seed to generate the noise, for reproducible simulations.
# You can try different seeds to explore different noise patterns.
seed!(42)

Nx = Nz = 64
Lz = 4 * 1024
grid = RectilinearGrid(size=(Nx, Nz), x=(0, 2Lz), z=(0, Lz), topology=(Periodic, Flat, Bounded))

p₀, θ₀ = 1e5, 288 # reference state parameters
buoyancy = Breeze.MoistAirBuoyancy(grid, base_pressure=p₀, reference_potential_temperature=θ₀)

thermo = buoyancy.thermodynamics
ρ₀ = Breeze.Thermodynamics.base_density(p₀, θ₀, thermo)
cₚ = buoyancy.thermodynamics.dry_air.heat_capacity
Q₀ = 1000 # heat flux in W / m²
θ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Q₀ / (ρ₀ * cₚ)))
qᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(1e-2))

advection = WENO()
model = NonhydrostaticModel(; grid, advection, buoyancy,
                            tracers = (:θ, :qᵗ),
                            boundary_conditions = (θ=θ_bcs, qᵗ=qᵗ_bcs))

Δθ = 2 # ᵒK
Tₛ = buoyancy.reference_state.potential_temperature # K
θᵢ(x, z) = Tₛ + Δθ * z / grid.Lz + 2e-2 * Δθ * (rand() - 0.5)
set!(model, θ=θᵢ)

simulation = Simulation(model, Δt=10, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

run!(simulation)

T = Breeze.TemperatureField(model)
heatmap(T)
```
