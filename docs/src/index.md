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

```@example intro
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Breeze

Nx = Nz = 64
Lz = 4 * 1024
grid = RectilinearGrid(CPU(), size=(Nx, Nz), x=(0, 2Lz), z=(0, Lz), topology=(Periodic, Flat, Bounded))

reference_state = Breeze.Thermodynamics.ReferenceState(grid, base_pressure=1e5, reference_potential_temperature=288)
buoyancy = Breeze.MoistAirBuoyancy(; reference_state)

Q₀ = 1000 # heat flux in W / m²
ρ₀ = Breeze.MoistAirBuoyancies.base_density(buoyancy) # air density at z=0
cₚ = buoyancy.thermodynamics.dry_air.heat_capacity
θ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Q₀ / (ρ₀ * cₚ)))
q_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(1e-2))

advection = WENO()
tracers = (:θ, :q)
model = NonhydrostaticModel(; grid, advection, buoyancy,
                            tracers = (:θ, :q),
                            boundary_conditions = (θ=θ_bcs, q=q_bcs))

Δθ = 2 # ᵒK
Tₛ = reference_state.potential_temperature # K
θᵢ(x, z) = Tₛ + Δθ * z / grid.Lz + 1e-2 * Δθ * randn()
qᵢ(x, z) = 0 # 1e-2 + 1e-5 * rand()
set!(model, θ=θᵢ, q=qᵢ)

simulation = Simulation(model, Δt=10, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

run!(simulation)

T = Breeze.TemperatureField(model)
heatmap(T)
```
