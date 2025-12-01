using ..Thermodynamics:
    MoistureMassFractions,
    StaticEnergyState,
    PotentialTemperatureState,
    ThermodynamicConstants,
    ReferenceState,
    mixture_gas_constant,
    mixture_heat_capacity,
    dry_air_gas_constant

using Oceananigans: Oceananigans, CenterField
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: inactive_cell, prettysummary
using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ, divᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.Solvers: solve!, AbstractHomogeneousNeumannFormulation

using KernelAbstractions: @kernel, @index
using Adapt: Adapt, adapt

import Oceananigans.Solvers: tridiagonal_direction, compute_main_diagonal!, compute_lower_diagonal!
import Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!

#####
##### Formulation definition
#####

"""
$(TYPEDSIGNATURES)

`AnelasticFormulation` is a dynamical formulation wherein the density and pressure are
small perturbations from a dry, hydrostatic, adiabatic `reference_state`.
The prognostic energy variable is the moist static energy density.
The energy density equation includes a buoyancy flux term, following [Pauluis2008](@citet).
"""
struct AnelasticFormulation{T, R, P}
    thermodynamics :: T
    reference_state :: R
    pressure_anomaly :: P
end

struct StaticEnergyThermodynamics{E}
    energy_density :: E
    specific_energy :: E
end

struct PotentialTemperatureThermodynamics{F}
    potential_temperature_density :: F  # ρθ (prognostic)
    potential_temperature :: F          # θ = ρθ / ρᵣ (diagnostic)
end

Adapt.adapt_structure(to, thermo::StaticEnergyThermodynamics) =
    StaticEnergyThermodynamics(adapt(to, thermo.energy_density),
                               adapt(to, thermo.specific_energy))

Adapt.adapt_structure(to, thermo::PotentialTemperatureThermodynamics) =
    PotentialTemperatureThermodynamics(adapt(to, thermo.potential_temperature_density),
                                       adapt(to, thermo.potential_temperature))

Adapt.adapt_structure(to, formulation::AnelasticFormulation) =
    AnelasticFormulation(adapt(to, formulation.thermodynamics),
                         adapt(to, formulation.reference_state),
                         adapt(to, formulation.pressure_anomaly))

const AnelasticModel = AtmosphereModel{<:AnelasticFormulation}

# Type aliases for convenience
const ASEF = AnelasticFormulation{<:StaticEnergyThermodynamics}
const APTF = AnelasticFormulation{<:PotentialTemperatureThermodynamics}
const StaticEnergyAnelasticModel = AtmosphereModel{<:ASEF}
const PotentialTemperatureAnelasticModel = AtmosphereModel{<:APTF}

"""
    $(TYPEDSIGNATURES)

Construct a "stub" `AnelasticFormulation` with just the `reference_state`.
The thermodynamics and pressure fields are materialized later in the model constructor.

Keyword Arguments
=================

- `thermodynamics`: The thermodynamic variable to use. Options are `:static_energy` (default)
  or `:potential_temperature`.
"""
AnelasticFormulation(reference_state; thermodynamics=:static_energy) =
    AnelasticFormulation(thermodynamics, reference_state, nothing)

function default_formulation(grid, constants)
    reference_state = ReferenceState(grid, constants)
    return AnelasticFormulation(reference_state)
end

"""
    $(TYPEDSIGNATURES)

Materialize a stub `AnelasticFormulation` into a full formulation with thermodynamic fields
and the pressure anomaly field. The thermodynamic fields depend on the type of thermodynamics
specified in the stub (`:static_energy` or `:potential_temperature`).
"""
function materialize_formulation(stub::AnelasticFormulation, grid, boundary_conditions)
    thermo_type = stub.thermodynamics
    pressure_anomaly = CenterField(grid)
    
    if thermo_type === :potential_temperature
        potential_temperature_density = CenterField(grid, boundary_conditions=boundary_conditions.ρθ)
        potential_temperature = CenterField(grid) # θ = ρθ / ρᵣ (diagnostic)
        thermodynamics = PotentialTemperatureThermodynamics(potential_temperature_density, potential_temperature)
    else # default to static energy
        energy_density = CenterField(grid, boundary_conditions=boundary_conditions.ρe)
        specific_energy = CenterField(grid) # e = ρe / ρᵣ (diagnostic per-mass energy)
        thermodynamics = StaticEnergyThermodynamics(energy_density, specific_energy)
    end
    
    return AnelasticFormulation(thermodynamics, stub.reference_state, pressure_anomaly)
end

function Base.summary(formulation::AnelasticFormulation)
    p₀ = formulation.reference_state.base_pressure
    θ₀ = formulation.reference_state.potential_temperature
    return string("AnelasticFormulation(p₀=", prettysummary(p₀),
                  ", θ₀=", prettysummary(θ₀), ")")
end

Base.show(io::IO, formulation::AnelasticFormulation) = print(io, "AnelasticFormulation")

# Return :ρθ instead of :ρe for potential temperature thermodynamics
function prognostic_field_names(stub::AnelasticFormulation, microphysics, tracer_names)
    # Stub stores the symbol in thermodynamics field
    thermo_type = stub.thermodynamics
    if thermo_type === :potential_temperature
        default_names = (:ρu, :ρv, :ρw, :ρθ, :ρqᵗ)
    else
        default_names = (:ρu, :ρv, :ρw, :ρe, :ρqᵗ)
    end
    microphysical_names = prognostic_field_names(microphysics)
    return tuple(default_names..., microphysical_names..., tracer_names...)
end

#####
##### Thermodynamic state
#####

"""
    $(TYPEDSIGNATURES)

Return `StaticEnergyState` computed from the prognostic state including
energy density, moisture density, and microphysical fields.
"""
function diagnose_thermodynamic_state(i, j, k, grid, formulation::ASEF,
                                      microphysics,
                                      microphysical_fields,
                                      constants,
                                      specific_moisture)
    e = @inbounds formulation.thermodynamics.specific_energy[i, j, k]
    pᵣ = @inbounds formulation.reference_state.pressure[i, j, k]
    ρᵣ = @inbounds formulation.reference_state.density[i, j, k]
    qᵗ = @inbounds specific_moisture[i, j, k]

    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    z = znode(i, j, k, grid, c, c, c)

    return StaticEnergyState(e, q, z, pᵣ)
end

"""
    $(TYPEDSIGNATURES)

Return `PotentialTemperatureState` computed from the prognostic state including
potential temperature density, moisture density, and microphysical fields.
"""
function diagnose_thermodynamic_state(i, j, k, grid, formulation::APTF,
                                      microphysics,
                                      microphysical_fields,
                                      constants,
                                      specific_moisture)
    θ = @inbounds formulation.thermodynamics.potential_temperature[i, j, k]
    pᵣ = @inbounds formulation.reference_state.pressure[i, j, k]
    ρᵣ = @inbounds formulation.reference_state.density[i, j, k]
    p₀ = formulation.reference_state.base_pressure
    qᵗ = @inbounds specific_moisture[i, j, k]

    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)

    return PotentialTemperatureState(θ, q, p₀, pᵣ)
end


function collect_prognostic_fields(::ASEF,
                                   momentum,
                                   thermodynamic_density,
                                   moisture_density,
                                   microphysical_fields,
                                   tracers)

    thermodynamic_variables = (ρe=thermodynamic_density, ρqᵗ=moisture_density)
    return merge(momentum, thermodynamic_variables, microphysical_fields, tracers)
end

function collect_prognostic_fields(::APTF,
                                   momentum,
                                   thermodynamic_density,
                                   moisture_density,
                                   microphysical_fields,
                                   tracers)

    thermodynamic_variables = (ρθ=thermodynamic_density, ρqᵗ=moisture_density)
    return merge(momentum, thermodynamic_variables, microphysical_fields, tracers)
end

#####
##### Accessor functions for thermodynamic fields
#####

# Get the prognostic thermodynamic density field (ρe or ρθ)
get_thermodynamic_density(f::ASEF) = f.thermodynamics.energy_density
get_thermodynamic_density(f::APTF) = f.thermodynamics.potential_temperature_density

# Get the name of the thermodynamic density field
thermodynamic_density_name(::ASEF) = :ρe
thermodynamic_density_name(::APTF) = :ρθ

# Accessor functions for individual thermodynamic fields
energy_density(thermo::StaticEnergyThermodynamics) = thermo.energy_density
energy_density(::PotentialTemperatureThermodynamics) = nothing

specific_energy(thermo::StaticEnergyThermodynamics) = thermo.specific_energy
specific_energy(::PotentialTemperatureThermodynamics) = nothing

potential_temperature_density(::StaticEnergyThermodynamics) = nothing
potential_temperature_density(thermo::PotentialTemperatureThermodynamics) = thermo.potential_temperature_density

potential_temperature(::StaticEnergyThermodynamics) = nothing
potential_temperature(thermo::PotentialTemperatureThermodynamics) = thermo.potential_temperature

#####
##### fields() and prognostic_fields() implementations
#####

function _fields(model, ::ASEF)
    e = model.formulation.thermodynamics.specific_energy
    auxiliary = (e=e, T=model.temperature, qᵗ=model.specific_moisture)
    return merge(prognostic_fields(model), model.velocities, auxiliary)
end

function _fields(model, ::APTF)
    θ = model.formulation.thermodynamics.potential_temperature
    auxiliary = (θ=θ, T=model.temperature, qᵗ=model.specific_moisture)
    return merge(prognostic_fields(model), model.velocities, auxiliary)
end

function _prognostic_fields(model, ::ASEF)
    ρe = model.formulation.thermodynamics.energy_density
    thermodynamic_fields = (ρe=ρe, ρqᵗ=model.moisture_density)
    microphysical_names = prognostic_field_names(model.microphysics)
    prognostic_microphysical_fields = NamedTuple{microphysical_names}(
        model.microphysical_fields[name] for name in microphysical_names)
    return merge(model.momentum, thermodynamic_fields, prognostic_microphysical_fields, model.tracers)
end

function _prognostic_fields(model, ::APTF)
    ρθ = model.formulation.thermodynamics.potential_temperature_density
    thermodynamic_fields = (ρθ=ρθ, ρqᵗ=model.moisture_density)
    microphysical_names = prognostic_field_names(model.microphysics)
    prognostic_microphysical_fields = NamedTuple{microphysical_names}(
        model.microphysical_fields[name] for name in microphysical_names)
    return merge(model.momentum, thermodynamic_fields, prognostic_microphysical_fields, model.tracers)
end

function materialize_momentum_and_velocities(formulation::AnelasticFormulation, grid, boundary_conditions)
    ρu = XFaceField(grid, boundary_conditions=boundary_conditions.ρu)
    ρv = YFaceField(grid, boundary_conditions=boundary_conditions.ρv)
    ρw = ZFaceField(grid, boundary_conditions=boundary_conditions.ρw)
    momentum = (; ρu, ρv, ρw)

    velocity_bcs = NamedTuple(name => FieldBoundaryConditions() for name in (:u, :v, :w))
    velocity_bcs = regularize_field_boundary_conditions(velocity_bcs, grid, (:u, :v, :w))
    u = XFaceField(grid, boundary_conditions=velocity_bcs.u)
    v = YFaceField(grid, boundary_conditions=velocity_bcs.v)
    w = ZFaceField(grid, boundary_conditions=velocity_bcs.w)
    velocities = (; u, v, w)

    return velocities, momentum
end

#####
##### Anelastic pressure solver utilities
#####

struct AnelasticTridiagonalSolverFormulation{R} <: AbstractHomogeneousNeumannFormulation
    reference_density :: R
end

tridiagonal_direction(formulation::AnelasticTridiagonalSolverFormulation) = ZDirection()

function formulation_pressure_solver(anelastic_formulation::AnelasticFormulation, grid)
    reference_density = anelastic_formulation.reference_state.density
    tridiagonal_formulation = AnelasticTridiagonalSolverFormulation(reference_density)

    solver = if grid isa Oceananigans.ImmersedBoundaries.ImmersedBoundaryGrid
        # With this method, we are using an approximate solver that
        # will produce a divergent velocity field near terrain.
        FourierTridiagonalPoissonSolver(grid.underlying_grid; tridiagonal_formulation)
    else # the solver is exact
        FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation)
    end

    return solver
end

# Note: diagonal coefficients depend on non-tridiagonal directions because
# eigenvalues depend on non-tridiagonal directions.
function compute_main_diagonal!(main_diagonal, formulation::AnelasticTridiagonalSolverFormulation, grid, λ1, λ2)
    arch = grid.architecture
    reference_density = formulation.reference_density
    launch!(arch, grid, :xy, _compute_anelastic_main_diagonal!, main_diagonal, grid, λ1, λ2, reference_density)
    return nothing
end

@kernel function _compute_anelastic_main_diagonal!(D, grid, λx, λy, reference_density)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    ρᵣ = reference_density

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    @inbounds begin
        ρ¹ = ρᵣ[1, 1, 1]
        ρᴺ = ρᵣ[1, 1, Nz]
        ρ̄² = ℑzᵃᵃᶠ(i, j, 2, grid, ρᵣ)
        ρ̄ᴺ = ℑzᵃᵃᶠ(i, j, Nz, grid, ρᵣ)

        D[i, j, 1]  = - ρ̄² / Δzᵃᵃᶠ(i, j,  2, grid) - ρ¹ * Δzᵃᵃᶜ(i, j,  1, grid) * (λx[i] + λy[j])
        D[i, j, Nz] = - ρ̄ᴺ / Δzᵃᵃᶠ(i, j, Nz, grid) - ρᴺ * Δzᵃᵃᶜ(i, j, Nz, grid) * (λx[i] + λy[j])

        for k in 2:Nz-1
            ρᵏ = ρᵣ[1, 1, k]
            ρ̄⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρᵣ)
            ρ̄ᵏ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)

            D[i, j, k] = - (ρ̄⁺ / Δzᵃᵃᶠ(i, j, k+1, grid) + ρ̄ᵏ / Δzᵃᵃᶠ(i, j, k, grid)) - ρᵏ * Δzᵃᵃᶜ(i, j, k, grid) * (λx[i] + λy[j])
        end
    end
end

function compute_lower_diagonal!(lower_diagonal, formulation::AnelasticTridiagonalSolverFormulation, grid)
    N = length(lower_diagonal)
    arch = grid.architecture
    reference_density = formulation.reference_density
    launch!(arch, grid, tuple(N), _compute_anelastic_lower_diagonal!, lower_diagonal, grid, reference_density)
    return nothing
end

@kernel function _compute_anelastic_lower_diagonal!(lower_diagonal, grid, reference_density)
    k = @index(Global)
    @inbounds begin
        ρ̄⁺ = ℑzᵃᵃᶠ(1, 1, k+1, grid, reference_density)
        lower_diagonal[k] = ρ̄⁺ / Δzᵃᵃᶠ(1, 1, k+1, grid)
    end
end

function compute_pressure_correction!(model::AnelasticModel, Δt)
    # Mask immersed velocities
    foreach(mask_immersed_field!, model.momentum)
    fill_halo_regions!(model.momentum, model.clock, fields(model))

    ρᵣ = model.formulation.reference_state.density
    ρŨ = model.momentum
    solver = model.pressure_solver
    p′ = model.pressure
    solve_for_anelastic_pressure!(p′, solver, ρŨ, Δt)
    fill_halo_regions!(p′)

    return nothing
end

function solve_for_anelastic_pressure!(pₙ, solver, ρŨ, Δt)
    compute_anelastic_source_term!(solver, ρŨ, Δt)
    solve!(pₙ, solver)
    return pₙ
end

function compute_anelastic_source_term!(solver::FourierTridiagonalPoissonSolver, ρŨ, Δt)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    launch!(arch, grid, :xyz, _compute_anelastic_source_term!, rhs, grid, ρŨ, Δt)
    return nothing
end

@kernel function _compute_anelastic_source_term!(rhs, grid, ρŨ, Δt)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    ρu, ρv, ρw = ρŨ
    δ = divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)
    @inbounds rhs[i, j, k] = active * Δzᶜᶜᶜ(i, j, k, grid) * δ / Δt
end

#=
function compute_source_term!(solver::DistributedFourierTridiagonalPoissonSolver, Ũ)
    rhs = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, Ũ)
    return nothing
end
=#

#####
##### Fractional and time stepping
#####

@kernel function _pressure_correct_momentum!(M, grid, Δt, αᵣ_pₙ, ρᵣ)
    i, j, k = @index(Global, NTuple)

    ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)
    ρᶜ = @inbounds ρᵣ[i, j, k]

    @inbounds M.ρu[i, j, k] -= ρᶜ * Δt * ∂xᶠᶜᶜ(i, j, k, grid, αᵣ_pₙ)
    @inbounds M.ρv[i, j, k] -= ρᶜ * Δt * ∂yᶜᶠᶜ(i, j, k, grid, αᵣ_pₙ)
    @inbounds M.ρw[i, j, k] -= ρᶠ * Δt * ∂zᶜᶜᶠ(i, j, k, grid, αᵣ_pₙ)
end

"""
$(TYPEDSIGNATURES)

Update the predictor momentum ``(ρu, ρv, ρw)`` with the non-hydrostatic pressure via

```math
(\\rho\\boldsymbol{u})^{n+1} = (\\rho\\boldsymbol{u})^n - \\Delta t \\, \\rho_r \\boldsymbol{\\nabla} \\left( \\alpha_r p_{nh} \\right)
```
"""
function make_pressure_correction!(model::AnelasticModel, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _pressure_correct_momentum!,
            model.momentum,
            model.grid,
            Δt,
            model.pressure,
            model.formulation.reference_state.density)

    return nothing
end
