using ..Thermodynamics:
    MoistureMassFractions,
    ThermodynamicConstants,
    ReferenceState,
    AnelasticThermodynamicState,
    mixture_gas_constant,
    mixture_heat_capacity,
    dry_air_gas_constant

using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: inactive_cell, prettysummary
using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ, divᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.Solvers: solve!

using KernelAbstractions: @kernel, @index
using Adapt: Adapt, adapt

import Oceananigans.Solvers: tridiagonal_direction, compute_main_diagonal!, compute_lower_diagonal!
import Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!

#####
##### Formulation definition
#####

struct AnelasticFormulation{R}
    reference_state :: R
end

Adapt.adapt_structure(to, formulation::AnelasticFormulation) =
    AnelasticFormulation(adapt(to, formulation.reference_state))

const AnelasticModel = AtmosphereModel{<:AnelasticFormulation}

function Base.summary(formulation::AnelasticFormulation)
    p₀ = formulation.reference_state.base_pressure
    θ₀ = formulation.reference_state.potential_temperature
    return string("AnelasticFormulation(p₀=", prettysummary(p₀),
                  ", θ₀=", prettysummary(θ₀), ")")
end

Base.show(io::IO, formulation::AnelasticFormulation) = print(io, "AnelasticFormulation")

field_names(::AnelasticFormulation, tracer_names) = (:ρu, :ρv, :ρw, :ρe, :ρqᵗ, tracer_names...)

function AnelasticFormulation(grid, state_constants, thermo)
    pᵣ = Field{Nothing, Nothing, Center}(grid)
    ρᵣ = Field{Nothing, Nothing, Center}(grid)
    set!(pᵣ, z -> reference_pressure(z, state_constants, thermo))
    set!(ρᵣ, z -> reference_density(z, state_constants, thermo))
    fill_halo_regions!(pᵣ)
    fill_halo_regions!(ρᵣ)
    return AnelasticFormulation(state_constants, pᵣ, ρᵣ)
end

#####
##### Thermodynamic state
#####

function thermodynamic_state(i, j, k, grid, formulation::AnelasticFormulation, thermo, energy_density, moisture_density)
    @inbounds begin
        e = energy_density[i, j, k]
        pᵣ = formulation.reference_state.pressure[i, j, k]
        ρᵣ = formulation.reference_state.density[i, j, k]
        ρqᵗ = moisture_density[i, j, k]
    end

    cᵖᵈ = thermo.dry_air.heat_capacity
    θ = e / (cᵖᵈ * ρᵣ)

    qᵗ = ρqᵗ / ρᵣ
    q = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ)) # assuming non-condensed state
    Rᵐ = mixture_gas_constant(q, thermo)
    cᵖᵐ = mixture_heat_capacity(q, thermo)

    p₀ = formulation.reference_state.base_pressure
    Π = (pᵣ / p₀)^(Rᵐ / cᵖᵐ)

    return AnelasticThermodynamicState(θ, q, ρᵣ, pᵣ, Π)
end

@inline function specific_volume(i, j, k, grid, formulation, temperature, moisture_fraction, thermo)
    @inbounds begin
        qᵗ = moisture_fraction[i, j, k]
        pᵣ = formulation.reference_state.pressure[i, j, k]
        T = temperature[i, j, k]
    end

    # TODO: fix this assumption of non-condensed state
    q = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
    Rᵐ = mixture_gas_constant(q, thermo)

    return Rᵐ * T / pᵣ
end

@inline function buoyancy(i, j, k, grid, formulation, T, qᵗ, thermo)
    α = specific_volume(i, j, k, grid, formulation, T, qᵗ, thermo)
    αᵣ = reference_specific_volume(i, j, k, grid, formulation, thermo)
    g = thermo.gravitational_acceleration
    return g * (α - αᵣ) / αᵣ
end

@inline function reference_specific_volume(i, j, k, grid, formulation, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    pᵣ = @inbounds formulation.reference_state.pressure[i, j, k]
    θ₀ = formulation.reference_state.potential_temperature
    return Rᵈ * θ₀ / pᵣ
end

function collect_prognostic_fields(::AnelasticFormulation,
                                   density,
                                   momentum,
                                   energy_density,
                                   moisture_density,
                                   condensates,
                                   tracers)

    thermodynamic_variables = (ρe=energy_density, ρqᵗ=moisture_density)

    return merge(momentum, thermodynamic_variables, condensates, tracers)
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

struct AnelasticTridiagonalSolverFormulation{R}
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
    pₙ = model.nonhydrostatic_pressure
    solve_for_anelastic_pressure!(pₙ, solver, ρŨ, Δt)

    fill_halo_regions!(pₙ)

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
(\\rho\\boldsymbol{u})^{n+1} = (\\rho\\boldsymbol{u})^n - \\Delta t \\rho_r \\boldsymbol{\\nabla} \\left( \\alpha_r p_{nh} \\right)
```
"""
function make_pressure_correction!(model::AnelasticModel, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _pressure_correct_momentum!,
            model.momentum,
            model.grid,
            Δt,
            model.nonhydrostatic_pressure,
            model.formulation.reference_state.density)

    return nothing
end
