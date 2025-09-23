using Oceananigans.Architectures: architecture
using Oceananigans.Grids: inactive_cell, Center, znode, ZDirection
using Oceananigans.Utils: prettysummary, launch!
using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ, Δzᶜᶜᶜ, ℑzᵃᵃᶠ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, divᶜᶜᶜ
using Oceananigans.Solvers: solve!, FourierTridiagonalPoissonSolver
using Oceananigans.Fields: Field, XFaceField, YFaceField, ZFaceField, set!
using Oceananigans.BoundaryConditions: fill_halo_regions!, FieldBoundaryConditions, regularize_field_boundary_conditions

using ..Thermodynamics:
    AtmosphereThermodynamics,
    ReferenceStateConstants,
    reference_pressure,
    reference_density,
    mixture_gas_constant,
    mixture_heat_capacity,
    dry_air_gas_constant

using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: tridiagonal_direction, compute_main_diagonal!, compute_lower_diagonal!

#####
##### Formulation definition
#####

struct AnelasticFormulation{FT, F}
    constants :: ReferenceStateConstants{FT}
    reference_pressure :: F
    reference_density :: F
end

function Base.summary(formulation::AnelasticFormulation)
    p₀ = formulation.constants.base_pressure
    θᵣ = formulation.constants.reference_potential_temperature
    return string("AnelasticFormulation(p₀=", prettysummary(p₀),
                  ", θᵣ=", prettysummary(θᵣ), ")")
end

Base.show(io::IO, formulation::AnelasticFormulation) = print(io, "AnelasticFormulation")

field_names(::AnelasticFormulation, tracer_names) = (:ρu, :ρv, :ρw, :ρe, :ρq, tracer_names...)

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

struct AnelasticThermodynamicState{FT}
    specific_moist_static_energy :: FT
    specific_humidity :: FT
    height :: FT
end

const c = Center()

function thermodynamic_state(i, j, k, grid,
                             formulation::AnelasticFormulation,
                             thermo,
                             energy,
                             absolute_humidity)
    @inbounds begin
        ρe = energy[i, j, k]
        ρᵣ = formulation.reference_density[i, j, k]
        ρqᵗ = absolute_humidity[i, j, k]
    end

    cᵖᵈ = thermo.dry_air.heat_capacity
    qᵗ = ρqᵗ / ρᵣ
    e = ρe / ρᵣ
    z = znode(i, j, k, grid, c, c, c)
    
    return AnelasticThermodynamicState(e, qᵗ, z)
end

@inline function specific_volume(i, j, k, grid, formulation, temperature, specific_humidity, thermo)
    @inbounds begin
        q  = specific_humidity[i, j, k]
        pᵣ = formulation.reference_pressure[i, j, k]
        T = temperature[i, j, k]
    end

    qᵛ = q
    qᵈ = one(qᵛ) - qᵛ
    Rᵐ = mixture_gas_constant(qᵈ, qᵛ, thermo)

    return Rᵐ * T / pᵣ
end

@inline function reference_specific_volume(i, j, k, grid, formulation, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    pᵣ = @inbounds formulation.reference_pressure[i, j, k]
    θᵣ = formulation.constants.reference_potential_temperature
    return Rᵈ * θᵣ / pᵣ
end

function collect_prognostic_fields(::AnelasticFormulation,
                                   density,
                                   momentum,
                                   energy,
                                   absolute_humidity,
                                   condensates,
                                   tracers)

    thermodynamic_variables = (ρe=energy, ρq=absolute_humidity)

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
    reference_density = anelastic_formulation.reference_density
    tridiagonal_formulation = AnelasticTridiagonalSolverFormulation(reference_density)
    solver = FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation)
    # solver = FourierTridiagonalPoissonSolver(grid) #; tridiagonal_formulation)
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
    ρʳ = reference_density

    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    @inbounds begin
        ρ¹ = ρʳ[i, j, 1]
        ρᴺ = ρʳ[i, j, Nz]
        ρ̄² = ℑzᵃᵃᶠ(i, j, 2, grid, ρʳ)
        ρ̄ᴺ = ℑzᵃᵃᶠ(i, j, Nz, grid, ρʳ)

        D[i, j, 1]  = - ρ̄² / Δzᵃᵃᶠ(i, j,  2, grid) - ρ¹ * Δzᵃᵃᶜ(i, j,  1, grid) * (λx[i] + λy[j])
        D[i, j, Nz] = - ρ̄ᴺ / Δzᵃᵃᶠ(i, j, Nz, grid) - ρᴺ * Δzᵃᵃᶜ(i, j, Nz, grid) * (λx[i] + λy[j])

        for k in 2:Nz-1
            ρᵏ = ρʳ[i, j, k]
            ρ̄⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρʳ)
            ρ̄ᵏ = ℑzᵃᵃᶠ(i, j, k, grid, ρʳ)

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

#####
##### Fractional and time stepping
#####

"""
Update the predictor momentum (ρu, ρv, ρw) with the non-hydrostatic pressure via

    u^{n+1} = u^n - δₓp_{NH} / Δx * Δt
"""
@kernel function _pressure_correct_momentum!(M, grid, Δt, αʳ_pₙ, ρʳ)
    i, j, k = @index(Global, NTuple)

    ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρʳ)
    ρᶜ = @inbounds ρʳ[i, j, k]

    @inbounds M.ρu[i, j, k] -= ρᶜ * Δt * ∂xᶠᶜᶜ(i, j, k, grid, αʳ_pₙ)
    @inbounds M.ρv[i, j, k] -= ρᶜ * Δt * ∂yᶜᶠᶜ(i, j, k, grid, αʳ_pₙ)
    @inbounds M.ρw[i, j, k] -= ρᶠ * Δt * ∂zᶜᶜᶠ(i, j, k, grid, αʳ_pₙ)
end
