import Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!

"""
    compute_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function compute_pressure_correction!(model::AnelasticModel, Δt)

    # Mask immersed velocities
    foreach(mask_immersed_field!, model.momentum)
    fill_halo_regions!(model.momentum, model.clock, fields(model))
    solver = model.pressure_solver
    pₙ = model.nonhydrostatic_pressure
    ρᵣ = model.formulation.reference_density
    solve_for_pressure!(pₙ, solver, Δt, ρᵣ, model.momentum)
    fill_halo_regions!(pₙ)

    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the predictor velocities u, v, and w with the non-hydrostatic pressure via

    `u^{n+1} = u^n - δₓp_{NH} / Δx * Δt`
"""
@kernel function _pressure_correct_momentum!(M, grid, Δt, ρᵣ, pₙ)
    i, j, k = @index(Global, NTuple)

    ρᵣᵃᵃᶜ = @inbounds ρᵣ[i, j, k]
    ρᵣᵃᵃᶠ = @inbounds ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)

    @inbounds begin
        M.ρu[i, j, k] -= ρᵣᵃᵃᶜ * ∂xᶠᶜᶜ(i, j, k, grid, pₙ) * Δt
        M.ρv[i, j, k] -= ρᵣᵃᵃᶜ * ∂yᶜᶠᶜ(i, j, k, grid, pₙ) * Δt
        M.ρw[i, j, k] -= ρᵣᵃᵃᶠ * ∂zᶜᶜᶠ(i, j, k, grid, pₙ) * Δt
    end
end

function make_pressure_correction!(model::AnelasticModel, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _pressure_correct_momentum!,
            model.momentum,
            model.grid,
            Δt,
            model.formulation.reference_density,
            model.nonhydrostatic_pressure)

    return nothing
end

using Oceananigans.Operators
using Oceananigans.DistributedComputations: DistributedFFTBasedPoissonSolver, DistributedFourierTridiagonalPoissonSolver
using Oceananigans.Grids: XDirection, YDirection, ZDirection, inactive_cell
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Solvers: solve!

using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!
import Oceananigans.Models.NonhydrostaticModels: compute_source_term!

#####
##### Calculate the right-hand-side of the anelastic pressure Poisson equation.
#####

@kernel function _fourier_tridiagonal_source_term!(rhs, ::ZDirection, grid, ρᵣ, ρŨ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = ρŨ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    ρᵣᵏ = ρᵣ[i, j, k]
    @inbounds rhs[i, j, k] = active * Δzᶜᶜᶜ(i, j, k, grid) * δ / ρᵣᵏ
end

function compute_source_term!(solver::DistributedFourierTridiagonalPoissonSolver, ρᵣ, ρŨ)
    rhs = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, ρᵣ, ρŨ)
    return nothing
end

function compute_source_term!(solver::FourierTridiagonalPoissonSolver, ρᵣ, ρŨ)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, ρᵣ, ρŨ)
    return nothing
end
