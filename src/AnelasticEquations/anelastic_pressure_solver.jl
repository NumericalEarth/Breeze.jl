#####
##### Anelastic pressure solver utilities
#####

struct AnelasticTridiagonalSolverFormulation{R} <: AbstractHomogeneousNeumannFormulation
    reference_density :: R
end

Solvers.tridiagonal_direction(_formulation::AnelasticTridiagonalSolverFormulation) = ZDirection()

function AtmosphereModels.dynamics_pressure_solver(dynamics::AnelasticDynamics, grid)
    reference_density = dynamics.reference_state.density
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

#####
##### Tridiagonal solver coefficients
#####

# Note: diagonal coefficients depend on non-tridiagonal directions because
# eigenvalues depend on non-tridiagonal directions.
function Solvers.compute_main_diagonal!(main_diagonal, formulation::AnelasticTridiagonalSolverFormulation, grid, λ1, λ2)
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

function Solvers.compute_lower_diagonal!(lower_diagonal, formulation::AnelasticTridiagonalSolverFormulation, grid)
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

#####
##### Pressure solve
#####

function solve_for_anelastic_pressure!(pₙ, solver, ρŨ, Δt)
    compute_anelastic_source_term!(solver, ρŨ, Δt)
    solve!(pₙ, solver)
    return pₙ
end

function compute_anelastic_source_term!(solver::FourierTridiagonalPoissonSolver, ρŨ, Δt)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    launch!(arch, grid, :xyz, _compute_anelastic_source_term!, rhs, grid, ρŨ, Δt)
    return nothing
end

@kernel function _compute_anelastic_source_term!(rhs, grid, ρŨ, Δt)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    ρu, ρv, ρw = ρŨ
    δ = divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)
    @inbounds rhs[i, j, k] = active * Δzᶜᶜᶜ(i, j, k, grid) * δ / Δt
end
