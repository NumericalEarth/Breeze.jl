#####
##### Anelastic pressure solver utilities
#####

struct AnelasticTridiagonalSolverFormulation{R} <: AbstractHomogeneousNeumannFormulation
    reference_density :: R
end

Solvers.tridiagonal_direction(formulation::AnelasticTridiagonalSolverFormulation) = ZDirection()

AtmosphereModels.dynamics_pressure_solver(dynamics::AnelasticDynamics, grid) =
    anelastic_pressure_solver(dynamics.pressure_solver_float_type, pressure_solver_grid(grid),
                              dynamics.reference_state.density)

# The solver is exact on a grid without an immersed boundary.
pressure_solver_grid(grid) = grid

# With an immersed boundary we use an approximate solver on the underlying grid,
# which produces a divergent velocity field near terrain.
pressure_solver_grid(grid::ImmersedBoundaryGrid) = grid.underlying_grid

# The grid is already in the solver's float type
function anelastic_pressure_solver(::Type{FT}, grid::AbstractGrid{FT}, reference_density) where FT
    tridiagonal_formulation = AnelasticTridiagonalSolverFormulation(reference_density)
    return FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation)
end

# Another float type: solve on copies of the grid and of ρᵣ in that type
function anelastic_pressure_solver(FT, grid, reference_density)
    solver_grid = with_number_type(FT, grid)
    solver_density = Field(Oceananigans.instantiated_location(reference_density), solver_grid)
    parent(solver_density) .= parent(reference_density)
    return anelastic_pressure_solver(FT, solver_grid, solver_density)
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

# The tridiagonal coefficients carry ρᵣ, so they are recomputed whenever ρᵣ changes.
function AtmosphereModels.update_pressure_solver!(solver::FourierTridiagonalPoissonSolver, dynamics::AnelasticDynamics)
    tridiagonal_solver = solver.batched_tridiagonal_solver
    formulation = solver.tridiagonal_formulation

    # A solver in another float type holds its own copy of ρᵣ; otherwise this copies ρᵣ onto itself
    parent(formulation.reference_density) .= parent(dynamics.reference_state.density)

    Solvers.compute_main_diagonal!(tridiagonal_solver.b, formulation, solver.grid, solver.poisson_eigenvalues...)
    Solvers.compute_lower_diagonal!(tridiagonal_solver.a, formulation, solver.grid)
    return nothing
end

#####
##### Pressure solve
#####

function solve_for_anelastic_pressure!(pₙ, solver, ρŨ, Δt)
    compute_anelastic_source_term!(solver, ρŨ, Δt)
    fill!(solver.storage, 0)  # workaround for bitwise reproducibility — see Oceananigans #5960
    solve!(pₙ, solver)
    return pₙ
end

function compute_anelastic_source_term!(solver::FourierTridiagonalPoissonSolver, ρŨ, Δt)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    kernel_Δt = kernel_time_step(arch, grid, Δt)
    launch!(arch, grid, :xyz, _compute_anelastic_source_term!, rhs, grid, ρŨ, kernel_Δt)
    return nothing
end

@kernel function _compute_anelastic_source_term!(rhs, grid, ρŨ, Δt)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    ρu, ρv, ρw = ρŨ
    δ = divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)
    @inbounds rhs[i, j, k] = active * Δzᶜᶜᶜ(i, j, k, grid) * δ / Δt
end
