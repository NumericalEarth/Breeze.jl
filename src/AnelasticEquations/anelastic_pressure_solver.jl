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
    return fourier_tridiagonal_solver(architecture(grid), grid, tridiagonal_formulation)
end

# Another float type: solve on a copy of the grid in that type. The coefficient kernels
# read ρᵣ in the model's precision and promote it, so the solver needs no copy of ρᵣ.
anelastic_pressure_solver(FT, grid, reference_density) =
    anelastic_pressure_solver(FT, with_number_type(FT, grid), reference_density)

fourier_tridiagonal_solver(arch, grid, tridiagonal_formulation) =
    FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation)

function fourier_tridiagonal_solver(::Distributed, grid, tridiagonal_formulation)
    global_grid = reconstruct_global_grid(grid)
    return DistributedFourierTridiagonalPoissonSolver(global_grid, grid; tridiagonal_formulation)
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

    # ρᵣ is read from its first column, like the lower diagonal: `grid` may be a transposed layout
    # of a distributed grid (complete in x, say), whose (i, j) do not index the local ρᵣ.
    # Using a homogeneous Neumann (zero Gradient) boundary condition:
    @inbounds begin
        ρ¹ = ρᵣ[1, 1, 1]
        ρᴺ = ρᵣ[1, 1, Nz]
        ρ̄² = ℑzᵃᵃᶠ(1, 1, 2, grid, ρᵣ)
        ρ̄ᴺ = ℑzᵃᵃᶠ(1, 1, Nz, grid, ρᵣ)

        D[i, j, 1]  = - ρ̄² / Δzᵃᵃᶠ(i, j,  2, grid) - ρ¹ * Δzᵃᵃᶜ(i, j,  1, grid) * (λx[i] + λy[j])
        D[i, j, Nz] = - ρ̄ᴺ / Δzᵃᵃᶠ(i, j, Nz, grid) - ρᴺ * Δzᵃᵃᶜ(i, j, Nz, grid) * (λx[i] + λy[j])

        for k in 2:Nz-1
            ρᵏ = ρᵣ[1, 1, k]
            ρ̄⁺ = ℑzᵃᵃᶠ(1, 1, k+1, grid, ρᵣ)
            ρ̄ᵏ = ℑzᵃᵃᶠ(1, 1, k, grid, ρᵣ)

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
function AtmosphereModels.update_pressure_solver!(solver::FourierTridiagonalPoissonSolver, ::AnelasticDynamics)
    tridiagonal_solver = solver.batched_tridiagonal_solver
    formulation = solver.tridiagonal_formulation

    Solvers.compute_main_diagonal!(tridiagonal_solver.b, formulation, solver.grid, solver.poisson_eigenvalues...)
    Solvers.compute_lower_diagonal!(tridiagonal_solver.a, formulation, solver.grid)
    return nothing
end

function AtmosphereModels.update_pressure_solver!(solver::DistributedFourierTridiagonalPoissonSolver, dynamics::AnelasticDynamics)
    tridiagonal_solver = solver.batched_tridiagonal_solver
    tridiagonal_grid = tridiagonal_solver.grid
    formulation = AnelasticTridiagonalSolverFormulation(dynamics.reference_state.density)
    λx, λy = distributed_horizontal_eigenvalues(solver)

    Solvers.compute_main_diagonal!(tridiagonal_solver.b, formulation, tridiagonal_grid, λx, λy)
    Solvers.compute_lower_diagonal!(tridiagonal_solver.a, formulation, tridiagonal_grid)
    return nothing
end

# TODO: unlike `FourierTridiagonalPoissonSolver`, `DistributedFourierTridiagonalPoissonSolver` does not
# store its Poisson eigenvalues, so we recompute them here, mirroring its constructor. Storing them
# (partitioned to the layout of the tridiagonal solve) in Oceananigans would remove this duplication.
function distributed_horizontal_eigenvalues(solver)
    global_grid = solver.global_grid
    tridiagonal_grid = solver.batched_tridiagonal_solver.grid
    arch = architecture(tridiagonal_grid)
    TX, TY, _ = topology(global_grid)

    λx_global = dropdims(poisson_eigenvalues(global_grid, global_grid.Nx, global_grid.Lx, 1, TX()), dims=(2, 3))
    λy_global = dropdims(poisson_eigenvalues(global_grid, global_grid.Ny, global_grid.Ly, 2, TY()), dims=(1, 3))

    # The tridiagonal solve acts on a transposed layout, in which x may be complete (slab-x decomposition)
    # or partitioned (pencil decomposition); `partition_coordinate` returns the local part in either case.
    λx = partition_coordinate(λx_global, size(tridiagonal_grid, 1), arch, 1)
    λy = partition_coordinate(λy_global, size(tridiagonal_grid, 2), arch, 2)

    child_arch = child_architecture(arch)
    return on_architecture(child_arch, λx), on_architecture(child_arch, λy)
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
    kernel_Δt = kernel_time_step(arch, grid, Δt)
    launch!(arch, grid, :xyz, _compute_anelastic_source_term!, rhs, grid, ρŨ, kernel_Δt)
    fill!(solver.storage, 0)  # workaround for bitwise reproducibility — see Oceananigans #5960
    return nothing
end

function compute_anelastic_source_term!(solver::DistributedFourierTridiagonalPoissonSolver, ρŨ, Δt)
    rhs = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
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
