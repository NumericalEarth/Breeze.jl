#####
##### Stored Thomas factors for column systems solved many times with the same coefficients
#####
##### The acoustic loop solves the same vertical system on every substep of a stage when its coefficients
##### depend only on stage-frozen quantities. Factoring once and substituting per substep removes the
##### per-substep coefficient evaluation. The arithmetic follows `Oceananigans.Solvers`'
##### `solve_batched_tridiagonal_system_z!` operation for operation, so results are bitwise identical.
##### TODO: move upstream as a `factor!`/`substitute!` split of `BatchedTridiagonalSolver`.
#####

using Oceananigans.Solvers: BatchedTridiagonalSolver, get_coefficient, masked_multiply
using Oceananigans.Architectures: architecture as architecture_of
using Oceananigans.Grids: ZDirection

"""
$(TYPEDEF)

Stored Thomas factors of a batched z-direction tridiagonal system: the pivots `β`, the forward-elimination ratios
`t`, and the lower diagonal `a`, each an `Nx × Ny × Nz` array. Built by [`factor_tridiagonal!`](@ref) and used
by [`substitute_tridiagonal!`](@ref).
"""
struct StoredTridiagonalFactors{A}
    β :: A
    t :: A
    a :: A
end

Adapt.adapt_structure(to, F::StoredTridiagonalFactors) =
    StoredTridiagonalFactors(adapt(to, F.β), adapt(to, F.t), adapt(to, F.a))

function StoredTridiagonalFactors(grid)
    arch = architecture_of(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    return StoredTridiagonalFactors(zeros(arch, FT, Nx, Ny, Nz), zeros(arch, FT, Nx, Ny, Nz), zeros(arch, FT, Nx, Ny, Nz))
end

"""
$(TYPEDSIGNATURES)

Factor the z-direction tridiagonal system of `solver` with coefficient arguments `args` (as `solve!` would receive
them) into `F`.
"""
function factor_tridiagonal!(F::StoredTridiagonalFactors, solver::BatchedTridiagonalSolver, args)
    grid = solver.grid
    launch!(architecture_of(grid), grid, :xy, _factor_tridiagonal_z!, F.β, F.t, F.a,
            solver.a, solver.b, solver.c, grid, solver.parameters, args, ZDirection())
    return nothing
end

@kernel function _factor_tridiagonal_z!(β, t, a, sa, sb, sc, grid, p, args, direction)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    @inbounds begin
        βᵏ = get_coefficient(i, j, 1, grid, sb, p, direction, args...)
        β[i, j, 1] = βᵏ
        for k = 2:Nz
            cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, sc, p, direction, args...)
            bᵏ   = get_coefficient(i, j, k,   grid, sb, p, direction, args...)
            aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, sa, p, direction, args...)
            tᵏ = cᵏ⁻¹ / βᵏ
            βᵏ = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)
            t[i, j, k] = tᵏ
            a[i, j, k] = aᵏ⁻¹
            β[i, j, k] = βᵏ
        end
    end
end

"""
$(TYPEDSIGNATURES)

Solve the factored system in place: `ϕ` holds the right-hand side on entry and the solution on return.
"""
function substitute_tridiagonal!(ϕ, F::StoredTridiagonalFactors, grid)
    launch!(architecture_of(grid), grid, :xy, _substitute_tridiagonal_z!, ϕ, F.β, F.t, F.a, grid)
    return nothing
end

@kernel function _substitute_tridiagonal_z!(ϕ, β, t, a, grid)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    FT = eltype(ϕ)
    @inbounds begin
        ϕ[i, j, 1] = ϕ[i, j, 1] / β[i, j, 1]
        for k = 2:Nz
            βᵏ = β[i, j, k]
            # As in Oceananigans: elide the update where the system is not diagonally dominant.
            definitely_diagonally_dominant = abs(βᵏ) > 10 * eps(FT)
            ϕ★ = (ϕ[i, j, k] - masked_multiply(a[i, j, k], ϕ[i, j, k-1])) / βᵏ
            ϕ[i, j, k] = ifelse(definitely_diagonally_dominant, ϕ★, ϕ[i, j, k])
        end
        for k = Nz-1:-1:1
            ϕ[i, j, k] -= masked_multiply(t[i, j, k+1], ϕ[i, j, k+1])
        end
    end
end
