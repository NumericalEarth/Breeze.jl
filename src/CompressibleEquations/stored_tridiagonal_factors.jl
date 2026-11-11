#####
##### Column (z-direction) Thomas sweeps for `BatchedTridiagonalSolver` systems
#####
##### Two specializations of `Oceananigans.Solvers`' `solve_batched_tridiagonal_system_z!`, whose arithmetic
##### they follow operation for operation, so results are bitwise identical:
#####
##### 1. Stored factors for systems solved many times with the same coefficients. The acoustic loop solves
#####    the same vertical systems on every substep of a stage, since their coefficients depend only on
#####    stage-frozen quantities. Factoring once per stage and substituting per substep removes the
#####    per-substep coefficient evaluation.
#####
##### 2. An in-place solve for the once-per-stage implicit steps, whose right-hand side is the solution
#####    array (`solve_tridiagonal_in_place!`).
#####
##### Both sweeps carry the recurrence in registers. The Thomas recurrence makes level k depend on level
##### k ∓ 1, and a column kernel runs one thread per column, so its run time is the length of that
##### dependency chain rather than the memory it streams. Oceananigans' kernel reads `ϕ[i, j, k-1]`,
##### `t[i, j, k]` and `ϕ[i, j, k+1]` back from the arrays it has just stored them in: without aliasing
##### information the compiler cannot forward the stored value, so every level waits for a store and a
##### load in global memory (the compiled sweep of the in-place implicit step re-reads `ϕ[i, j, k-1]`
##### from global memory in each iteration). Here the previous level's solution and pivot stay in local
##### variables, converted to the element type of the array they are stored in so they equal the value a
##### re-read would return.
#####
##### TODO: move both upstream. `BatchedTridiagonalSolver` should (a) carry `ϕ[k ∓ 1]`, `β` and `t[k]`
##### in registers in all three `solve_batched_tridiagonal_system_{x,y,z}!` sweeps, which changes no
##### result; (b) offer a `factor!`/`substitute!` split for coefficients that stay fixed over many
##### solves, with `StoredTridiagonalFactors` as its storage; and (c) solve several right-hand sides
##### (a tuple of fields, each with its own coefficient arguments) per thread, which would give a column
##### thread several independent recurrences to overlap. (c) is not done here: each field needs its own
##### `t` scratch, and the coefficient arguments of an implicit step (the model fields among them) would
##### have to be shared across the batch rather than repeated per field to stay within the size limit
##### of a GPU kernel's arguments. It should be measured on a GPU before it is built.
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

Solve the factored system for the right-hand side `rhs`, writing the solution to `ϕ`. By default the
solve is in place: `ϕ` holds the right-hand side on entry and the solution on return.
"""
function substitute_tridiagonal!(ϕ, F::StoredTridiagonalFactors, grid, rhs = ϕ)
    launch!(architecture_of(grid), grid, :xy, _substitute_tridiagonal_z!, ϕ, rhs, F.β, F.t, F.a, grid)
    return nothing
end

# `ϕ⁻` and `ϕ⁺` carry the solution of the level just solved (see the header).
@kernel function _substitute_tridiagonal_z!(ϕ, f, β, t, a, grid)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    FT = eltype(ϕ)
    @inbounds begin
        ϕ⁻ = convert(FT, f[i, j, 1] / β[i, j, 1])
        ϕ[i, j, 1] = ϕ⁻
        for k = 2:Nz
            βᵏ = β[i, j, k]
            fᵏ = f[i, j, k]
            # As in Oceananigans: elide the update where the system is not diagonally dominant.
            definitely_diagonally_dominant = abs(βᵏ) > 10 * eps(FT)
            ϕ★ = (fᵏ - masked_multiply(a[i, j, k], ϕ⁻)) / βᵏ
            ϕ⁻ = convert(FT, ifelse(definitely_diagonally_dominant, ϕ★, fᵏ))
            ϕ[i, j, k] = ϕ⁻
        end
        ϕ⁺ = ϕ⁻
        for k = Nz-1:-1:1
            ϕ⁺ = convert(FT, ϕ[i, j, k] - masked_multiply(t[i, j, k+1], ϕ⁺))
            ϕ[i, j, k] = ϕ⁺
        end
    end
end

"""
$(TYPEDSIGNATURES)

Solve in place the z-direction tridiagonal system of `solver` with coefficient arguments `args` (as
`solve!` would receive them): `ϕ` holds the right-hand side on entry and the solution on return.
Equivalent to `solve!(ϕ, solver, ϕ, args...)`, bitwise, with the recurrence carried in registers.
"""
function solve_tridiagonal_in_place!(ϕ, solver::BatchedTridiagonalSolver, args)
    grid = solver.grid
    launch!(architecture_of(grid), grid, :xy, _solve_tridiagonal_in_place_z!, ϕ, solver.t,
            solver.a, solver.b, solver.c, grid, solver.parameters, args, ZDirection())
    return nothing
end

# `βᵏ`, `ϕ⁻` and `ϕ⁺` carry the pivot and the solution of the level just solved (see the header).
# The coefficients of level k are evaluated before `ϕ[i, j, k]` is overwritten, as in Oceananigans:
# the boundary rows of an implicit step read it.
@kernel function _solve_tridiagonal_in_place_z!(ϕ, t, sa, sb, sc, grid, p, args, direction)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    FT = eltype(ϕ)
    @inbounds begin
        βᵏ = get_coefficient(i, j, 1, grid, sb, p, direction, args...)
        ϕ⁻ = convert(FT, ϕ[i, j, 1] / βᵏ)
        ϕ[i, j, 1] = ϕ⁻
        for k = 2:Nz
            cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, sc, p, direction, args...)
            bᵏ   = get_coefficient(i, j, k,   grid, sb, p, direction, args...)
            aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, sa, p, direction, args...)
            tᵏ = convert(eltype(t), cᵏ⁻¹ / βᵏ)
            βᵏ = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)
            fᵏ = ϕ[i, j, k]
            # As in Oceananigans: elide the update where the system is not diagonally dominant.
            definitely_diagonally_dominant = abs(βᵏ) > 10 * eps(FT)
            ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕ⁻)) / βᵏ
            ϕ⁻ = convert(FT, ifelse(definitely_diagonally_dominant, ϕ★, fᵏ))
            t[i, j, k] = tᵏ
            ϕ[i, j, k] = ϕ⁻
        end
        ϕ⁺ = ϕ⁻
        for k = Nz-1:-1:1
            ϕ⁺ = convert(FT, ϕ[i, j, k] - masked_multiply(t[i, j, k+1], ϕ⁺))
            ϕ[i, j, k] = ϕ⁺
        end
    end
end
