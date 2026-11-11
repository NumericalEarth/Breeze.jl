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
##### 3. The same in-place solve for several fields per column thread, each with its own coefficient
#####    arguments (`solve_tridiagonal_columns_in_place!`), which can also evaluate each level's
#####    coefficients one level ahead of its use.
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
##### (a tuple of fields, each with its own coefficient arguments and its own `t` scratch) per thread, as
##### `solve_tridiagonal_columns_in_place!` does here, with the arguments the fields have in common
##### passed once.
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

#####
##### Several fields per column thread
#####
##### A column thread that sweeps one field waits, at every level, for that level's loads. Sweeping
##### several fields level by level gives it as many independent recurrences: each level first loads
##### the coefficients and right-hand sides of every field, then eliminates, so the loads overlap.
##### `prefetch` moves a level's loads ahead of the stores of the level before, which the compiler
##### cannot do itself for want of aliasing information. Neither changes the arithmetic of a field.
#####

"""
$(TYPEDSIGNATURES)

Solve in place the z-direction tridiagonal systems of `solver` for the tuple of fields `ϕ`, in one
kernel that sweeps all of them in each column thread: `ϕ[n]` holds the right-hand side of system `n` on
entry and its solution on return, bitwise as `solve_tridiagonal_in_place!` would leave it.

The coefficient arguments of system `n` are `arguments(shared, specific[n], ϕ[n])`: `shared` holds what
all systems have in common and is passed to the kernel once, which keeps a kernel's arguments from
growing with every field by the size of, say, the model's fields (GPU kernels limit the total size of
their arguments). `scratch` is a tuple of at least `length(ϕ)` arrays of the solver's scratch size, one
per system. With `prefetch`, the coefficients of each level are evaluated one level ahead of their use.
"""
function solve_tridiagonal_columns_in_place!(ϕ::Tuple, solver::BatchedTridiagonalSolver, scratch::Tuple,
                                             arguments, shared, specific::Tuple; prefetch = true)
    grid = solver.grid
    t = ntuple(n -> scratch[n], length(ϕ))
    launch!(architecture_of(grid), grid, :xy, _solve_tridiagonal_columns_in_place_z!, ϕ, t,
            solver.a, solver.b, solver.c, grid, solver.parameters, arguments, shared, specific,
            ZDirection(), Val(prefetch))
    return nothing
end

@kernel function _solve_tridiagonal_columns_in_place_z!(ϕ, t, sa, sb, sc, grid, p, arguments, shared, specific,
                                                        direction, prefetch)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    # (βᵏ, ϕ⁻) of each field: the pivot and the solution of the level just solved
    pivots = first_rows!(i, j, ϕ, sb, grid, p, arguments, shared, specific, direction)
    ahead = rows_ahead(prefetch, i, j, min(2, Nz), ϕ, sa, sb, sc, grid, p, arguments, shared, specific, direction)

    for k = 2:Nz
        rows = current_rows(ahead, i, j, k, ϕ, sa, sb, sc, grid, p, arguments, shared, specific, direction)
        ahead = rows_ahead(prefetch, i, j, min(k + 1, Nz), ϕ, sa, sb, sc, grid, p, arguments, shared, specific, direction)
        pivots = eliminate_rows!(i, j, k, ϕ, t, pivots, rows)
    end

    solutions = last_solutions(pivots)
    for k = Nz-1:-1:1
        solutions = substitute_rows!(i, j, k, ϕ, t, solutions)
    end
end

# Row 1 of each system; the diagonal is evaluated before `ϕ[i, j, 1]` is overwritten.
@inline first_rows!(i, j, ::Tuple{}, sb, grid, p, arguments, shared, ::Tuple{}, direction) = ()

@inline function first_rows!(i, j, ϕ::Tuple, sb, grid, p, arguments, shared, specific::Tuple, direction)
    ϕ₁ = first(ϕ)
    args = arguments(shared, first(specific), ϕ₁)
    β¹ = get_coefficient(i, j, 1, grid, sb, p, direction, args...)
    ϕ⁻ = @inbounds convert(eltype(ϕ₁), ϕ₁[i, j, 1] / β¹)
    @inbounds ϕ₁[i, j, 1] = ϕ⁻
    return ((β¹, ϕ⁻), first_rows!(i, j, Base.tail(ϕ), sb, grid, p, arguments, shared, Base.tail(specific), direction)...)
end

# (cᵏ⁻¹, bᵏ, aᵏ⁻¹, fᵏ) of row k of each system: every load of the level, before any of its stores.
@inline load_rows(i, j, k, ::Tuple{}, sa, sb, sc, grid, p, arguments, shared, ::Tuple{}, direction) = ()

@inline function load_rows(i, j, k, ϕ::Tuple, sa, sb, sc, grid, p, arguments, shared, specific::Tuple, direction)
    ϕ₁ = first(ϕ)
    args = arguments(shared, first(specific), ϕ₁)
    cᵏ⁻¹ = get_coefficient(i, j, k-1, grid, sc, p, direction, args...)
    bᵏ   = get_coefficient(i, j, k,   grid, sb, p, direction, args...)
    aᵏ⁻¹ = get_coefficient(i, j, k-1, grid, sa, p, direction, args...)
    fᵏ   = @inbounds ϕ₁[i, j, k]
    return ((cᵏ⁻¹, bᵏ, aᵏ⁻¹, fᵏ),
            load_rows(i, j, k, Base.tail(ϕ), sa, sb, sc, grid, p, arguments, shared, Base.tail(specific), direction)...)
end

# With prefetch the rows of level k were loaded during level k - 1; without, they are loaded now.
@inline rows_ahead(::Val{true}, i, j, k, args...) = load_rows(i, j, k, args...)
@inline rows_ahead(::Val{false}, i, j, k, args...) = nothing
@inline current_rows(ahead::Tuple, i, j, k, args...) = ahead
@inline current_rows(::Nothing, i, j, k, args...) = load_rows(i, j, k, args...)

# Forward elimination of row k of each system, as in `_solve_tridiagonal_in_place_z!`.
@inline eliminate_rows!(i, j, k, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()

@inline function eliminate_rows!(i, j, k, ϕ::Tuple, t::Tuple, pivots::Tuple, rows::Tuple)
    ϕ₁ = first(ϕ)
    t₁ = first(t)
    FT = eltype(ϕ₁)
    βᵏ⁻¹, ϕ⁻ = first(pivots)
    cᵏ⁻¹, bᵏ, aᵏ⁻¹, fᵏ = first(rows)
    tᵏ = convert(eltype(t₁), cᵏ⁻¹ / βᵏ⁻¹)
    βᵏ = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)
    # As in Oceananigans: elide the update where the system is not diagonally dominant.
    definitely_diagonally_dominant = abs(βᵏ) > 10 * eps(FT)
    ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕ⁻)) / βᵏ
    ϕᵏ = convert(FT, ifelse(definitely_diagonally_dominant, ϕ★, fᵏ))
    @inbounds t₁[i, j, k] = tᵏ
    @inbounds ϕ₁[i, j, k] = ϕᵏ
    return ((βᵏ, ϕᵏ), eliminate_rows!(i, j, k, Base.tail(ϕ), Base.tail(t), Base.tail(pivots), Base.tail(rows))...)
end

@inline last_solutions(::Tuple{}) = ()
@inline last_solutions(pivots::Tuple) = (last(first(pivots)), last_solutions(Base.tail(pivots))...)

# Back substitution of row k of each system, carrying the solution of row k + 1.
@inline substitute_rows!(i, j, k, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()

@inline function substitute_rows!(i, j, k, ϕ::Tuple, t::Tuple, solutions::Tuple)
    ϕ₁ = first(ϕ)
    t₁ = first(t)
    ϕᵏ = @inbounds convert(eltype(ϕ₁), ϕ₁[i, j, k] - masked_multiply(t₁[i, j, k+1], first(solutions)))
    @inbounds ϕ₁[i, j, k] = ϕᵏ
    return (ϕᵏ, substitute_rows!(i, j, k, Base.tail(ϕ), Base.tail(t), Base.tail(solutions))...)
end
