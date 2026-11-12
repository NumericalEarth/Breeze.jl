#####
##### Several fields per column thread for `BatchedTridiagonalSolver` systems
#####
##### The per-stage implicit steps solve several fields, each with its own coefficient arguments, with
##### one column thread sweeping all of them (`solve_tridiagonal_columns_in_place!`): the Thomas
##### recurrence makes level k depend on level k ∓ 1, so a column sweep's run time is the length of
##### that dependency chain, and interleaving several independent chains hides it. Each level's
##### coefficients are evaluated one level ahead of their use. The arithmetic follows
##### `Oceananigans.Solvers`' sweep operation for operation, so results are bitwise identical.
#####
##### TODO: move upstream as an option of `BatchedTridiagonalSolver` that solves a tuple of fields per
##### thread, each with its own coefficient arguments and `t` scratch, with the arguments they have in
##### common passed once.
#####

using Oceananigans.Solvers: BatchedTridiagonalSolver, get_coefficient, masked_multiply
using Oceananigans.Architectures: architecture as architecture_of
using Oceananigans.Grids: ZDirection

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Solve in place the z-direction tridiagonal systems of `solver` for the tuple of fields `ϕ`, in one
kernel that sweeps all of them in each column thread: `ϕ[n]` holds the right-hand side of system `n` on
entry and its solution on return, bitwise as `solve!(ϕ[n], solver, ϕ[n], args...)` would leave it.

The coefficient arguments of system `n` are `arguments(shared, specific[n], ϕ[n])`: `shared` holds what
all systems have in common and is passed to the kernel once, which keeps a kernel's arguments from
growing with every field by the size of, say, the model's fields (GPU kernels limit the total size of
their arguments). `scratch` is a tuple of at least `length(ϕ)` arrays of the solver's scratch size, one
per system. The coefficients of each level are evaluated one level ahead of their use.
"""
function solve_tridiagonal_columns_in_place!(ϕ::Tuple, solver::BatchedTridiagonalSolver, scratch::Tuple,
                                             arguments, shared, specific::Tuple)
    grid = solver.grid
    t = ntuple(n -> scratch[n], length(ϕ))
    launch!(architecture_of(grid), grid, :xy, _solve_tridiagonal_columns_in_place_z!, ϕ, t,
            solver.a, solver.b, solver.c, grid, solver.parameters, arguments, shared, specific,
            ZDirection())
    return nothing
end

@kernel function _solve_tridiagonal_columns_in_place_z!(ϕ, t, sa, sb, sc, grid, p, arguments, shared, specific,
                                                        direction)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    # (βᵏ, ϕ⁻) of each field: the pivot and the solution of the level just solved
    pivots = first_rows!(i, j, ϕ, sb, grid, p, arguments, shared, specific, direction)
    # The rows of level k are loaded during level k - 1, before that level's stores.
    ahead = load_rows(i, j, min(2, Nz), ϕ, sa, sb, sc, grid, p, arguments, shared, specific, direction)

    for k = 2:Nz
        rows = ahead
        ahead = load_rows(i, j, min(k + 1, Nz), ϕ, sa, sb, sc, grid, p, arguments, shared, specific, direction)
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
