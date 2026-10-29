#####
##### Density-weighted implicit-diffusion rows carried from level to level
#####
##### The z sweep of `BatchedTridiagonalSolver` evaluates, at level k, `c(k-1)`, `b(k)` and `a(k-1)` through
##### `get_coefficient`. For a `DensityWeightedImplicitOperator` over a single scalar-diffusivity closure, those three
##### rows (see density_weighted_implicit_diffusion.jl) read: at a z-Center location the diffusivity of faces k and
##### k+1, the inverse spacings of center k and of faces k and k+1, the densities of faces k and k+1 and of cells
##### k-1 and k; and `du(k-1)` again, which row k-1's diagonal already formed. Every one of those except the values of
##### face k+1 and cell k was evaluated at level k-1.
#####
##### The sweep below is Oceananigans' `solve_batched_tridiagonal_system_z!` (same operations on values, same order,
##### same types) with the diffusion half of the rows formed from carried values: per level one diffusivity, two
##### inverse spacings, one face and one cell density, and the products `du ρ` and `dl ρ` that the off-diagonals and
##### the diagonal share. Each value is the expression of Oceananigans' `ivd_upper_diagonal` / `ivd_lower_diagonal`
##### and of `density_weighted_ivd_*`, on the same operands, so every coefficient is the same number: it is
##### computed once instead of up to three times. The implicit-advection parts, the linear coefficient and the
##### boundary-flux diagonal are evaluated per row exactly as before.
#####
##### Only closures whose `_ivd_*_diagonal` are Oceananigans' generic single-closure rows carry
##### (`AbstractScalarDiffusivity`: `ScalarDiffusivity`, `TKEBasedTurbulenceClosure`); closure tuples, `nothing` and
##### everything else keep the `get_coefficient` sweep.
#####

using Oceananigans.Grids: peripheral_node
using Oceananigans.Advection: densityᶜᶜᶜ, densityᶜᶜᶠ
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, getclosure, ivd_diffusivity, Δz⁻¹
using Oceananigans.Solvers: masked_multiply, back_substitute_z!, right_hand_side_ahead, float_eltype


# The trailing arguments `implicit_step!` hands `solve!`:
# (closure, closure_fields, id, ℓx, ℓy, ℓz, Δt, clock, fields, advection, w, density, top_bc, bottom_bc, immersed_bc, ϕ)
const CarriedRowArguments = Tuple{<:AbstractScalarDiffusivity, Any, Any, Any, Any, Union{Center, Face}, Any, Any, Any,
                                  <:DensityWeightedImplicitOperator, Any, Any, Any, Any, Any, Any}

#####
##### The diffusion half of row k from the carried values. Each returns
#####
#####   Pu = du(k) ρ⁺        (the product `du * ρ` of the upper diagonal at k and of the diagonal at k)
#####   Pl = dl(k-1) ρ⁻      (the product `dl * ρ` of the lower diagonal at k-1 and of the diagonal at k)
#####   D  = the density the diagonal at k divides by (ρᶜₖ at z-Centers, ρᶠₖ at z-Faces)
#####   S  = the values to carry to level k+1
#####

# z-Center: S holds face k's diffusivity, inverse spacing, density and activity (row k's lower face).
@inline function carried_face_values(i, j, k, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)
    κ    = ivd_diffusivity(i, j, k, grid, ℓx, ℓy, Face(), closure_ij, K, id, clk, fields)
    Δz⁻¹ᶠ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, Face())
    ρᶠ   = densityᶜᶜᶠ(i, j, k, grid, ρ)
    p    = !peripheral_node(i, j, k, grid, ℓx, ℓy, Face())
    return (κ, Δz⁻¹ᶠ, ρᶠ, p)
end

@inline carried_initial_state(i, j, grid, ::Center, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ) =
    carried_face_values(i, j, 1, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)

@inline function carried_diffusion_row(i, j, k, grid, ::Center, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ)
    κᵏ, Δz⁻¹ᶠₖ, ρᶠₖ, pₖ = S
    κᵏ⁺¹, Δz⁻¹ᶠₖ₊₁, ρᶠₖ₊₁, pₖ₊₁ = S⁺ = carried_face_values(i, j, k+1, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)
    Δz⁻¹ᶜₖ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, Center())
    ρᶜₖ = densityᶜᶜᶜ(i, j, k, grid, ρ)

    du = - Δt * κᵏ⁺¹ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ₊₁)   # ivd_upper_diagonal(k)
    du = du * pₖ₊₁
    dl = - Δt * κᵏ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ)       # ivd_lower_diagonal(k-1)
    dl = dl * pₖ

    return du * ρᶠₖ₊₁, dl * ρᶠₖ, ρᶜₖ, S⁺
end

# z-Face: S holds center k-1's diffusivity, inverse spacing and density (row k's lower stress).
@inline function carried_center_values(i, j, k, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)
    ν    = ivd_diffusivity(i, j, k, grid, ℓx, ℓy, Center(), closure_ij, K, id, clk, fields)
    Δz⁻¹ᶜ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, Center())
    ρᶜ   = densityᶜᶜᶜ(i, j, k, grid, ρ)
    return (ν, Δz⁻¹ᶜ, ρᶜ)
end

@inline carried_initial_state(i, j, grid, ::Face, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ) =
    carried_center_values(i, j, 0, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)

@inline function carried_diffusion_row(i, j, k, grid, ::Face, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ)
    νᵏ⁻¹, Δz⁻¹ᶜₖ₋₁, ρᶜₖ₋₁ = S
    νᵏ, Δz⁻¹ᶜₖ, ρᶜₖ = S⁺ = carried_center_values(i, j, k, grid, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ)
    Δz⁻¹ᶠₖ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, Face())
    ρᶠₖ = densityᶜᶜᶠ(i, j, k, grid, ρ)
    pₖ = !peripheral_node(i, j, k, grid, ℓx, ℓy, Face())

    du = - Δt * νᵏ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ)          # ivd_upper_diagonal(k)
    du = du * pₖ
    dl = - Δt * νᵏ⁻¹ * (Δz⁻¹ᶜₖ₋₁ * Δz⁻¹ᶠₖ)      # ivd_lower_diagonal(k-1)
    dl = dl * pₖ

    return du * ρᶜₖ, dl * ρᶜₖ₋₁, ρᶠₖ, S⁺
end

# The diagonal at k: `density_weighted_ivd_diagonal + density_weighted_advection_diagonal + boundary_flux_diagonal`.
@inline function carried_diagonal(i, j, k, grid, Pu, Pl, D, args)
    clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, operator, w, ρ, top_bc, bottom_bc, immersed_bc, ϕ = args
    lin    = _implicit_linear_coefficient(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    d_diff = one(grid) - Δt * lin - (Pu + Pl) / D
    d_adv  = density_weighted_advection_diagonal(i, j, k, grid, operator.scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
    d_bc   = boundary_flux_diagonal(i, j, k, grid, ℓx, ℓy, ℓz, Δt, clk, fields, top_bc, bottom_bc, immersed_bc, ϕ)
    return d_diff + d_adv + d_bc
end

@inline function Solvers.solve_batched_tridiagonal_system_z!(i, j, Nz, ϕ,
                                                     a::VerticallyImplicitDiffusionLowerDiagonal,
                                                     b::VerticallyImplicitDiffusionDiagonal,
                                                     c::VerticallyImplicitDiffusionUpperDiagonal,
                                                     f, t, grid, p, args::CarriedRowArguments, tridiagonal_direction::ZDirection)
    FT = eltype(ϕ)
    TT = eltype(t)
    ahead = right_hand_side_ahead(f)
    ε = 10 * eps(float_eltype(ϕ))

    clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, operator, w, ρ = args
    scheme = operator.scheme
    ρ_diff = weighting_density(operator.density, ρ)
    closure_ij = getclosure(i, j, clo)

    @inbounds begin
        S = carried_initial_state(i, j, grid, ℓz, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ_diff)
        Puᵏ, Plᵏ, Dᵏ, S = carried_diffusion_row(i, j, 1, grid, ℓz, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ_diff)

        β  = carried_diagonal(i, j, 1, grid, Puᵏ, Plᵏ, Dᵏ, args)
        f₁ = Solvers.get_coefficient(i, j, 1, grid, f, p, tridiagonal_direction, args...)
        ϕᵏ⁻¹ = convert(FT, f₁ / β)
        ϕ[i, j, 1] = ϕᵏ⁻¹

        fᵏ = ahead ? Solvers.get_coefficient(i, j, min(2, Nz), grid, f, p, tridiagonal_direction, args...) : f₁

        for k = 2:Nz
            fᵏ⁺¹ = ahead ? Solvers.get_coefficient(i, j, ifelse(k < Nz, k+1, k), grid, f, p, tridiagonal_direction, args...) : fᵏ

            Puᵏ⁻¹ = Puᵏ
            Dᵏ⁻¹  = Dᵏ
            Puᵏ, Plᵏ, Dᵏ, S = carried_diffusion_row(i, j, k, grid, ℓz, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ_diff)

            # density_weighted_ivd_upper_diagonal(k-1) = du(k-1) ρ / D(k), lower_diagonal(k-1) = dl(k-1) ρ / D(k-1)
            cᵏ⁻¹ = Puᵏ⁻¹ / Dᵏ + density_weighted_advection_upper_diagonal(i, j, k-1, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
            bᵏ   = carried_diagonal(i, j, k, grid, Puᵏ, Plᵏ, Dᵏ, args)
            aᵏ⁻¹ = Plᵏ / Dᵏ⁻¹ + density_weighted_advection_lower_diagonal(i, j, k-1, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ)

            tᵏ = convert(TT, cᵏ⁻¹ / β)
            t[i, j, k] = tᵏ
            β = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)
            fᵏ = ahead ? fᵏ : Solvers.get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            definitely_diagonally_dominant = abs(β) > ε
            ϕ★ = (fᵏ - masked_multiply(aᵏ⁻¹, ϕᵏ⁻¹)) / β
            if definitely_diagonally_dominant
                ϕᵏ = convert(FT, ϕ★)
                ϕ[i, j, k] = ϕᵏ
            else
                ϕᵏ = ϕ[i, j, k]
            end

            ϕᵏ⁻¹ = ϕᵏ
            fᵏ = fᵏ⁺¹
        end

        back_substitute_z!(ϕ, t, i, j, Nz, ϕᵏ⁻¹)
    end

    return nothing
end

#####
##### Several right-hand sides through one set of rows
#####
##### Cell-centred scalars whose rows are identical (the same diffusivity array, no linear coefficient, no implicit
##### advection, no implicit-explicit boundary flux: see `shared_implicit_rows_key`) are solved in one launch. The
##### rows, β and t are formed once per column; each right-hand side takes the arithmetic of the single-RHS sweep
##### above (same β and t sequence, same operations, same order), so each solution has the bits of its own solve.
#####

"""
$(TYPEDSIGNATURES)

An object that identifies the diffusion rows of the closure scalar `id` (compared with `===`), or `nothing` when
they must be solved on their own: when they carry a linear coefficient, or the closure does not opt in.
Scalars with the same key, no implicit advection and no implicit-explicit boundary flux share every row.
"""
shared_implicit_rows_key(closure, closure_fields, id) = nothing

@kernel function _multiple_right_hand_side_solve!(ϕs, rhs, t, grid, p, args)
    i, j = @index(Global, NTuple)
    multiple_right_hand_side_sweep!(i, j, size(grid, 3), ϕs, rhs, t, grid, p, args)
end

# Where the right-hand sides come from. In place: each ϕ is its own right-hand side (as `implicit_step!` does it).
# Folded RK3 (handoff C6 e97db3f8, tier A): the right-hand side is `u⁰ + Δt G`, the value `_rk3_substep!` stores,
# formed on load; where the sweep keeps the right-hand side (β ≈ 0) it stores that value, which is what the in-place
# solve's `ϕ[i, j, k]` held.
struct InPlaceRightHandSides end

struct FoldedRK3RightHandSides{U, G, T}
    ϕ⁰s :: U
    Gs :: G
    Δt :: T
end

Adapt.adapt_structure(to, s::FoldedRK3RightHandSides) =
    FoldedRK3RightHandSides(Adapt.adapt(to, s.ϕ⁰s), Adapt.adapt(to, s.Gs), s.Δt)

@inline right_hand_side_values(ϕs, ::InPlaceRightHandSides, i, j, k) = level_values(ϕs, i, j, k)
@inline right_hand_side_values(ϕs, s::FoldedRK3RightHandSides, i, j, k) = rk3_values(ϕs, s.ϕ⁰s, s.Gs, s.Δt, i, j, k)

@inline rk3_values(::Tuple{}, ::Tuple{}, ::Tuple{}, Δt, i, j, k) = ()
@inline rk3_values(ϕs, ϕ⁰s, Gs, Δt, i, j, k) =
    (convert(eltype(first(ϕs)), @inbounds(first(ϕ⁰s)[i, j, k]) + Δt * @inbounds(first(Gs)[i, j, k])),
     rk3_values(Base.tail(ϕs), Base.tail(ϕ⁰s), Base.tail(Gs), Δt, i, j, k)...)

# Each ϕ is its own right-hand side (an in-place solve, as `implicit_step!` does it).
@inline function multiple_right_hand_side_sweep!(i, j, Nz, ϕs, rhs, t, grid, p, args::CarriedRowArguments)
    FT = eltype(first(ϕs))
    TT = eltype(t)
    ε = 10 * eps(float_eltype(first(ϕs)))

    clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, operator, w, ρ = args
    scheme = operator.scheme
    ρ_diff = weighting_density(operator.density, ρ)
    closure_ij = getclosure(i, j, clo)

    @inbounds begin
        S = carried_initial_state(i, j, grid, ℓz, closure_ij, K, id, ℓx, ℓy, clk, fields, ρ_diff)
        Puᵏ, Plᵏ, Dᵏ, S = carried_diffusion_row(i, j, 1, grid, ℓz, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ_diff)

        β = carried_diagonal(i, j, 1, grid, Puᵏ, Plᵏ, Dᵏ, args)
        ϕᵏ⁻¹ = first_level_solutions!(ϕs, right_hand_side_values(ϕs, rhs, i, j, 1), i, j, β, FT)
        fᵏ = right_hand_side_values(ϕs, rhs, i, j, min(2, Nz))

        for k = 2:Nz
            fᵏ⁺¹ = right_hand_side_values(ϕs, rhs, i, j, ifelse(k < Nz, k+1, k))

            Puᵏ⁻¹ = Puᵏ
            Dᵏ⁻¹  = Dᵏ
            Puᵏ, Plᵏ, Dᵏ, S = carried_diffusion_row(i, j, k, grid, ℓz, S, closure_ij, K, id, ℓx, ℓy, Δt, clk, fields, ρ_diff)

            cᵏ⁻¹ = Puᵏ⁻¹ / Dᵏ + density_weighted_advection_upper_diagonal(i, j, k-1, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ)
            bᵏ   = carried_diagonal(i, j, k, grid, Puᵏ, Plᵏ, Dᵏ, args)
            aᵏ⁻¹ = Plᵏ / Dᵏ⁻¹ + density_weighted_advection_lower_diagonal(i, j, k-1, grid, scheme, w, Δt, ℓx, ℓy, ℓz, ρ)

            tᵏ = convert(TT, cᵏ⁻¹ / β)
            t[i, j, k] = tᵏ
            β = bᵏ - masked_multiply(aᵏ⁻¹, tᵏ)
            ϕᵏ⁻¹ = forward_solutions!(ϕs, rhs, fᵏ, ϕᵏ⁻¹, i, j, k, aᵏ⁻¹, β, ε, FT)
            fᵏ = fᵏ⁺¹
        end

        back_substitutions!(ϕs, ϕᵏ⁻¹, t, i, j, Nz)
    end

    return nothing
end

# Helpers with every value passed as an argument: a closure over the sweep's reassigned β would be boxed on the GPU.
@inline level_values(::Tuple{}, i, j, k) = ()
@inline level_values(ϕs, i, j, k) = (@inbounds(first(ϕs)[i, j, k]), level_values(Base.tail(ϕs), i, j, k)...)

@inline first_level_solutions!(::Tuple{}, ::Tuple{}, i, j, β, FT) = ()
@inline first_level_solutions!(ϕs, fs, i, j, β, FT) =
    (first_level_solution!(first(ϕs), first(fs), i, j, β, FT), first_level_solutions!(Base.tail(ϕs), Base.tail(fs), i, j, β, FT)...)

@inline function first_level_solution!(ϕ, f₁, i, j, β, FT)
    ϕ₁ = convert(FT, f₁ / β)
    @inbounds ϕ[i, j, 1] = ϕ₁
    return ϕ₁
end

# The value the sweep keeps where β ≈ 0: the in-place solve's ϕ[k]; the folded update's stored right-hand side.
@inline kept_value!(ϕ, ::InPlaceRightHandSides, f, i, j, k, FT) = @inbounds ϕ[i, j, k]

@inline function kept_value!(ϕ, ::FoldedRK3RightHandSides, f, i, j, k, FT)
    ϕᵏ = convert(FT, f)
    @inbounds ϕ[i, j, k] = ϕᵏ
    return ϕᵏ
end

# Row k of the forward sweep for one right-hand side, as the single-RHS sweep does it.
@inline function forward_solution!(ϕ, rhs, f, ϕ⁻, i, j, k, aᵏ⁻¹, β, ε, FT)
    definitely_diagonally_dominant = abs(β) > ε
    ϕ★ = (f - masked_multiply(aᵏ⁻¹, ϕ⁻)) / β
    @inbounds if definitely_diagonally_dominant
        ϕᵏ = convert(FT, ϕ★)
        ϕ[i, j, k] = ϕᵏ
    else
        ϕᵏ = kept_value!(ϕ, rhs, f, i, j, k, FT)
    end
    return ϕᵏ
end

@inline forward_solutions!(::Tuple{}, rhs, ::Tuple{}, ::Tuple{}, i, j, k, aᵏ⁻¹, β, ε, FT) = ()
@inline forward_solutions!(ϕs, rhs, fs, ϕ⁻s, i, j, k, aᵏ⁻¹, β, ε, FT) =
    (forward_solution!(first(ϕs), rhs, first(fs), first(ϕ⁻s), i, j, k, aᵏ⁻¹, β, ε, FT),
     forward_solutions!(Base.tail(ϕs), rhs, Base.tail(fs), Base.tail(ϕ⁻s), i, j, k, aᵏ⁻¹, β, ε, FT)...)

@inline back_substitutions!(::Tuple{}, ::Tuple{}, t, i, j, Nz) = nothing
@inline function back_substitutions!(ϕs, ϕᴺs, t, i, j, Nz)
    back_substitute_z!(first(ϕs), t, i, j, Nz, first(ϕᴺs))
    return back_substitutions!(Base.tail(ϕs), Base.tail(ϕᴺs), t, i, j, Nz)
end

"""
$(TYPEDSIGNATURES)

`implicit_step!` for each field of `ϕs`, in one launch, when their rows are identical: the arguments are those of
`implicit_step!` for the first field, whose boundary conditions and scheme stand for all (the caller checks that the
rows agree, see [`shared_implicit_rows_key`](@ref)).
"""
function multiple_right_hand_side_implicit_step!(ϕs::Tuple, solver, closure, closure_fields, id, clock, model_fields,
                                                 Δt, advection, velocities, density;
                                                 right_hand_sides = InPlaceRightHandSides())
    ϕ = first(ϕs)
    vi_closure, vi_closure_fields = Oceananigans.TurbulenceClosures.vertically_implicit_closures(closure, closure_fields)
    bcs = ϕ.boundary_conditions
    LX, LY, LZ = Oceananigans.Fields.location(ϕ)
    w = isnothing(velocities) ? nothing : velocities.w
    args = (vi_closure, vi_closure_fields, id, LX(), LY(), LZ(), Δt, clock, model_fields,
            advection, w, density, bcs.top, bcs.bottom, bcs.immersed, ϕ)
    args isa CarriedRowArguments || throw(ArgumentError("multiple right-hand sides need carried rows"))
    grid = solver.grid
    arch = grid.architecture
    if shared_scratch_available() && Solvers.shared_thomas_scratch(arch, grid, solver.t)
        Solvers.launch_shared_column_solve!(_multiple_right_hand_side_shared_solve!, arch, grid,
                                            (ϕs, right_hand_sides, solver.t, grid, solver.parameters, args))
    else
        launch!(arch, grid, :xy, _multiple_right_hand_side_solve!, ϕs, right_hand_sides, solver.t, grid,
                solver.parameters, args)
    end
    return nothing
end

# The multi-right-hand-side sweep with its Thomas scratch in the block's shared memory (handoff P01 42166bec; see
# Oceananigans' `SharedColumnScratch`): the same stored values, so the same bits; the global `t` is not written.
shared_scratch_available() = isdefined(Solvers, :SharedColumnScratch)

@kernel function _multiple_right_hand_side_shared_solve!(ϕs, rhs, t, grid, p, args, ::Val{Nz}) where Nz
    i, j = @index(Global, NTuple)
    column = @index(Local, Linear)
    TT = @uniform eltype(t)
    scratch = @localmem TT (64, Nz)
    multiple_right_hand_side_sweep!(i, j, Nz, ϕs, rhs, Solvers.SharedColumnScratch(scratch, column), grid, p, args)
end
