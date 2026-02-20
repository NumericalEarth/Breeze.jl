#####
##### PerturbationMomentumAdvection: density-aware open boundary radiation scheme
#####
##### This is the Breeze analog of Oceananigans' PerturbationAdvection. It converts
##### density-weighted fields (ρψ) to intensive fields (ψ) before applying the
##### backward-Euler radiation formula, then converts back. This fixes the dimensional
##### inconsistency that arises when PerturbationAdvection is applied directly to
##### density-weighted prognostic variables (ρu, ρv, ρθ, ρq).
#####
##### TODO: Unify with Oceananigans' PerturbationAdvection by adding optional `density`
##### and `gravity_wave_speed` fields upstream. This would eliminate ~270 lines of
##### duplicated radiation logic. The upstream PerturbationAdvection would need:
#####   1. Optional `density` field (default nothing) for ρψ → ψ conversion
#####   2. Optional `gravity_wave_speed` (default 0) added to phase speed
#####   3. `_fill_*_halo!` methods for Center-located fields (currently only Face)
##### See https://github.com/CliMA/Oceananigans.jl for upstream PR.
#####

using Oceananigans: defaults
using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ

"""
$(TYPEDSIGNATURES)

A density-aware open boundary radiation scheme for density-weighted prognostic fields.

`PerturbationMomentumAdvection` extends Oceananigans'
[`PerturbationAdvection`](@ref Oceananigans.BoundaryConditions.PerturbationAdvection)
to correctly handle density-weighted fields (ρu, ρv, ρθ, ρq) by converting to
intensive (per-unit-mass) space before computing phase speeds and radiation.

The boundary condition value should be specified in **intensive units**
(m/s for velocity, K for potential temperature, kg/kg for moisture).

The scheme adds a `gravity_wave_speed` to the phase speed for momentum fields,
following [Klemp and Wilhelmson (1978)](@cite KlempWilhelmson1978).
Setting `gravity_wave_speed = 0` recovers pure advective radiation
(equivalent to `PerturbationAdvection` with density correction).

# Fields

- `inflow_timescale`: relaxation timescale for inflow [s]; default 0 (instant Dirichlet)
- `outflow_timescale`: relaxation timescale for outflow [s]; default Inf (pure radiation)
- `gravity_wave_speed`: additional phase speed [m/s]; default 0
- `density`: density field for ρψ → ψ conversion (e.g., `reference_state.density`)
"""
struct PerturbationMomentumAdvection{FT, D}
    inflow_timescale   :: FT
    outflow_timescale  :: FT
    gravity_wave_speed :: FT
    density            :: D
end

"""
$(TYPEDSIGNATURES)

Construct a `PerturbationMomentumAdvection` scheme for use with `OpenBoundaryCondition`.

# Keyword Arguments

- `density` (required): density field for converting density-weighted fields to intensive units.
  For anelastic dynamics, use `reference_state.density`.
- `gravity_wave_speed`: additional phase speed added to the exterior velocity [m/s].
  Use ~30 m/s for momentum fields (gravity wave radiation) and 0 for scalar fields. Default: 0.
- `inflow_timescale`: relaxation timescale when flow enters the domain [s]. Default: 0 (Dirichlet).
- `outflow_timescale`: relaxation timescale when flow exits the domain [s]. Default: Inf (pure radiation).

# Example

```julia
reference_state = model.dynamics.reference_state
ρ₀ = reference_state.density

# Momentum BC with gravity wave radiation
momentum_scheme = PerturbationMomentumAdvection(density=ρ₀, gravity_wave_speed=30)
ρu_bcs = FieldBoundaryConditions(
    west = OpenBoundaryCondition(U_ext; scheme=momentum_scheme),
    east = OpenBoundaryCondition(U_ext; scheme=momentum_scheme))

# Scalar BC with pure advective radiation
scalar_scheme = PerturbationMomentumAdvection(density=ρ₀)
ρθ_bcs = FieldBoundaryConditions(
    west = OpenBoundaryCondition(θ_ext; scheme=scalar_scheme),
    east = OpenBoundaryCondition(θ_ext; scheme=scalar_scheme))
```
"""
function PerturbationMomentumAdvection(FT = defaults.FloatType;
                                       density,
                                       gravity_wave_speed = 0,
                                       inflow_timescale = 0,
                                       outflow_timescale = Inf)
    return PerturbationMomentumAdvection(convert(FT, inflow_timescale),
                                         convert(FT, outflow_timescale),
                                         convert(FT, gravity_wave_speed),
                                         density)
end

Adapt.adapt_structure(to, s::PerturbationMomentumAdvection) =
    PerturbationMomentumAdvection(Adapt.adapt(to, s.inflow_timescale),
                                  Adapt.adapt(to, s.outflow_timescale),
                                  Adapt.adapt(to, s.gravity_wave_speed),
                                  Adapt.adapt(to, s.density))

const PMABC = BoundaryCondition{<:Open{<:PerturbationMomentumAdvection}}

#####
##### Right boundary stepping (east, north)
#####

@inline function step_right_pma_boundary!(bc::PMABC, l, m, boundary_indices, boundary_adjacent_indices,
                                          grid, ρψ, clock, model_fields, ΔX, k)
    iᴮ, jᴮ, kᴮ = boundary_indices
    iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    scheme = bc.classification.scheme
    cstar = scheme.gravity_wave_speed
    ρ = scheme.density
    ρₖ = @inbounds ρ[1, 1, k]

    # Convert to intensive space
    ψ_b   = @inbounds ρψ[iᴮ, jᴮ, kᴮ] / ρₖ
    ψ_int = @inbounds ρψ[iᴬ, jᴬ, kᴬ] / ρₖ

    # Prescribed exterior value in intensive units
    ψ̄_ext = getbc(bc, l, m, grid, clock, model_fields)

    # Phase speed: exterior value + gravity wave speed (outflow is +x at east / +y at north)
    c_b = ψ̄_ext + cstar
    Ũ = max(0, min(1, Δt / ΔX * c_b))

    # Inflow vs outflow timescale
    τ = ifelse(ψ̄_ext >= 0, scheme.outflow_timescale, scheme.inflow_timescale)
    τ̃ = Δt / τ

    # Backward-Euler PerturbationAdvection formula
    ψ_new = (ψ_b + Ũ * ψ_int + ψ̄_ext * τ̃) / (1 + τ̃ + Ũ)
    ψ_new = ifelse(iszero(τ), ψ̄_ext, ψ_new)

    # Convert back to extensive (density-weighted) space
    @inbounds ρψ[iᴮ, jᴮ, kᴮ] = ρₖ * ψ_new

    return nothing
end

#####
##### Left boundary stepping (west, south)
#####

@inline function step_left_pma_boundary!(bc::PMABC, l, m, boundary_indices, boundary_adjacent_indices,
                                         grid, ρψ, clock, model_fields, ΔX, k)
    iᴮ, jᴮ, kᴮ = boundary_indices
    iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    scheme = bc.classification.scheme
    cstar = scheme.gravity_wave_speed
    ρ = scheme.density
    ρₖ = @inbounds ρ[1, 1, k]

    # Convert to intensive space
    ψ_b   = @inbounds ρψ[iᴮ, jᴮ, kᴮ] / ρₖ
    ψ_int = @inbounds ρψ[iᴬ, jᴬ, kᴬ] / ρₖ

    ψ̄_ext = getbc(bc, l, m, grid, clock, model_fields)

    # Phase speed: exterior value - gravity wave speed (outflow is -x at west / -y at south)
    c_b = ψ̄_ext - cstar
    Ũ = min(0, max(-1, Δt / ΔX * c_b))

    τ = ifelse(ψ̄_ext <= 0, scheme.outflow_timescale, scheme.inflow_timescale)
    τ̃ = Δt / τ

    ψ_new = (ψ_b - Ũ * ψ_int + ψ̄_ext * τ̃) / (1 + τ̃ - Ũ)
    ψ_new = ifelse(iszero(τ), ψ̄_ext, ψ_new)

    @inbounds ρψ[iᴮ, jᴮ, kᴮ] = ρₖ * ψ_new

    return nothing
end

#####
##### Halo fill methods extending Oceananigans' _fill_*_halo! for PMABC
#####

# East boundary (Face in x)
@inline function OceananigansBC._fill_east_halo!(j, k, grid, ρψ, bc::PMABC, ::Tuple{Face, Any, Any},
                                                 clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i - 1, j, k)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    step_right_pma_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                             grid, ρψ, clock, model_fields, Δx, k)
    return nothing
end

# West boundary (Face in x)
@inline function OceananigansBC._fill_west_halo!(j, k, grid, ρψ, bc::PMABC, ::Tuple{Face, Any, Any},
                                                 clock, model_fields)
    boundary_indices = (1, j, k)
    boundary_adjacent_indices = (2, j, k)
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    step_left_pma_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                            grid, ρψ, clock, model_fields, Δx, k)
    return nothing
end

# North boundary (Face in y)
@inline function OceananigansBC._fill_north_halo!(i, k, grid, ρψ, bc::PMABC, ::Tuple{Any, Face, Any},
                                                  clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j - 1, k)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_pma_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                             grid, ρψ, clock, model_fields, Δy, k)
    return nothing
end

# South boundary (Face in y)
@inline function OceananigansBC._fill_south_halo!(i, k, grid, ρψ, bc::PMABC, ::Tuple{Any, Face, Any},
                                                  clock, model_fields)
    boundary_indices = (i, 1, k)
    boundary_adjacent_indices = (i, 2, k)
    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    step_left_pma_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                            grid, ρψ, clock, model_fields, Δy, k)
    return nothing
end

# Center-located fields (scalars: ρθ, ρq, etc.) at east/west boundaries
@inline function OceananigansBC._fill_east_halo!(j, k, grid, ρψ, bc::PMABC, ::Tuple{Center, Any, Any},
                                                 clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i - 1, j, k)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    step_right_pma_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                             grid, ρψ, clock, model_fields, Δx, k)
    return nothing
end

@inline function OceananigansBC._fill_west_halo!(j, k, grid, ρψ, bc::PMABC, ::Tuple{Center, Any, Any},
                                                 clock, model_fields)
    boundary_indices = (1, j, k)
    boundary_adjacent_indices = (2, j, k)
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    step_left_pma_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                            grid, ρψ, clock, model_fields, Δx, k)
    return nothing
end

# Center-located fields at north/south boundaries
@inline function OceananigansBC._fill_north_halo!(i, k, grid, ρψ, bc::PMABC, ::Tuple{Any, Center, Any},
                                                  clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j - 1, k)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_pma_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                             grid, ρψ, clock, model_fields, Δy, k)
    return nothing
end

@inline function OceananigansBC._fill_south_halo!(i, k, grid, ρψ, bc::PMABC, ::Tuple{Any, Center, Any},
                                                  clock, model_fields)
    boundary_indices = (i, 1, k)
    boundary_adjacent_indices = (i, 2, k)
    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    step_left_pma_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                            grid, ρψ, clock, model_fields, Δy, k)
    return nothing
end
