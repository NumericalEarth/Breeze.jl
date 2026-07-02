#####
##### Adaptive implicit vertical advection (AIVA) for the anelastic, mass-flux formulation
#####
##### Oceananigans' AIVA splits the vertical velocity per cell into an explicit part that
##### respects a target CFL and an implicit part solved with a first-order-upwind tridiagonal
##### update. The explicit part rides on the existing flux dispatch and needs no Breeze code
##### (see src/Advection.jl: `tracer_mass_flux_z` already routes through `_advective_tracer_flux_z`,
##### which dispatches on `time_discretization(scheme)`). What Breeze must add is the *implicit*
##### tridiagonal contribution, which — unlike Oceananigans' volume-conserving form — must be
##### density-weighted to stay consistent with the mass flux `∇·(ρ 𝐯 c)` used by `div_ρUc`.
#####
##### For a scalar density `q = ρc` advected vertically, the implicit upwind flux at face k+½ is
#####
#####   Fⁱ_{k+½} = Az_{k+½} ℑz(ρ)_{k+½} [ max(wⁱ_{k+½}, 0) q_k/ρ_k + min(wⁱ_{k+½}, 0) q_{k+1}/ρ_{k+1} ]
#####
##### (the `q/ρ` reconstructs the specific quantity `c` that is actually upwinded). With ρ ≡ 1
##### these reduce exactly to Oceananigans' `implicit_advection_*_diagonal`.
#####
##### TODO (upstream): Oceananigans' `implicit_advection_{upper,lower,diagonal}` hardcode the
##### volume-conserving form (`Δt V⁻¹ Az w`) with no hook for a face weight. Generalizing them to
##### accept an optional mass-flux/density weight (defaulting to unity) would let anelastic and
##### compressible models reuse them instead of the density-weighted copies below.

using Oceananigans: Field
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection, vertical_scheme, implicit_vertical_velocityᶜᶜᶠ
using Oceananigans.Fields: location
using Oceananigans.Grids: Center, Face, ZDirection, peripheral_node
using Oceananigans.Operators: Az, V⁻¹ᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.TimeSteppers: time_discretization
using Oceananigans.TurbulenceClosures:
    VerticallyImplicitDiffusionLowerDiagonal,
    VerticallyImplicitDiffusionDiagonal,
    VerticallyImplicitDiffusionUpperDiagonal,
    is_vertically_implicit,
    _ivd_lower_diagonal,
    _ivd_upper_diagonal,
    ivd_diagonal

const AIVA = AdaptiveImplicitVerticalAdvection

# Breeze-owned wrapper carrying the anelastic reference density into the implicit solve. Its sole
# purpose is to put a Breeze-owned type into the `get_coefficient` signatures below so those methods
# extend Oceananigans' `get_coefficient` without committing type piracy (every other argument type is
# Oceananigans-owned). It is unwrapped to the bare density field inside the coefficient functions.
struct AIVADensity{D}
    density :: D
end

Adapt.adapt_structure(to, ρ::AIVADensity) = AIVADensity(adapt(to, ρ.density))

#####
##### Density-weighted implicit advection diagonals (scalars at z-Centers)
#####

# Upper diagonal: coefficient of q_{k+1} in row k of (I - Δt L)
@inline function breeze_implicit_advection_upper_diagonal(i, j, k, grid, advection, w, ρ, Δt)
    scheme = vertical_scheme(advection)
    td     = time_discretization(scheme)
    wⁱ     = implicit_vertical_velocityᶜᶜᶠ(i, j, k+1, grid, scheme, td, w)
    Azᵏ⁺¹  = Az(i, j, k+1, grid, Center(), Center(), Face())
    ℑρᵏ⁺¹  = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ)
    ρᵏ⁺¹   = @inbounds ρ[i, j, k+1]
    V⁻¹    = V⁻¹ᶜᶜᶜ(i, j, k, grid)
    active = !peripheral_node(i, j, k+1, grid, Center(), Center(), Face())
    return Δt * V⁻¹ * Azᵏ⁺¹ * ℑρᵏ⁺¹ * min(wⁱ, zero(wⁱ)) / ρᵏ⁺¹ * active
end

# Lower diagonal: coefficient of q_{k-1} in row k. Uses k′ = k-1 (LinearAlgebra.Tridiagonal convention).
@inline function breeze_implicit_advection_lower_diagonal(i, j, k′, grid, advection, w, ρ, Δt)
    scheme = vertical_scheme(advection)
    td     = time_discretization(scheme)
    k      = k′ + 1
    wⁱ     = implicit_vertical_velocityᶜᶜᶠ(i, j, k, grid, scheme, td, w)
    Azᵏ    = Az(i, j, k, grid, Center(), Center(), Face())
    ℑρᵏ    = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
    ρᵏ⁻¹   = @inbounds ρ[i, j, k-1]
    V⁻¹    = V⁻¹ᶜᶜᶜ(i, j, k, grid)
    active = !peripheral_node(i, j, k′, grid, Center(), Center(), Center())
    return - Δt * V⁻¹ * Azᵏ * ℑρᵏ * max(wⁱ, zero(wⁱ)) / ρᵏ⁻¹ * active
end

# Diagonal: advection contribution to row k (added to the diffusion `ivd_diagonal`, which carries the identity)
@inline function breeze_implicit_advection_diagonal(i, j, k, grid, advection, w, ρ, Δt)
    scheme = vertical_scheme(advection)
    td     = time_discretization(scheme)
    wⁱ⁺    = implicit_vertical_velocityᶜᶜᶠ(i, j, k+1, grid, scheme, td, w)
    wⁱ⁻    = implicit_vertical_velocityᶜᶜᶠ(i, j, k,   grid, scheme, td, w)

    Az⁺ = Az(i, j, k+1, grid, Center(), Center(), Face())
    Az⁻ = Az(i, j, k,   grid, Center(), Center(), Face())
    ℑρ⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ)
    ℑρ⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ)
    ρᵏ  = @inbounds ρ[i, j, k]

    active⁺ = !peripheral_node(i, j, k+1, grid, Center(), Center(), Face())
    active⁻ = !peripheral_node(i, j, k,   grid, Center(), Center(), Face())

    V⁻¹ = V⁻¹ᶜᶜᶜ(i, j, k, grid)

    return Δt * V⁻¹ / ρᵏ * (Az⁺ * ℑρ⁺ * max(wⁱ⁺, zero(wⁱ⁺)) * active⁺ -
                            Az⁻ * ℑρ⁻ * min(wⁱ⁻, zero(wⁱ⁻)) * active⁻)
end

#####
##### get_coefficient seam: sum diffusion + density-weighted advection diagonals
#####
##### `solve!` forwards trailing args to `get_coefficient`. `breeze_implicit_step!` (below) passes
##### `(advection, w, ρ::AIVADensity)`; the `AIVADensity` wrapper is unwrapped here. These three
##### trailing args do not collide with Oceananigans' own two-arg `(advection::AIVA, w)` methods.
#####

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionUpperDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, advection::AIVA, w, ρ::AIVADensity)
    du_diff = _ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    du_adv  = breeze_implicit_advection_upper_diagonal(i, j, k, grid, advection, w, ρ.density, Δt)
    return du_diff + du_adv
end

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionLowerDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, advection::AIVA, w, ρ::AIVADensity)
    dl_diff = _ivd_lower_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    dl_adv  = breeze_implicit_advection_lower_diagonal(i, j, k, grid, advection, w, ρ.density, Δt)
    return dl_diff + dl_adv
end

@inline function Solvers.get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionDiagonal, p, ::ZDirection,
                                         clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, advection::AIVA, w, ρ::AIVADensity)
    d_diff = ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    d_adv  = breeze_implicit_advection_diagonal(i, j, k, grid, advection, w, ρ.density, Δt)
    return d_diff + d_adv
end

#####
##### Breeze implicit step that threads the reference density ρ through the solve
#####

# AIVA scalar: combined implicit (diffusion + density-weighted vertical advection) solve.
# Runs even when `closure` is `nothing` (AIVA without a vertically-implicit closure). This is a
# Breeze-owned function (not an extension of `TimeSteppers.implicit_step!`) so that wiring the
# density-weighted advection into the solve does not require pirating Oceananigans' `implicit_step!`.
function breeze_implicit_step!(field::Field,
                               implicit_solver::BatchedTridiagonalSolver,
                               closure, closure_fields, tracer_index,
                               clock, fields, Δt,
                               advection::AIVA, velocities, ρ)

    if closure isa Tuple
        N = length(closure)
        vi_closure        = Tuple(closure[n]        for n in 1:N if is_vertically_implicit(closure[n]))
        vi_closure_fields = Tuple(closure_fields[n] for n in 1:N if is_vertically_implicit(closure[n]))
    else
        vi_closure        = closure
        vi_closure_fields = closure_fields
    end

    LX, LY, LZ = location(field)
    return solve!(field, implicit_solver, field,
                  vi_closure, vi_closure_fields, tracer_index,
                  LX(), LY(), LZ(), Δt, clock, fields,
                  advection, velocities.w, AIVADensity(ρ))
end

#####
##### Per-field advection lookup for the SSP-RK3 substep
#####

# Momentum prognostics share the single `:momentum` scheme; scalars are keyed by name.
# AIVA is currently disallowed for momentum (validated in the constructor), so the momentum
# branch always returns a non-AIVA scheme and the implicit advection step is a no-op there.
@inline function field_advection_scheme(advection, name::Symbol)
    (name === :ρu || name === :ρv || name === :ρw) && return advection.momentum
    return haskey(advection, name) ? advection[name] : nothing
end
