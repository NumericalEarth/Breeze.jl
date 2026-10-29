#####
##### Terrain-following coordinate physics for compressible dynamics
#####
##### For terrain-following coordinates, three modifications are needed:
#####
##### 1. Contravariant vertical velocity w̃ replaces w in vertical transport
##### 2. Horizontal pressure gradient includes terrain correction
##### 3. Density tendency uses ρw̃ instead of ρw
#####
##### The contravariant vertical velocity is:
#####   w̃ = w - (∂z/∂x)_r · u - (∂z/∂y)_r · v
#####
##### The terrain-corrected horizontal pressure gradient is:
#####   (∂p/∂x)_z = (∂p/∂x)_r - (∂z/∂x)_r · (∂p/∂z)
#####
##### On MutableVerticalDiscretization grids, Oceananigans' generalized
##### derivatives (∂xᶠᶜᶜ, ∂yᶜᶠᶜ) already include the chain-rule correction,
##### so they compute (∂ϕ/∂x)_z directly. The SlopeOutsideInterpolation PG
##### delegates to these operators. The SlopeInsideInterpolation PG uses
##### δx/Δx (computational-coordinate derivative) with the slope multiplied
##### inside the interpolation stencil.
#####

using Oceananigans: architecture
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: Field, ZeroField, interior, set!
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, δxᶠᶜᶜ, δyᶜᶠᶜ, Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, ∂zᶜᶜᶠ, Δzᶜᶜᶜ, ∂zᶜᶜᶜ, Δz⁻¹ᶜᶜᶠ, ∂x_zᶠᶜᶜ, ∂y_zᶜᶠᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!, NormalFlowBoundaryCondition, FieldBoundaryConditions

using Breeze.TerrainFollowingDiscretization: TerrainMetrics, SlopeOutsideInterpolation,
                                              SlopeInsideInterpolation,
                                              TerrainFollowingGrid, ∂z∂x, ∂z∂y
using Breeze.Thermodynamics: evaluate_profile, integrate_exner_column!,
                             newton_hydrostatic_pressure, moist_reference_constants,
                             moist_hydrostatic_pressure
using Breeze.Solvers: FixedIterations

#####
##### Terrain-aware type alias
#####

const TerrainCompressibleDynamics = CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:TerrainMetrics}
const TerrainCompressibleModel = AtmosphereModel{<:TerrainCompressibleDynamics}

# Vertical transport on terrain-following grids goes through the tilted coordinate surfaces:
# momentum advects with the contravariant momentum ρw̃ (see `advecting_momentum` below), so the
# adaptive-implicit vertical-advection split partitions the contravariant velocity w̃.
@inline AtmosphereModels.advecting_vertical_velocity(dynamics::TerrainCompressibleDynamics, velocities) =
    dynamics.contravariant_vertical_velocity

"""
$(TYPEDEF)

Callable piecewise-linear vertical profile. `profile(z)` linearly interpolates `values`
against `heights` (both ordered bottom-to-top) and holds the nearest end value constant for
`z` below `heights[1]` or above `heights[end]`. Subtypes `Function` so it is picked up by
`evaluate_profile` wherever a `z`-dependent reference profile is expected.
"""
struct HorizontalMeanProfile{H, V} <: Function
    heights :: H
    values :: V
end

function (profile::HorizontalMeanProfile)(z)
    heights = profile.heights
    values = profile.values

    z ≤ heights[1] && return values[1]

    for k in 2:length(heights)
        if z ≤ heights[k]
            lower_height = heights[k-1]
            upper_height = heights[k]
            weight = (z - lower_height) / (upper_height - lower_height)
            return (1 - weight) * values[k-1] + weight * values[k]
        end
    end

    return values[end]
end

"""
$(TYPEDEF)

Adapter that makes a vertical profile — a `Number`, a callable `φ(z)`, or a
[`HorizontalMeanProfile`](@ref) — settable onto a `Field` of any dimensionality. `set!` calls it
with the field's node coordinates, which are `(x, z)` on a `Flat`-`y` grid, `(x, y, z)` in 3D and
`(λ, φ, z)` on a latitude-longitude grid; only the last of those — the physical height, which on a
terrain-following grid varies per column — is used. Subtypes `Function` so `set!` takes its
function-evaluation path (which evaluates on the host and transfers once).
"""
struct HeightProfile{P} <: Function
    profile :: P
end

@inline (hp::HeightProfile)(coordinates...) = evaluate_profile(hp.profile, last(coordinates))

@inline physical_heightᶜᶜᶜ(i, j, k, grid) = znode(i, j, k, grid, Center(), Center(), Center())

"""
$(TYPEDSIGNATURES)

Reduce a 3D `field` to a `HorizontalMeanProfile` of its horizontal mean **at constant physical
height**. On a terrain-following grid the cell-center height `znode(i, j, k, …)` varies with
`(i, j)`, so a plain per-`k` average would blend, e.g., valley-floor and mountain-top air at the
same computational level. Instead every column is first interpolated onto a common set of physical
heights and the mean is taken there, giving a genuine `θ̄(z)` — the horizontally-uniform reference
profile that WRF-style base states use, evaluated per column at its own terrain by
[`compute_terrain_reference_state!`](@ref).

The common heights are the columnwise minima of the cell-center heights at each level `k`, obtained
by a `minimum!` reduction: on a terrain-following grid these are the levels of the lowest-terrain
column, they increase strictly with `k`, and every column's level-`k` center lies at or above
them — so the per-column interpolation never extrapolates above a column's top.

A column whose terrain rises above a given height has no air there, and contributes nothing to that
height's mean: the interpolation kernel also fills an indicator field, and the mean is the ratio of
the two `sum!` reductions. Extending such columns downward instead (holding the surface value) would
bias the mean towards near-surface air by `~(z̄ − z_surface)·dφ/dz` at the lowest levels. The
lowest-terrain column contributes to every height, so no level is ever empty.
"""
function horizontal_mean_profile(field)
    grid = field.grid
    arch = architecture(grid)
    Nz = size(grid, 3)

    z̄ = Field{Nothing, Nothing, Center}(grid)
    minimum!(z̄, KernelFunctionOperation{Center, Center, Center}(physical_heightᶜᶜᶜ, grid))

    φ̂ = CenterField(grid)
    contributions = CenterField(grid)
    launch!(arch, grid, :xy, _interpolate_columns_to_heights!, φ̂, contributions, field, z̄, grid, Nz)

    φ_sum = Field{Nothing, Nothing, Center}(grid)
    contribution_sum = Field{Nothing, Nothing, Center}(grid)
    sum!(φ_sum, φ̂)
    sum!(contribution_sum, contributions)

    heights = vec(Array(interior(z̄)))
    values = vec(Array(interior(φ_sum))) ./ vec(Array(interior(contribution_sum)))

    return HorizontalMeanProfile(heights, values)
end

# Interpolate each column of `φ` onto the common heights `z̄`, piecewise-linearly in physical
# height, and record which columns contribute (those whose lowest cell center lies at or below the
# target height — the others are inside the terrain there). `m`, the highest cell center at or below
# z̄[k], never decreases with `k` (both the column's centers and z̄ increase), so it advances
# monotonically: O(Nz) per column. The `&` is deliberate (not `&&`): both operands are cheap, and a
# non-short-circuiting comparison keeps the loop free of nested branches on the GPU.
@kernel function _interpolate_columns_to_heights!(φ̂, contributions, φ, z̄, grid, Nz)
    i, j = @index(Global, NTuple)

    lowest_center = physical_heightᶜᶜᶜ(i, j, 1, grid)
    m = 1

    for k in 1:Nz
        @inbounds z★ = z̄[1, 1, k]

        while (m < Nz) & (physical_heightᶜᶜᶜ(i, j, min(m + 1, Nz), grid) ≤ z★)
            m += 1
        end

        m⁺ = min(m + 1, Nz)
        zᵐ = physical_heightᶜᶜᶜ(i, j, m, grid)
        zᵐ⁺ = physical_heightᶜᶜᶜ(i, j, m⁺, grid)

        # `max(…, eps)` guards the m = Nz case, where zᵐ⁺ = zᵐ and z★ = zᵐ, so the weight is 0.
        Δz = max(zᵐ⁺ - zᵐ, eps(zᵐ))
        w = clamp((z★ - zᵐ) / Δz, 0, 1)
        φ★ = (1 - w) * @inbounds(φ[i, j, m]) + w * @inbounds(φ[i, j, m⁺])

        contributes = z★ ≥ lowest_center
        @inbounds begin
            φ̂[i, j, k] = ifelse(contributes, φ★, 0)
            contributions[i, j, k] = ifelse(contributes, 1, 0)
        end
    end
end

#####
##### Compute contravariant vertical velocity and momentum
#####

"""
$(TYPEDSIGNATURES)

Compute the contravariant vertical velocity ``\\tilde{w}`` and
contravariant vertical momentum ``\\rho \\tilde{w}`` from the
Cartesian velocity and momentum fields.

The contravariant vertical velocity is the velocity component normal
to the terrain-following coordinate surfaces:

```math
\\tilde{w} = w - \\left(\\frac{\\partial z}{\\partial x}\\right)_r u
                    - \\left(\\frac{\\partial z}{\\partial y}\\right)_r v
```
"""
function compute_contravariant_velocity!(model::TerrainCompressibleModel; interior = true)
    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics
    w̃ = dynamics.contravariant_vertical_velocity
    ρw̃ = dynamics.contravariant_vertical_momentum

    # `interior = false`: the velocity launch of this update stored the interior (N10), only the fills are left.
    interior && launch!(arch, grid, :xyz,
                        _compute_contravariant_velocity!,
                        w̃, ρw̃,
                        grid, model.momentum, dynamics.dry_density)

    # The terrain kinematic BC (w̃ = 0 at the surface) is enforced declaratively:
    # `ρw` carries a NormalFlow bottom BC ρw|₁ = slopeₓ·ρu + slopeᵧ·ρv (see
    # `terrain_kinematic_bottom_ρw`), applied by `fill_halo_regions!(model.momentum, …)`
    # before this kernel runs. Because the slope/interpolation here matches that BC,
    # ρw̃|₁ = ρw|₁ − slope·ρu = 0 falls out automatically — no imperative bottom-face
    # zeroing (and no one-shot IC kernel) required.
    fill_halo_regions!(w̃)
    fill_halo_regions!(ρw̃)

    return nothing
end

@kernel function _compute_contravariant_velocity!(w̃, ρw̃, grid, momentum, density)
    i, j, k = @index(Global, NTuple)
    w̃_ijk, ρw̃_ijk = contravariant_vertical_transport(i, j, k, grid, momentum, density)
    @inbounds begin
        w̃[i, j, k] = w̃_ijk
        ρw̃[i, j, k] = ρw̃_ijk
    end
end

#####
##### N10 (handoff 08bcc1cf): the contravariant pair stored by the velocity launch
#####
##### The velocity launch reads the same ρu, ρv, ρw and ρᵈ. Between it and the contravariant launch only halos
##### are filled (ρᵈ and the thermodynamic prognostics again, by the fused thermodynamics pass); the contravariant
##### stencil reads ρᵈ at (i, j, k − 1 … k) for interior i, j only: interior cells and the bottom halo row,
##### which the second fill rewrites from the same interior with the same (non-lateral) bottom condition.
#####

"""
Whether the terrain velocity launch also stores the contravariant w̃ and ρw̃ (N10, handoff 08bcc1cf; `true` by
default, `BREEZE_VELOCITY_PASS_CONTRAVARIANT=false` at load: separate launch). Read at every update.
"""
const VELOCITY_PASS_CONTRAVARIANT = Ref(true)

struct ContravariantVerticalTransport{W, M}
    velocity :: W
    momentum :: M
end

Adapt.adapt_structure(to, c::ContravariantVerticalTransport) =
    ContravariantVerticalTransport(adapt(to, c.velocity), adapt(to, c.momentum))

# The velocity launch covers the Bounded boundary faces too; the pair is stored over 1:Nx × 1:Ny × 1:Nz only.
@inline function AtmosphereModels.compute_velocity_pass_auxiliaries!(i, j, k, c::ContravariantVerticalTransport,
                                                                     grid, dynamics, ρu, ρv, ρw)
    Nx, Ny, Nz = size(grid)
    if (i ≤ Nx) & (j ≤ Ny) & (k ≤ Nz)
        w̃_ijk, ρw̃_ijk = contravariant_vertical_transport(i, j, k, grid, (; ρu, ρv, ρw), dynamics.dry_density)
        @inbounds begin
            c.velocity[i, j, k] = w̃_ijk
            c.momentum[i, j, k] = ρw̃_ijk
        end
    end
    return nothing
end

@inline function contravariant_vertical_transport(i, j, k, grid, momentum, density)
    # Terrain slopes (∂z/∂x, ∂z/∂y)_r at (Center, Center, Face), from the grid
    # operator (which carries the formulation decay).
    slope_x = terrain_slope_x_ccf(i, j, k, grid)
    slope_y = terrain_slope_y_ccf(i, j, k, grid)

    # Momentum interpolated to (Center, Center, Face).
    # ρu is at (Face, Center, Center) → ℑx then ℑz to (Center, Center, Face)
    ρu_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, momentum.ρu)
    ρv_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, momentum.ρv)
    @inbounds ρw_ccf = momentum.ρw[i, j, k]

    # Contravariant vertical momentum (primary quantity)
    ρw̃_ijk = ρw_ccf - slope_x * ρu_ccf - slope_y * ρv_ccf

    # Diagnose velocity from momentum for discrete consistency: ρ_face · w̃ ≡ ρw̃
    ρ_ccf = ℑzᵃᵃᶠ(i, j, k, grid, density)
    w̃_ijk = ρw̃_ijk / ρ_ccf

    return w̃_ijk, ρw̃_ijk
end

#####
##### Transport velocity/momentum interface for terrain-following coordinates
#####

function AtmosphereModels.transport_velocities(model::TerrainCompressibleModel)
    w̃ = model.dynamics.contravariant_vertical_velocity
    u = model.velocities.u
    v = model.velocities.v
    return (; u, v, w=w̃)
end

function outer_step_start_transport_velocities(model::TerrainCompressibleModel)
    w̃ = model.dynamics.contravariant_vertical_velocity
    u = model.velocities.u
    v = model.velocities.v
    return (; u, v, w=w̃)
end

function AtmosphereModels.advecting_momentum(model::TerrainCompressibleModel)
    ρw̃ = model.dynamics.contravariant_vertical_momentum
    ρu = model.momentum.ρu
    ρv = model.momentum.ρv
    return (; ρu, ρv, ρw=ρw̃)
end

#####
##### Terrain-specialized acoustic substep helpers
#####

function initialize_vertical_momentum_perturbation!(substepper, model::TerrainCompressibleModel, Uᴸ_outer)
    grid = model.grid
    arch = architecture(grid)
    if LOAD_STAGE_CONTRAVARIANT_MOMENTUM[]
        # The stage-entry ρw̃ is the one `update_state!` stored (port of handoff N11 84d73a82).
        launch!(arch, grid, :xyz, _initialize_terrain_vertical_momentum_perturbation_loaded!,
                substepper.momentum_perturbation.w,
                Uᴸ_outer.ρu, Uᴸ_outer.ρv, Uᴸ_outer.ρw,
                model.dynamics.contravariant_vertical_momentum,
                grid)
    else
        launch!(arch, grid, :xyz, _initialize_terrain_vertical_momentum_perturbation!,
                substepper.momentum_perturbation.w,
                Uᴸ_outer.ρu, Uᴸ_outer.ρv, Uᴸ_outer.ρw,
                model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
                grid)
    end
    return nothing
end

stage_transport_vertical_momentum(model::TerrainCompressibleModel) =
    LOAD_STAGE_CONTRAVARIANT_MOMENTUM[] ? (model.dynamics.contravariant_vertical_momentum, nothing) :
                                          (model.momentum.ρw, model.dynamics)

@kernel function _initialize_terrain_vertical_momentum_perturbation_loaded!(ρw̃′, ρu_outer, ρv_outer, ρw_outer,
                                                                            ρw̃_stage, grid)
    i, j, k = @index(Global, NTuple)
    ρw̃_outer = terrain_vertical_transport_momentum(i, j, k, grid,
                                                    ρu_outer, ρv_outer, ρw_outer)
    @inbounds ρw̃′[i, j, k] = ρw̃_outer - ρw̃_stage[i, j, k]
end

# The fused stage initialization's ρw̃′ rewind: the expression of `_initialize_terrain_vertical_momentum_perturbation_loaded!`.
struct TerrainVerticalRewind{U, V, W, S, G}
    ρu_outer :: U
    ρv_outer :: V
    ρw_outer :: W
    ρw̃_stage :: S
    grid :: G
end

Adapt.adapt_structure(to, r::TerrainVerticalRewind) =
    TerrainVerticalRewind(adapt(to, r.ρu_outer), adapt(to, r.ρv_outer), adapt(to, r.ρw_outer),
                          adapt(to, r.ρw̃_stage), adapt(to, r.grid))

@inline function rewind_value(r::TerrainVerticalRewind, i, j, k)
    ρw̃_outer = terrain_vertical_transport_momentum(i, j, k, r.grid, r.ρu_outer, r.ρv_outer, r.ρw_outer)
    return @inbounds ρw̃_outer - r.ρw̃_stage[i, j, k]
end

vertical_momentum_rewind(model::TerrainCompressibleModel, Uᴸ_outer) =
    LOAD_STAGE_CONTRAVARIANT_MOMENTUM[] ?
        TerrainVerticalRewind(Uᴸ_outer.ρu, Uᴸ_outer.ρv, Uᴸ_outer.ρw,
                              model.dynamics.contravariant_vertical_momentum, model.grid) : nothing

@kernel function _initialize_terrain_vertical_momentum_perturbation!(ρw̃′,
                                                                     ρu_outer, ρv_outer, ρw_outer,
                                                                     ρu_stage, ρv_stage, ρw_stage,
                                                                     grid)
    i, j, k = @index(Global, NTuple)
    ρw̃_outer = terrain_vertical_transport_momentum(i, j, k, grid,
                                                    ρu_outer, ρv_outer, ρw_outer)
    ρw̃_stage = terrain_vertical_transport_momentum(i, j, k, grid,
                                                    ρu_stage, ρv_stage, ρw_stage)
    @inbounds ρw̃′[i, j, k] = ρw̃_outer - ρw̃_stage
end

@inline function ∇ˣp′(i, j, k, grid,
                                                dynamics::TerrainCompressibleDynamics,
                                                ρθ′, Πᴸ, γRᵐᴸ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_x_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                  stencil, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function ∇ʸp′(i, j, k, grid,
                                                dynamics::TerrainCompressibleDynamics,
                                                ρθ′, Πᴸ, γRᵐᴸ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_y_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                  stencil, ρθ′, Πᴸ, γRᵐᴸ)
end

# On a TerrainFollowingVerticalDiscretization grid the coordinate owns the
# slope: take (∂z/∂x)_r from the grid operator (which carries the formulation's
# decay — linear for LinearDecay, sinh for TwoLevelDecay) and interpolate the x-face
# value to (Center, Center) at the z-face.
# Use Oceananigans' stagger interpolators (`ℑxᶜᵃᵃ`/`ℑyᵃᶜᵃ`) instead of a
# manual `(idx, idx+1)/2` average: those handle Flat dimensions correctly.
# The naive form reads `∂z∂y(i, j+1, …)` which is out-of-bounds on a Flat-y
# grid (Ny = 1, no y halo) and returns uninitialised memory — which then
# propagates as NaN through `compute_contravariant_velocity!` and the rest
# of the substep. `ℑyᵃᶜᵃ` on a Flat-y grid collapses to a no-op.
@inline terrain_slope_x_ccf(i, j, k, grid::TerrainFollowingGrid) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂z∂x, Face())

@inline terrain_slope_y_ccf(i, j, k, grid::TerrainFollowingGrid) =
    ℑyᵃᶜᵃ(i, j, k, grid, ∂z∂y, Face())

#####
##### Bottom kinematic boundary condition for ρw (declarative, dispatch-selected)
#####
##### "No flow through the terrain surface" is, in contravariant form,
#####   ρw̃ = ρw - slopeₓ·ρu - slopeᵧ·ρv = 0   at the bottom face,
##### i.e. the Cartesian vertical momentum follows the terrain:
#####   ρw|₁ = slopeₓ·ρu + slopeᵧ·ρv .
##### We impose this as a `NormalFlowBoundaryCondition` on `ρw` whose value is computed
##### from the live momentum at fill time. `fill_halo_regions!(model.momentum,
##### clock, fields(model))` runs every `update_state!`/substep, so the surface
##### stays kinematically balanced at the IC and throughout the run — replacing the
##### previous one-shot IC kernel and the imperative bottom-face zeroing. The slope
##### and interpolation match `compute_contravariant_velocity!`, so ρw̃|₁ = 0 to
##### machine precision. The perturbation fields keep their impenetrable defaults
##### (the substepper deliberately does not inherit this BC; see the §"acoustic
##### substep" note on issue #716).

@inline function terrain_kinematic_bottom_ρw(i, j, grid, clock, model_fields)
    slope_x = terrain_slope_x_ccf(i, j, 1, grid)
    slope_y = terrain_slope_y_ccf(i, j, 1, grid)
    ρu_ccf  = ℑzᵃᵃᶠ(i, j, 1, grid, ℑxᶜᵃᵃ, model_fields.ρu)
    ρv_ccf  = ℑzᵃᵃᶠ(i, j, 1, grid, ℑyᵃᶜᵃ, model_fields.ρv)
    return slope_x * ρu_ccf + slope_y * ρv_ccf
end

# Dispatch the ρw bottom boundary condition on the grid type: terrain-following
# grids get the kinematic normal-flow BC; every other grid keeps the BCs it was given
# (impenetrable bottom by default). The terrain BC sets only the bottom; the top
# (impenetrable lid) and horizontal sides take their usual regularised defaults.
terrain_ρw_boundary_conditions(grid, ρw_bcs) = ρw_bcs
terrain_ρw_boundary_conditions(::TerrainFollowingGrid, ρw_bcs) =
    FieldBoundaryConditions(; west = ρw_bcs.west, east = ρw_bcs.east,
                              south = ρw_bcs.south, north = ρw_bcs.north,
                              bottom = NormalFlowBoundaryCondition(terrain_kinematic_bottom_ρw; discrete_form = true),
                              top = ρw_bcs.top, immersed = ρw_bcs.immersed)

@inline function terrain_horizontal_pressure_gradient_correction(i, j, k, grid, dynamics)
    slope_x = terrain_slope_x_ccf(i, j, k, grid)
    slope_y = terrain_slope_y_ccf(i, j, k, grid)
    ∂x_p_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, AtmosphereModels.x_pressure_gradient, dynamics)
    ∂y_p_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, AtmosphereModels.y_pressure_gradient, dynamics)
    return slope_x * ∂x_p_ccf + slope_y * ∂y_p_ccf
end

@inline function terrain_horizontal_linearized_pressure_gradient_correction(i, j, k, grid,
                                                                            dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    slope_x = terrain_slope_x_ccf(i, j, k, grid)
    slope_y = terrain_slope_y_ccf(i, j, k, grid)
    ∂x_p′_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ∇ˣp′,
                       dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    ∂y_p′_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, ∇ʸp′,
                       dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    return slope_x * ∂x_p′_ccf + slope_y * ∂y_p′_ccf
end

# The vertical gradient minus the horizontal slope correction slopeₓ·∂ₓ(Cᴸ(ρθ)′) + slopeᵧ·∂ᵧ(Cᴸ(ρθ)′),
# since ρw̃ = ρw − slopeₓ·ρu − slopeᵧ·ρv.
@inline function ∇ᶻp′(i, j, k, grid,
                      dynamics::TerrainCompressibleDynamics,
                      ρθ′, Πᴸ, γRᵐᴸ)
    ∂z_p′ = ∂zᶜᶜᶠ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
    correction = terrain_horizontal_linearized_pressure_gradient_correction(i, j, k, grid,
                                                                            dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    return ∂z_p′ - correction
end

#####
##### Stored horizontal pressure gradients (`TerrainPressureGradients`; port of handoff N3 5255302b)
#####

stored_pressure_gradients(grid::TerrainFollowingGrid) = TerrainPressureGradients(grid)

# slopeₓ ℑ(∂x ϕ) + slopeᵧ ℑ(∂y ϕ) at a w-point from stored u- and v-face gradients: the expression of
# `terrain_horizontal_linearized_pressure_gradient_correction` with each gradient evaluation replaced by
# a load (Oceananigans' array and function interpolations are the same `FT(0.5) * (a + b)`).
@inline function stored_horizontal_gradient_correction(i, j, k, grid, ∂xϕ, ∂yϕ)
    slope_x = terrain_slope_x_ccf(i, j, k, grid)
    slope_y = terrain_slope_y_ccf(i, j, k, grid)
    ∂xϕ_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ∂xϕ)
    ∂yϕ_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, ∂yϕ)
    return slope_x * ∂xϕ_ccf + slope_y * ∂yϕ_ccf
end

@inline function ∇ᶻp′ˢ⁻(P::TerrainPressureGradients, i, j, k, grid,
                        dynamics::TerrainCompressibleDynamics, ρθ′, Πᴸ, γRᵐᴸ)
    ∂z_p′ = ∂zᶜᶜᶠ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
    correction = stored_horizontal_gradient_correction(i, j, k, grid, P.perturbation_x, P.perturbation_y)
    return ∂z_p′ - correction
end

# ∂pᴸ loaded from the per-stage store, ∂p′ evaluated and stored for the vertical rhs of this substep.
@inline function horizontal_pressure_gradients!(P::TerrainPressureGradients, i, j, k, grid,
                                                dynamics::TerrainCompressibleDynamics, ρθ′, Πᴸ, γRᵐᴸ)
    ∂x_p′ = ∇ˣp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    ∂y_p′ = ∇ʸp′(i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    @inbounds begin
        P.perturbation_x[i, j, k] = ∂x_p′
        P.perturbation_y[i, j, k] = ∂y_p′
    end
    store_last_face_perturbation_gradients!(P, i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    @inbounds begin
        ∂x_p = P.linearization_x[i, j, k] + ∂x_p′
        ∂y_p = P.linearization_y[i, j, k] + ∂y_p′
    end
    return ∂x_p, ∂y_p
end

# The rhs interpolation at i = Nx (j = Ny) reads face Nx+1 (Ny+1), outside the explicit step's range:
# the last cell's thread evaluates it too (predicated, so only those warps pay).
@inline function store_last_face_perturbation_gradients!(P, i, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    Nx, Ny = size(grid, 1), size(grid, 2)
    if has_last_face(topology(grid, 1)) & (i == Nx)
        @inbounds P.perturbation_x[Nx + 1, j, k] = ∇ˣp′(Nx + 1, j, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    end
    if has_last_face(topology(grid, 2)) & (j == Ny)
        @inbounds P.perturbation_y[i, Ny + 1, k] = ∇ʸp′(i, Ny + 1, k, grid, dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    end
    return nothing
end

#####
##### The vertical rhs carried up the column (`_build_vertical_rhs_carried!`, port of handoff bc1c3e6e)
#####
##### ∇ᶻp′((ρθ)′★) at face k = Dz(i, j, k) − [slopeₓ ½(Fx(k−1) + Fx(k)) + slopeᵧ ½(Fy(k−1) + Fy(k))] with
##### p★ = γRᵐᴸ Πᴸ (ρθ)′★, Dz(c, f) = (p★(c, f) − p★(c, f − 1)) Δz⁻¹ᶜᶜᶠ(c, f), Fx(k) = ½(∇ˣ(i) + ∇ˣ(i + 1)) and
##### ∇ˣ(i′) = (p★(i′) − p★(i′ − 1)) Δx⁻¹ − ∂x_z ½(Ax(k) + Ax(k + 1)), Ax(f) = ½(Dz(i′ − 1, f) + Dz(i′, f))
##### (`terrain_amg_operators.jl`, Oceananigans' `FT(0.5) * (a + b)` interpolations). Per face the sweep evaluates
##### p★ and Dz of the five columns at the next level, the gradients of level k, and carries the rest.

carries_column_stencils(grid) = (topology(grid, 1) !== Flat) & (topology(grid, 2) !== Flat)

carries_vertical_rhs(dynamics::TerrainCompressibleDynamics, ::TerrainPressureGradients, grid) =
    (dynamics.terrain_metrics.pressure_gradient_stencil isa SlopeOutsideInterpolation) & carries_column_stencils(grid)

@inline function five_column_differences(i, j, f, grid, p⁻, ρθ′★, Πᴸ, γRᵐᴸ)
    pW⁻, pC⁻, pE⁻, pS⁻, pN⁻ = p⁻
    pW = δpᴸ(i - 1, j, f, grid, ρθ′★, Πᴸ, γRᵐᴸ)
    pC = δpᴸ(i,     j, f, grid, ρθ′★, Πᴸ, γRᵐᴸ)
    pE = δpᴸ(i + 1, j, f, grid, ρθ′★, Πᴸ, γRᵐᴸ)
    pS = δpᴸ(i, j - 1, f, grid, ρθ′★, Πᴸ, γRᵐᴸ)
    pN = δpᴸ(i, j + 1, f, grid, ρθ′★, Πᴸ, γRᵐᴸ)
    DW = (pW - pW⁻) * Δz⁻¹ᶜᶜᶠ(i - 1, j, f, grid)
    DC = (pC - pC⁻) * Δz⁻¹ᶜᶜᶠ(i,     j, f, grid)
    DE = (pE - pE⁻) * Δz⁻¹ᶜᶜᶠ(i + 1, j, f, grid)
    DS = (pS - pS⁻) * Δz⁻¹ᶜᶜᶠ(i, j - 1, f, grid)
    DN = (pN - pN⁻) * Δz⁻¹ᶜᶜᶠ(i, j + 1, f, grid)
    return (pW, pC, pE, pS, pN), (DW, DC, DE, DS, DN)
end

@inline function face_difference_averages(grid, D)
    FT = eltype(grid)
    DW, DC, DE, DS, DN = D
    return (FT(0.5) * (DW + DC), FT(0.5) * (DC + DE), FT(0.5) * (DS + DC), FT(0.5) * (DC + DN))
end

@inline function centered_gradient_averages(i, j, k, grid, p, A, A⁺)
    FT = eltype(grid)
    pW, pC, pE, pS, pN = p
    AxW, AxE, AyS, AyN = A
    AxW⁺, AxE⁺, AyS⁺, AyN⁺ = A⁺
    ∇xW = (pC - pW) * Δx⁻¹ᶠᶜᶜ(i,     j, k, grid) - ∂x_zᶠᶜᶜ(i,     j, k, grid) * (FT(0.5) * (AxW + AxW⁺))
    ∇xE = (pE - pC) * Δx⁻¹ᶠᶜᶜ(i + 1, j, k, grid) - ∂x_zᶠᶜᶜ(i + 1, j, k, grid) * (FT(0.5) * (AxE + AxE⁺))
    ∇yS = (pC - pS) * Δy⁻¹ᶜᶠᶜ(i, j,     k, grid) - ∂y_zᶜᶠᶜ(i, j,     k, grid) * (FT(0.5) * (AyS + AyS⁺))
    ∇yN = (pN - pC) * Δy⁻¹ᶜᶠᶜ(i, j + 1, k, grid) - ∂y_zᶜᶠᶜ(i, j + 1, k, grid) * (FT(0.5) * (AyN + AyN⁺))
    return FT(0.5) * (∇xW + ∇xE), FT(0.5) * (∇yS + ∇yN)
end

# The carried values at the start of face 2 (levels 0, 1, 2 of p★, as the 3-D kernel's face 2 reads them).
@inline function carried_vertical_rhs_state(r, i, j)
    grid = r.grid
    FT = eltype(grid)
    P = r.P
    @inbounds begin
        z = zero(FT)
        p₀, _  = five_column_differences(i, j, 0, grid, (z, z, z, z, z), r.ρθ′★, r.Πᴸ, r.γRᵐᴸ)
        p₁, D₁ = five_column_differences(i, j, 1, grid, p₀, r.ρθ′★, r.Πᴸ, r.γRᵐᴸ)
        p₂, D₂ = five_column_differences(i, j, 2, grid, p₁, r.ρθ′★, r.Πᴸ, r.γRᵐᴸ)
        A₁ = face_difference_averages(grid, D₁)
        A₂ = face_difference_averages(grid, D₂)
        Fx₁, Fy₁ = centered_gradient_averages(i, j, 1, grid, p₁, A₁, A₂)
        pˢ₁ = δpᴸ(i, j, 1, grid, r.ρθ′, r.Πᴸ, r.γRᵐᴸ)
        SX₁ = FT(0.5) * (P.perturbation_x[i, j, 1] + P.perturbation_x[i + 1, j, 1])
        SY₁ = FT(0.5) * (P.perturbation_y[i, j, 1] + P.perturbation_y[i, j + 1, 1])
        return (p = p₂, A = A₂, DzC = D₂[2], Fx = Fx₁, Fy = Fy₁, pˢ = pˢ₁, SX = SX₁, SY = SY₁,
                ρ★ = r.ρ′★[i, j, 1], ρˢ = r.ρ′[i, j, 1])
    end
end

# rhs(k) — the value `vertical_rhs` returns — and the carried values at the start of face k + 1.
@inline function carried_vertical_rhs(r, state, i, j, k)
    grid = r.grid
    FT = eltype(grid)
    P = r.P
    Δτ, δτᵐ⁺, δτˢ⁻ = r.Δτ, r.δτᵐ⁺, r.δτˢ⁻
    @inbounds begin
        p⁺, D⁺ = five_column_differences(i, j, k + 1, grid, state.p, r.ρθ′★, r.Πᴸ, r.γRᵐᴸ)
        A⁺ = face_difference_averages(grid, D⁺)
        Fx, Fy = centered_gradient_averages(i, j, k, grid, state.p, state.A, A⁺)

        slope_x = terrain_slope_x_ccf(i, j, k, grid)
        slope_y = terrain_slope_y_ccf(i, j, k, grid)

        # ∇ᶻp′ of (ρθ)′★
        ∂x_p′_ccf = FT(0.5) * (state.Fx + Fx)
        ∂y_p′_ccf = FT(0.5) * (state.Fy + Fy)
        correction = slope_x * ∂x_p′_ccf + slope_y * ∂y_p′_ccf
        ∂r_p′★ = state.DzC - correction

        # ∇ᶻp′ˢ⁻ of the old (ρθ)′ with the stored perturbation gradients
        pˢ = δpᴸ(i, j, k, grid, r.ρθ′, r.Πᴸ, r.γRᵐᴸ)
        ∂z_pˢ = (pˢ - state.pˢ) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)
        SX = FT(0.5) * (P.perturbation_x[i, j, k] + P.perturbation_x[i + 1, j, k])
        SY = FT(0.5) * (P.perturbation_y[i, j, k] + P.perturbation_y[i, j + 1, k])
        stored = slope_x * (FT(0.5) * (state.SX + SX)) + slope_y * (FT(0.5) * (state.SY + SY))
        ∂r_p′ˢ⁻ = ∂z_pˢ - stored

        Gρwᵖ = δτˢ⁻ * ∂r_p′ˢ⁻ + δτᵐ⁺ * ∂r_p′★

        ρ★ = r.ρ′★[i, j, k]
        ρˢ = r.ρ′[i, j, k]
        ρ′ᶜᶜᶠ★  = FT(0.5) * (state.ρ★ + ρ★)
        ρ′ᶜᶜᶠˢ⁻ = FT(0.5) * (state.ρˢ + ρˢ)
        Gρwᵇ = r.g * (δτˢ⁻ * ρ′ᶜᶜᶠˢ⁻ + δτᵐ⁺ * ρ′ᶜᶜᶠ★)

        ∂z²_ρw′ˢ⁻ = ∂zᶜᶜᶠ(i, j, k, grid, ∂zᶜᶜᶜ, r.ρw′)
        Gρwᵈ = - r.dˢ⁻ * ∂z²_ρw′ˢ⁻

        Gρwˢ = sponge_rhs(i, j, k, grid, r.sponge, δτˢ⁻, r.ρw′)

        rhs = r.ρw′[i, j, k] + Δτ * r.fw * r.Gˢρw[i, j, k] - Gρwᵖ - Gρwᵇ - Gρwᵈ - Gρwˢ

        next = (; p = p⁺, A = A⁺, DzC = D⁺[2], Fx, Fy, pˢ, SX, SY, ρ★, ρˢ)
        return rhs, next
    end
end

@inline has_last_face(::Type{Flat}) = false
@inline has_last_face(direction_topology) = true

interpolated_face_indices(::Type{Flat}, N) = 1:1
interpolated_face_indices(direction_topology, N) = 1:N+1

# Store this stage's frozen ∂x pᴸ, ∂y pᴸ over every face the assembly and the explicit steps read
# (levels 0:Nz: the assembly's ℑzᵃᵃᶠ at k = 1 reads level 0). The extra v-face column i = Nx+1 and
# u-face row j = Ny+1 land in halos nothing reads.
function store_linearization_pressure_gradients!(P::TerrainPressureGradients, model)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    TX, TY, _ = topology(grid)
    if CARRIED_LINEARIZATION_GRADIENTS[] & carries_vertical_rhs(model.dynamics, P, grid)
        # one thread per face column (i, j) of the same range, levels 0:Nz carried (S7)
        launch!(architecture(grid), grid, KernelParameters(1:Nx+1, 1:Ny+1), _store_linearization_pressure_gradients_carried!,
                P.linearization_x, P.linearization_y, grid, model.dynamics)
    else
        indices = KernelParameters(interpolated_face_indices(TX, Nx), interpolated_face_indices(TY, Ny), 0:Nz)
        launch!(architecture(grid), grid, indices, _store_linearization_pressure_gradients!,
                P.linearization_x, P.linearization_y, grid, model.dynamics)
    end
    return nothing
end

#####
##### D3 S7 (handoff 513ceeee, store part): the per-stage ∂x pᴸ, ∂y pᴸ store as a column launch with the vertical
##### stencil carried. ∂xᶠᶜᶜ(q) at (i, k) = (q(i) − q(i − 1))·Δx⁻¹ᶠᶜᶜ − ∂x_zᶠᶜᶜ·½(Ax(k) + Ax(k + 1)),
##### Ax(f) = ½(Dz(i − 1, f) + Dz(i, f)), Dz(c, f) = (q(c, f) − q(c, f − 1))·Δz⁻¹ᶜᶜᶠ(c, f) (`terrain_amg_operators.jl`),
##### and ∂yᶜᶠᶜ likewise with the columns (i, j − 1), (i, j). Per level the three columns W, C, S are evaluated at the
##### next level; the face-k averages and level-k values come from the level below. Same expressions, same operands.
##### Applies where the carried vertical rhs does (`carries_vertical_rhs`: SlopeOutsideInterpolation, no Flat).
#####

"""
Whether the per-stage store of ∂x pᴸ, ∂y pᴸ runs as the carried column launch (`true` by default;
`BREEZE_CARRIED_LINEARIZATION_GRADIENTS=false` at load: the 3-D launch). Bitwise identical.
"""
const CARRIED_LINEARIZATION_GRADIENTS = Ref(true)

@inline function three_column_face_averages(i, j, f, grid, q⁻, q, a, b)
    FT = eltype(grid)
    qW⁻, qC⁻, qS⁻ = q⁻
    qW = q(i - 1, j, f, grid, a, b)
    qC = q(i,     j, f, grid, a, b)
    qS = q(i, j - 1, f, grid, a, b)
    DW = (qW - qW⁻) * Δz⁻¹ᶜᶜᶠ(i - 1, j, f, grid)
    DC = (qC - qC⁻) * Δz⁻¹ᶜᶜᶠ(i,     j, f, grid)
    DS = (qS - qS⁻) * Δz⁻¹ᶜᶜᶠ(i, j - 1, f, grid)
    return (qW, qC, qS), (FT(0.5) * (DW + DC), FT(0.5) * (DS + DC))
end

@inline function carried_face_gradients(i, j, k, grid, v, A, A⁺)
    FT = eltype(grid)
    qW, qC, qS = v
    Ax, Ay = A
    Ax⁺, Ay⁺ = A⁺
    ∂x_q = (qC - qW) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid) - ∂x_zᶠᶜᶜ(i, j, k, grid) * (FT(0.5) * (Ax + Ax⁺))
    ∂y_q = (qC - qS) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid) - ∂y_zᶜᶠᶜ(i, j, k, grid) * (FT(0.5) * (Ay + Ay⁺))
    return ∂x_q, ∂y_q
end

# The pressure `x_pressure_gradient` differentiates with the SlopeOutsideInterpolation stencil: p − pᵣ, or p.
@inline linearization_pressure(i, j, k, grid, p, pᵣ) = perturbation_pressure(i, j, k, grid, p, pᵣ)
@inline linearization_pressure(i, j, k, grid, p, ::Nothing) = @inbounds p[i, j, k]

@kernel function _store_linearization_pressure_gradients_carried!(∂x_pᴸ, ∂y_pᴸ, grid, dynamics)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    FT = eltype(grid)
    p = dynamics.pressure
    pᵣ = reference_pressure_field(dynamics.reference_state)
    z = zero(FT)
    @inbounds begin
        v₋, _ = three_column_face_averages(i, j, -1, grid, (z, z, z), linearization_pressure, p, pᵣ)  # (averages unused)
        v, A = three_column_face_averages(i, j, 0, grid, v₋, linearization_pressure, p, pᵣ)            # level 0, face 0
        for k = 0:Nz
            v⁺, A⁺ = three_column_face_averages(i, j, k + 1, grid, v, linearization_pressure, p, pᵣ)
            ∂x_q, ∂y_q = carried_face_gradients(i, j, k, grid, v, A, A⁺)
            ∂x_pᴸ[i, j, k] = ∂x_q
            ∂y_pᴸ[i, j, k] = ∂y_q
            v, A = v⁺, A⁺
        end
    end
end

@kernel function _store_linearization_pressure_gradients!(∂x_pᴸ, ∂y_pᴸ, grid, dynamics)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ∂x_pᴸ[i, j, k] = AtmosphereModels.x_pressure_gradient(i, j, k, grid, dynamics)
        ∂y_pᴸ[i, j, k] = AtmosphereModels.y_pressure_gradient(i, j, k, grid, dynamics)
    end
end

@inline function terrain_x_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                        ::SlopeOutsideInterpolation,
                                                        ρθ′, Πᴸ, γRᵐᴸ)
    return ∂xᶠᶜᶜ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function terrain_y_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                        ::SlopeOutsideInterpolation,
                                                        ρθ′, Πᴸ, γRᵐᴸ)
    return ∂yᶜᶠᶜ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function slope_x_times_∂z_linearized_pressure(i, j, k, grid::TerrainFollowingGrid, ρθ′, Πᴸ, γRᵐᴸ)
    slope = terrain_slope_x_ccf(i, j, k, grid)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function slope_y_times_∂z_linearized_pressure(i, j, k, grid::TerrainFollowingGrid, ρθ′, Πᴸ, γRᵐᴸ)
    slope = terrain_slope_y_ccf(i, j, k, grid)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function terrain_x_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                        ::SlopeInsideInterpolation,
                                                        ρθ′, Πᴸ, γRᵐᴸ)
    ∂x_p′ = δxᶠᶜᶜ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ) *
            Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_linearized_pressure,
                       ρθ′, Πᴸ, γRᵐᴸ)
    return ∂x_p′ - correction
end

@inline function terrain_y_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                        ::SlopeInsideInterpolation,
                                                        ρθ′, Πᴸ, γRᵐᴸ)
    ∂y_p′ = δyᶜᶠᶜ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ) *
            Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_linearized_pressure,
                       ρθ′, Πᴸ, γRᵐᴸ)
    return ∂y_p′ - correction
end

@inline function terrain_vertical_transport_momentum(i, j, k, grid::TerrainFollowingGrid,
                                                     ρu, ρv, ρw)
    slope_x = terrain_slope_x_ccf(i, j, k, grid)
    slope_y = terrain_slope_y_ccf(i, j, k, grid)

    ρu_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ρu)
    ρv_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, ρv)
    @inbounds ρw_ccf = ρw[i, j, k]

    return ρw_ccf - slope_x * ρu_ccf - slope_y * ρv_ccf
end

@inline function transport_ρw(i, j, k, grid,
                                                            dynamics::TerrainCompressibleDynamics,
                                                            ρu_stage, ρv_stage, ρw_stage)
    return terrain_vertical_transport_momentum(i, j, k, grid,
                                               ρu_stage, ρv_stage, ρw_stage)
end

# Convert the substepped contravariant momentum ρw̃ᵐ⁺ = ρw̃ᴸ + ρw̃′ back to Cartesian ρw.
# The stage-base horizontal momenta cancel algebraically, since ℑ is linear:
#
#   ρw = (ρw̃ᴸ + ρw̃′) + slopeₓ ℑ(ρuᴸ + ρu′) + slopeᵧ ℑ(ρvᴸ + ρv′)
#      = (ρwᴸ − slopeₓ ℑρuᴸ − slopeᵧ ℑρvᴸ) + ρw̃′ + slopeₓ ℑ(ρuᴸ + ρu′) + slopeᵧ ℑ(ρvᴸ + ρv′)
#      = ρwᴸ + ρw̃′ + slopeₓ ℑρu′ + slopeᵧ ℑρv′
#
# with ρw̃ᴸ = `terrain_vertical_transport_momentum` of the same stage-entry momenta that
# `_initialize_terrain_vertical_momentum_perturbation!` used as the base for ρw̃′. The base ρw
# arrives as a value rather than a field because `_recover_full_state!` writes the same momentum
# fields it reads as the stage base; keeping them out of this signature keeps them out of reach
# of a stencil.
@inline function acoustic_recovered_vertical_momentum(i, j, k, grid,
                                                      dynamics::TerrainCompressibleDynamics,
                                                      ρwᴸ_ccf, ρu′, ρv′, ρw̃′)
    slope_x = terrain_slope_x_ccf(i, j, k, grid)
    slope_y = terrain_slope_y_ccf(i, j, k, grid)

    ρu′ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ρu′)
    ρv′ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, ρv′)

    @inbounds return (ρwᴸ_ccf + ρw̃′[i, j, k] + slope_x * ρu′ᶜᶜᶠ + slope_y * ρv′ᶜᶜᶠ)
end

function assemble_slow_vertical_momentum_tendency!(substepper::AcousticSubstepper,
                                                   model::TerrainCompressibleModel,
                                                   β_stage = nothing)
    grid = model.grid
    arch = architecture(grid)
    g = convert(eltype(grid), model.thermodynamic_constants.gravitational_acceleration)
    Gⁿ = model.timestepper.Gⁿ
    dynamics = model.dynamics
    vertical_pressure_tendency_factor =
        β_stage == 1 ? substepper.final_stage_vertical_pressure_tendency_factor :
        substepper.vertical_pressure_tendency_factor

    pressure_gradients = active_pressure_gradients(substepper)
    if isnothing(pressure_gradients)
        launch!(arch, grid, :xyz, _assemble_terrain_slow_vertical_momentum_tendency!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿ.ρu, Gⁿ.ρv, Gⁿ.ρw,
                dynamics.pressure,
                dynamics.total_density,
                reference_pressure_field(dynamics.reference_state),
                reference_density_field(dynamics.reference_state),
                grid, dynamics, g, vertical_pressure_tendency_factor)
    else
        # The frozen ∂x pᴸ, ∂y pᴸ of this stage, read below and by every explicit step of the stage.
        store_linearization_pressure_gradients!(pressure_gradients, model)
        launch!(arch, grid, :xyz, _assemble_terrain_slow_vertical_momentum_tendency_stored!,
                substepper.slow_vertical_momentum_tendency,
                Gⁿ.ρu, Gⁿ.ρv, Gⁿ.ρw,
                dynamics.pressure,
                dynamics.total_density,
                reference_pressure_field(dynamics.reference_state),
                reference_density_field(dynamics.reference_state),
                pressure_gradients.linearization_x, pressure_gradients.linearization_y,
                grid, g, vertical_pressure_tendency_factor)
    end

    return nothing
end

@kernel function _assemble_terrain_slow_vertical_momentum_tendency_stored!(Gˢρw̃,
                                                                           Gⁿρu, Gⁿρv, Gⁿρw,
                                                                           pᴸ, ρᴸ, pᵣ, ρᵣ, ∂x_pᴸ, ∂y_pᴸ,
                                                                           grid::TerrainFollowingGrid, g,
                                                                           vertical_pressure_tendency_factor)
    i, j, k = @index(Global, NTuple)

    slope_x = terrain_slope_x_ccf(i, j, k, grid)
    slope_y = terrain_slope_y_ccf(i, j, k, grid)
    Gⁿρu_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, Gⁿρu)
    Gⁿρv_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, Gⁿρv)

    ∂z_p′ = terrain_vertical_pressure_gradient(i, j, k, grid, pᴸ, pᵣ)
    ρ′ᶜᶜᶠ = terrain_vertical_buoyancy_density(i, j, k, grid, ρᴸ, ρᵣ)
    horizontal_slow_tendency = slope_x * Gⁿρu_ccf + slope_y * Gⁿρv_ccf
    horizontal_pressure_gradient = stored_horizontal_gradient_correction(i, j, k, grid, ∂x_pᴸ, ∂y_pᴸ)

    @inbounds Gˢρw̃[i, j, k] = (Gⁿρw[i, j, k] -
                                horizontal_slow_tendency -
                                vertical_pressure_tendency_factor * ∂z_p′ +
                                horizontal_pressure_gradient -
                                g * ρ′ᶜᶜᶠ) * (k > 1)
end

@kernel function _assemble_terrain_slow_vertical_momentum_tendency!(Gˢρw̃,
                                                                    Gⁿρu, Gⁿρv, Gⁿρw,
                                                                    pᴸ, ρᴸ, pᵣ, ρᵣ,
                                                                    grid::TerrainFollowingGrid, dynamics, g,
                                                                    vertical_pressure_tendency_factor)
    i, j, k = @index(Global, NTuple)

    slope_x = terrain_slope_x_ccf(i, j, k, grid)
    slope_y = terrain_slope_y_ccf(i, j, k, grid)
    Gⁿρu_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, Gⁿρu)
    Gⁿρv_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, Gⁿρv)

    ∂z_p′ = terrain_vertical_pressure_gradient(i, j, k, grid, pᴸ, pᵣ)
    ρ′ᶜᶜᶠ = terrain_vertical_buoyancy_density(i, j, k, grid, ρᴸ, ρᵣ)
    horizontal_slow_tendency = slope_x * Gⁿρu_ccf + slope_y * Gⁿρv_ccf
    horizontal_pressure_gradient = terrain_horizontal_pressure_gradient_correction(i, j, k, grid, dynamics)

    @inbounds Gˢρw̃[i, j, k] = (Gⁿρw[i, j, k] -
                                horizontal_slow_tendency -
                                vertical_pressure_tendency_factor * ∂z_p′ +
                                horizontal_pressure_gradient -
                                g * ρ′ᶜᶜᶠ) * (k > 1)
end

@inline terrain_vertical_pressure_gradient(i, j, k, grid, p, ::Nothing) =
    ∂zᶜᶜᶠ(i, j, k, grid, p)

@inline terrain_vertical_pressure_gradient(i, j, k, grid, p, pᵣ) =
    ∂zᶜᶜᶠ(i, j, k, grid, δϕ, p, pᵣ)

@inline terrain_vertical_buoyancy_density(i, j, k, grid, ρ, ::Nothing) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ)

@inline terrain_vertical_buoyancy_density(i, j, k, grid, ρ, ρᵣ) =
    ℑzᵃᵃᶠ(i, j, k, grid, δϕ, ρ, ρᵣ)

#####
##### Terrain-corrected pressure gradient
#####
##### The true horizontal pressure gradient at constant z is:
#####   (∂p/∂x)_z = (∂p/∂x)_r - (∂z/∂x)_r · (∂p/∂z)
#####
##### For SlopeOutsideInterpolation (default), Oceananigans' generalized ∂xᶠᶜᶜ
##### on MutableVerticalDiscretization grids computes this chain-rule correction
##### automatically. For SlopeInsideInterpolation, we use basic δx/Δx operators
##### to compute (∂p/∂x)_r, then multiply the slope inside the interpolation.
#####
##### When a terrain reference pressure pᵣ(z) is available, the PG is
##### computed using perturbation pressure p' = p - pᵣ. Since pᵣ depends only
##### on physical height z, its true horizontal gradient (∂pᵣ/∂x)_z = 0 exactly.
##### The perturbation terms are much smaller than the full pressure terms, which
##### reduces the truncation error from the near-cancellation of the two large terms.
##### This is the standard approach for reducing PGF errors in terrain-following
##### (sigma) coordinate models (Klemp, 2011).
#####

@inline perturbation_pressure(i, j, k, grid, p, pᵣ) = @inbounds p[i, j, k] - pᵣ[i, j, k]

# Extract the reference pressure/density fields (or `nothing` when the reference is disabled) from
# the single `reference_state` slot, so the terrain PGF/buoyancy kernels keep dispatching on
# `::Nothing` (full pressure) vs a field (perturbation form).
@inline reference_pressure_field(::Nothing) = nothing
@inline reference_pressure_field(ref::ExnerReferenceState) = ref.pressure
@inline reference_density_field(::Nothing) = nothing
@inline reference_density_field(ref::ExnerReferenceState) = ref.density

@inline function AtmosphereModels.x_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    stencil = d.terrain_metrics.pressure_gradient_stencil
    return terrain_x_pressure_gradient(i, j, k, grid, d, stencil, reference_pressure_field(d.reference_state))
end

##### Slope-outside-interpolation (default): use Oceananigans' generalized ∂xᶠᶜᶜ
##### which applies the chain-rule correction (∂p/∂x)_z = (∂p/∂x)_r - (∂z/∂x)_r · (∂p/∂z)

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, ::Nothing)
    return ∂xᶠᶜᶜ(i, j, k, grid, d.pressure)
end

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, pᵣ)
    return ∂xᶠᶜᶜ(i, j, k, grid, perturbation_pressure, d.pressure, pᵣ)
end

##### Slope-inside-interpolation: ℑz(ℑx(slope * ∂z(p')))
#####
##### The slope is evaluated at each (Center, Center, Face) stencil point
##### and multiplied by ∂z(p') before the 4-point average to (Face, Center, Center).
#####
##### Note: both SlopeOutsideInterpolation and SlopeInsideInterpolation derive
##### the slope from the grid `∂z∂x`/`∂z∂y` operator (the formulation decay) via
##### `terrain_slope_{x,y}_ccf`; they differ only in whether the slope multiplies
##### inside or outside the interpolation stencil.

@inline function slope_x_times_∂z(i, j, k, grid::TerrainFollowingGrid, p)
    slope = terrain_slope_x_ccf(i, j, k, grid)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, p)
end

@inline function slope_x_times_∂z_p′(i, j, k, grid::TerrainFollowingGrid, p, pᵣ)
    slope = terrain_slope_x_ccf(i, j, k, grid)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, perturbation_pressure, p, pᵣ)
end

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, ::Nothing)
    ∂x_p = δxᶠᶜᶜ(i, j, k, grid, d.pressure) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, slope_x_times_∂z, d.pressure)
    return ∂x_p - correction
end

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, pᵣ)
    ∂x_p′ = δxᶠᶜᶜ(i, j, k, grid, perturbation_pressure, d.pressure, pᵣ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, slope_x_times_∂z_p′, d.pressure, pᵣ)
    return ∂x_p′ - correction
end

##### Y-direction pressure gradient

@inline function AtmosphereModels.y_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    stencil = d.terrain_metrics.pressure_gradient_stencil
    return terrain_y_pressure_gradient(i, j, k, grid, d, stencil, reference_pressure_field(d.reference_state))
end

##### Slope-outside-interpolation (default): use Oceananigans' generalized ∂yᶜᶠᶜ

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, ::Nothing)
    return ∂yᶜᶠᶜ(i, j, k, grid, d.pressure)
end

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, pᵣ)
    return ∂yᶜᶠᶜ(i, j, k, grid, perturbation_pressure, d.pressure, pᵣ)
end

##### Slope-inside-interpolation: ℑz(ℑy(slope * ∂z(p')))

@inline function slope_y_times_∂z(i, j, k, grid::TerrainFollowingGrid, p)
    slope = terrain_slope_y_ccf(i, j, k, grid)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, p)
end

@inline function slope_y_times_∂z_p′(i, j, k, grid::TerrainFollowingGrid, p, pᵣ)
    slope = terrain_slope_y_ccf(i, j, k, grid)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, perturbation_pressure, p, pᵣ)
end

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, ::Nothing)
    ∂y_p = δyᶜᶠᶜ(i, j, k, grid, d.pressure) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, slope_y_times_∂z, d.pressure)
    return ∂y_p - correction
end

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, pᵣ)
    ∂y_p′ = δyᶜᶠᶜ(i, j, k, grid, perturbation_pressure, d.pressure, pᵣ) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, slope_y_times_∂z_p′, d.pressure, pᵣ)
    return ∂y_p′ - correction
end

#####
##### Terrain-aware density tendency
#####

function AtmosphereModels.compute_dynamics_tendency!(model::TerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    Gρ = model.timestepper.Gⁿ.ρᵈ
    ρw̃ = model.dynamics.contravariant_vertical_momentum

    launch!(arch, grid, :xyz, _compute_terrain_density_tendency!, Gρ, grid, model.momentum, ρw̃)

    return nothing
end

@kernel function _compute_terrain_density_tendency!(Gρ, grid, momentum, ρw̃)
    i, j, k = @index(Global, NTuple)
    # Use ρw̃ (contravariant momentum) for vertical transport instead of ρw
    @inbounds Gρ[i, j, k] = terrain_dry_density_tendency(i, j, k, grid, momentum.ρu, momentum.ρv, ρw̃)
end

@inline terrain_dry_density_tendency(i, j, k, grid, ρu, ρv, ρw̃) = - divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw̃)

#####
##### Dry-density tendency in the predictors (port of handoff 117fc4c7, K2 campaign 3)
#####
##### Gⁿ.ρᵈ = −∇·(ρu, ρv, ρw̃) of the stage-entry momenta; its only reader is the acoustic predictors. On an
##### acoustic stepper, `update_state!` defers it (no TendencyCallsite callback, no flux condition on ρᵈ) and the
##### stage completes it: with one substep the predictors evaluate the expression (no launch, no store/load);
##### with several, the launch runs at the start of the substep loop. Between the two places nothing writes
##### ρu, ρv, ρw̃ (the stage start writes only stepper fields and U⁰, the end-of-step microphysics update
##### re-runs `update_state!`, which defers again), so the bits are the launch's.
#####

"""
Whether the terrain dry-density tendency is deferred to the acoustic stage and, in stages with one substep,
evaluated inside the predictors (`true` by default; `BREEZE_DENSITY_TENDENCY_IN_PREDICTORS=false` at load).
"""
const DENSITY_TENDENCY_IN_PREDICTORS = Ref(true)

# objectid of the `Gⁿ.ρᵈ` whose computation `update_state!` deferred (0: none pending)
const PENDING_DRY_DENSITY_TENDENCY = Ref{UInt}(0)

# Stage completions of a deferred tendency: low 32 bits folds (Nτ = 1), high bits deferred launches (Nτ > 1).
const DRY_DENSITY_TENDENCY_COMPLETIONS = Ref{Int}(0)

struct FoldedDensityTendency{U, V, W}
    ρu :: U
    ρv :: V
    ρw̃ :: W
end

Adapt.adapt_structure(to, d::FoldedDensityTendency) =
    FoldedDensityTendency(adapt(to, d.ρu), adapt(to, d.ρv), adapt(to, d.ρw̃))

@inline slow_density_tendency(i, j, k, grid, d::FoldedDensityTendency) =
    terrain_dry_density_tendency(i, j, k, grid, d.ρu, d.ρv, d.ρw̃)

# Whether `compute_flux_bc_tendencies!` adds anything to the field's tendency (mirrors Oceananigans' `compute_x_bcs!`
# shortcut per direction; the handoff's `adds_flux_boundary_tendencies`). Zero-flux, value, open etc. add nothing.
function adds_flux_boundary_tendencies(field)
    NotFluxBC = Oceananigans.BoundaryConditions.NotFluxBC
    bcs = field.boundary_conditions
    return !(bcs.west   isa NotFluxBC && bcs.east  isa NotFluxBC) |
           !(bcs.south  isa NotFluxBC && bcs.north isa NotFluxBC) |
           !(bcs.bottom isa NotFluxBC && bcs.top   isa NotFluxBC)
end

function defers_dry_density_tendency(model, callbacks)
    DENSITY_TENDENCY_IN_PREDICTORS[] || return false
    hasproperty(model.timestepper, :substepper) || return false
    any(callback -> callback.callsite isa Oceananigans.TendencyCallsite, callbacks) && return false
    adds_flux_boundary_tendencies(model.dynamics.dry_density) && return false
    return true
end

function AtmosphereModels.compute_dynamics_tendency!(model::TerrainCompressibleModel, callbacks)
    Gρ = model.timestepper.Gⁿ.ρᵈ
    if defers_dry_density_tendency(model, callbacks)
        PENDING_DRY_DENSITY_TENDENCY[] = objectid(Gρ)
    else
        PENDING_DRY_DENSITY_TENDENCY[] = 0
        AtmosphereModels.compute_dynamics_tendency!(model)
    end
    return nothing
end

# The tendency the predictors of a stage with `Nτ` substeps read.
complete_dry_density_tendency!(model, Nτ) = model.timestepper.Gⁿ.ρᵈ

function complete_dry_density_tendency!(model::TerrainCompressibleModel, Nτ)
    Gρ = model.timestepper.Gⁿ.ρᵈ
    PENDING_DRY_DENSITY_TENDENCY[] == objectid(Gρ) || return Gρ
    PENDING_DRY_DENSITY_TENDENCY[] = 0
    DRY_DENSITY_TENDENCY_COMPLETIONS[] += ifelse(Nτ == 1, 1, 1 << 32)  # (folds, deferred launches), for tests
    Nτ == 1 && return FoldedDensityTendency(model.momentum.ρu, model.momentum.ρv,
                                             model.dynamics.contravariant_vertical_momentum)
    AtmosphereModels.compute_dynamics_tendency!(model)
    return Gρ
end

#####
##### Hook into auxiliary variable computation to compute w̃ and ρw̃
#####

function compute_terrain_temperature_and_pressure!(model::TerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics

    # Ensure halos are filled
    AtmosphereModels.fill_halo_regions_together!((dynamics.dry_density, values(prognostic_fields(model.formulation))...))

    # Compute temperature and pressure (same as non-terrain CompressibleModel)
    launch!(arch, grid, :xyz,
            _compute_temperature_and_pressure!,
            model.temperature,
            dynamics.pressure,
            dynamics.dry_density,
            dynamics.total_density,
            model.formulation,
            dynamics,
            specific_prognostic_moisture(model),
            grid,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants,
            stored_energy_denominator(model))

    AtmosphereModels.fill_halo_regions_together!((model.temperature, dynamics.pressure))

    return nothing
end

function AtmosphereModels.compute_auxiliary_dynamics_variables!(model::TerrainCompressibleModel)
    compute_terrain_temperature_and_pressure!(model)

    # Compute contravariant velocity and momentum for terrain transport
    compute_contravariant_velocity!(model)

    return nothing
end

#####
##### Terrain-corrected vertical pressure gradient and buoyancy
#####
##### Without a reference state, the vertical momentum equation computes
##### -∂p/∂z - gρ, where both terms are O(ρg) ≈ 12 Pa/m and nearly cancel.
##### The O(Δz²) truncation error from this cancellation can dominate the
##### physical mountain wave signal. The terrain reference state provides
##### pᵣ and ρᵣ in approximate discrete hydrostatic balance, allowing
##### the vertical PG and buoyancy to be computed in perturbation form:
#####   -(∂p'/∂z) - g ρ'
##### where p' = p - pᵣ and ρ' = ρ - ρᵣ are small perturbations.
#####

# The vertical pressure gradient `-∂(p - pᵣ)/∂z` and buoyancy `-g(ρ - ρᵣ)` are identical on
# terrain-following and height-coordinate grids once both read the single grid-polymorphic
# `reference_state` — the 3D reference field simply broadcasts nothing and indexes `[i, j, k]`.
# The flat `z_pressure_gradient(::CompressibleDynamics)` and `buoyancy_forceᶜᶜᶜ(::CompressibleDynamics)`
# methods therefore serve terrain models too (via `∂z_reference_pressureᶜᶜᶠ`/`reference_densityᶜᶜᶜ`);
# no terrain-specific override is needed.

#####
##### 3D terrain reference state via per-column discrete Exner integration
#####

terrain_reference_profiles(ref_spec) = (ref_spec, nothing)

function terrain_reference_profiles(ref_spec::NamedTuple)
    haskey(ref_spec, :reference_temperature) &&
        throw(ArgumentError("Terrain-following compressible reference states do not support `reference_temperature` with `terrain_metrics`."))

    θᵣ = ref_spec.reference_potential_temperature === nothing ? 288 : ref_spec.reference_potential_temperature
    qᵛᵣ = ref_spec.reference_vapor_mass_fraction

    return θᵣ, qᵛᵣ
end

"""
$(TYPEDSIGNATURES)

Fill the 3D fields `pᵣ`, `ρᵣ` and `πᵣ` with the hydrostatic reference pressure, density and Exner
function, solving the discrete hydrostatic balance per column. On a terrain-following grid,
different columns have different physical heights at the same computational index `k`, so the
reference state varies horizontally even though the reference atmosphere is horizontally uniform.

Each column is anchored at its own terrain surface with the *continuous* hydrostatic state at that
physical height, and every face above it — starting with the surface-to-first-center half cell —
is closed by the Newton solve of the discrete balance
```math
\\frac{p_{ref}[k] - p_{ref}[k-1]}{Δz} + g \\frac{ρ_{ref}[k] + ρ_{ref}[k-1]}{2} = 0
```
to near machine precision (the Exner integration provides only the Newton initial guess).
The reference atmosphere uses level-local moist constants
``Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ``, ``cᵖᵐ = qᵈ cᵖᵈ + qᵛ cᵖᵛ``, ``κᵐ = Rᵐ/cᵖᵐ``, with the dry case
recovered exactly when ``qᵛ ≡ 0``. Enforcing the discrete balance is essential for
reducing the truncation error in the vertical momentum equation (``-∂p/∂z - gρ``), which
would otherwise be dominated by the near-cancellation of two large terms.

The reference pressure is also used for the perturbation horizontal pressure gradient,
reducing the terrain-following PGF error.
"""
function compute_terrain_reference_state!(pᵣ, ρᵣ, πᵣ, pˢ, grid, p₀, ref_spec, pˢᵗ, constants)
    arch = architecture(grid)
    Nz = size(grid, 3)

    θᵣ, qᵛᵣ = terrain_reference_profiles(ref_spec)

    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    g = constants.gravitational_acceleration

    # The reference profiles are host callables (a number, a user `θ(z)`, or a
    # `HorizontalMeanProfile`), so they are evaluated by `set!` — which broadcasts over a CPU mirror
    # of the grid and transfers once — rather than inside a kernel, which would impose GPU
    # type-stability and allocation requirements on user-supplied functions. `HeightProfile` feeds
    # each field's own physical `znode`s to the profile, so on a terrain-following grid every column
    # is sampled at its own heights.
    θ = CenterField(grid)
    set!(θ, HeightProfile(θᵣ))
    fill_halo_regions!(θ)

    qᵛ = terrain_reference_vapor_field(qᵛᵣ, grid)
    θˢ, qᵛˢ = terrain_surface_reference_fields!(pˢ, grid, p₀, θᵣ, qᵛᵣ, pˢᵗ, constants)

    launch!(arch, grid, :xy, _compute_terrain_reference_state!,
            πᵣ, pᵣ, ρᵣ, θ, qᵛ, θˢ, qᵛˢ, pˢ, grid, Nz, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)

    fill_halo_regions!(pᵣ)
    fill_halo_regions!(ρᵣ)
    fill_halo_regions!(πᵣ)

    return nothing
end

# Vapor mass fraction of the reference atmosphere at cell centers. A dry reference (`nothing`)
# becomes a `ZeroField`, so `moist_reference_constants(0, …)` recovers the dry constants exactly and
# the column kernel needs no dry/moist branch.
terrain_reference_vapor_field(::Nothing, grid) = ZeroField(eltype(grid))

function terrain_reference_vapor_field(qᵛᵣ, grid)
    qᵛ = CenterField(grid)
    set!(qᵛ, HeightProfile(qᵛᵣ))
    fill_halo_regions!(qᵛ)
    return qᵛ
end

@kernel function _compute_terrain_surface_heights!(zˢ, grid)
    i, j = @index(Global, NTuple)
    @inbounds zˢ[i, j, 1] = znode(i, j, 1, grid, Center(), Center(), Face())
end

"""
$(TYPEDSIGNATURES)

Fill `pˢ` with the reference pressure at the terrain surface — the bottom face of each column —
and return the matching `(θˢ, qᵛˢ)` there as 2D fields. `pˢ` is the *continuous* hydrostatic
pressure at the local terrain height, reduced from the ``z = 0`` datum `p₀` by
[`moist_hydrostatic_pressure`](@ref); it anchors the column integration in
[`compute_terrain_reference_state!`](@ref) and is retained on the reference state, because the
cold start and the diagnostic hydrostatic pressure must anchor at the same per-column pressure
or they disagree with the reference by ``O(ρgh)`` over terrain.

The surface heights come from a kernel; the profiles and the hydrostatic integration are host
callables (an arbitrary user `θ(z)` and, for moist references, an adaptive vertical integration), so
they are evaluated in a single vectorized pass over those heights — one evaluation per column, not
per cell — and transferred back. A dry reference returns `qᵛˢ::ZeroField`.
"""
function terrain_surface_reference_fields!(pˢ, grid, p₀, θᵣ, qᵛᵣ, pˢᵗ, constants)
    arch = architecture(grid)

    zˢ = Field{Center, Center, Nothing}(grid)
    launch!(arch, grid, :xy, _compute_terrain_surface_heights!, zˢ, grid)

    heights = Array(interior(zˢ))
    θˢ_host = map(z -> evaluate_profile(θᵣ, z), heights)
    pˢ_host = map(z -> moist_hydrostatic_pressure(z, p₀, θᵣ, qᵛᵣ, pˢᵗ, constants), heights)

    copy_to_terrain_surface_field!(pˢ, pˢ_host)
    θˢ = terrain_surface_field(θˢ_host, grid)
    qᵛˢ = qᵛᵣ === nothing ? ZeroField(eltype(grid)) :
          terrain_surface_field(map(z -> evaluate_profile(qᵛᵣ, z), heights), grid)

    return θˢ, qᵛˢ
end

function copy_to_terrain_surface_field!(field, host_values)
    grid = field.grid
    copyto!(interior(field), on_architecture(architecture(grid), convert.(eltype(grid), host_values)))
    return field
end

terrain_surface_field(host_values, grid) =
    copy_to_terrain_surface_field!(Field{Center, Center, Nothing}(grid), host_values)

# Per-column reference integration: anchor at the terrain surface, then integrate upward in discrete
# hydrostatic balance. The surface sits at the bottom face and the first cell center half a cell
# above it, so the anchoring balance spans Δzᶜᶜᶜ/2; the interior faces are the same recurrence the
# height-coordinate reference uses (`integrate_exner_column!`).
@kernel function _compute_terrain_reference_state!(πᵣ, pᵣ, ρᵣ, θ, qᵛ, θˢ, qᵛˢ, pˢ,
                                                   grid, Nz, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
    i, j = @index(Global, NTuple)

    @inbounds begin
        p_surface = pˢ[i, j, 1]
        θ_surface = θˢ[i, j, 1]
        qᵛ_surface = qᵛˢ[i, j, 1]
        θ¹ = θ[i, j, 1]
        qᵛ¹ = qᵛ[i, j, 1]
    end

    Rᵐ_surface, _, κ_surface = moist_reference_constants(qᵛ_surface, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    Π_surface = (p_surface / pˢᵗ)^κ_surface
    ρ_surface = p_surface / (Rᵐ_surface * θ_surface * Π_surface)

    Rᵐ¹, cᵖᵐ¹, κ¹ = moist_reference_constants(qᵛ¹, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    Δz = Δzᶜᶜᶜ(i, j, 1, grid) / 2
    θ_face = (θ¹ + θ_surface) / 2
    Π¹_init = Π_surface - g * Δz / (cᵖᵐ¹ * θ_face)
    p¹_init = pˢᵗ * Π¹_init^(1 / κ¹)
    p¹ = newton_hydrostatic_pressure(p_surface, ρ_surface, θ¹, Rᵐ¹, κ¹, Δz, pˢᵗ, g, p¹_init,
                                     FixedIterations(5))
    Π¹ = (p¹ / pˢᵗ)^κ¹
    ρ¹ = p¹ / (Rᵐ¹ * θ¹ * Π¹)

    @inbounds begin
        πᵣ[i, j, 1] = Π¹
        pᵣ[i, j, 1] = p¹
        ρᵣ[i, j, 1] = ρ¹
    end

    integrate_exner_column!(πᵣ, pᵣ, ρᵣ, θ, qᵛ, i, j, grid, 2, Nz, pˢᵗ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ, g)
end

"""
$(TYPEDSIGNATURES)

Build the single 3D `ExnerReferenceState` for a terrain-following compressible model from an
explicit reference profile `ref_spec` (a `reference_potential_temperature` — constant or `θ(z)` —
optionally with `reference_vapor_mass_fraction`). Its `pressure`/`density`/`exner_function` are
horizontally-varying `CenterField`s in per-column discrete hydrostatic balance
(`compute_terrain_reference_state!`); only `pressure`/`density` are read by the terrain kernels,
but `exner_function` is filled for consistency with the 1D-column form.
"""
function terrain_exner_reference_state(grid, base_pressure, ref_spec, standard_pressure, constants)
    FT = eltype(grid)
    pᵣ = CenterField(grid)
    ρᵣ = CenterField(grid)
    πᵣ = CenterField(grid)
    # The per-column terrain-surface pressure is kept on the reference state: it is the anchor
    # every other "pressure at the surface" consumer needs, and recomputing it elsewhere would
    # require the reference profile, which the reference state does not carry.
    pˢ = Field{Center, Center, Nothing}(grid)
    compute_terrain_reference_state!(pᵣ, ρᵣ, πᵣ, pˢ, grid, base_pressure, ref_spec,
                                     standard_pressure, constants)
    θᵣ, _ = terrain_reference_profiles(ref_spec)
    θ₀ = convert(FT, evaluate_profile(θᵣ, 0))
    # `surface_density = nothing`: the terrain-following pressure/density fields carry no bottom
    # boundary value, so there is none to keep in sync.
    return ExnerReferenceState(convert(FT, base_pressure), pˢ, nothing, θ₀,
                               convert(FT, standard_pressure), pᵣ, ρᵣ, πᵣ)
end

# Terrain-following method of the reference-state builder (the height-coordinate method is in
# `compressible_dynamics.jl`): an explicit profile builds the 3D `ExnerReferenceState`.
build_reference_state(grid, ::TerrainMetrics, ref_spec, base_pressure, standard_pressure, constants) =
    terrain_exner_reference_state(grid, base_pressure, ref_spec, standard_pressure, constants)

"""
$(TYPEDSIGNATURES)

Build the reference specification (`reference_potential_temperature`,
`reference_vapor_mass_fraction`) for a terrain-following compressible model from the
horizontal means of its current `θˡⁱ` and `qᵛ`. The vapor profile is dropped (set to
`nothing`, selecting the dry reference path) when the mean moisture is identically zero.
"""
function terrain_reference_mean_profiles(model)
    θ̄ = horizontal_mean_profile(AtmosphereModels.liquid_ice_potential_temperature(model))
    qᵛ_profile = horizontal_mean_profile(AtmosphereModels.specific_humidity(model))
    q̄ᵛ = all(iszero, qᵛ_profile.values) ? nothing : qᵛ_profile
    return (; reference_potential_temperature=θ̄,
              reference_vapor_mass_fraction=q̄ᵛ)
end

"""
$(TYPEDSIGNATURES)

Recompute the terrain-following model's 3D `ExnerReferenceState` in place from the height-resolved
horizontal-mean state, via `compute_terrain_reference_state!`. Unlike the flat Exner / anelastic
`set_to_mean!` reset, this specializes on the *model* (rather than the reference type) because the
terrain mean must be taken at constant physical height (`horizontal_mean_profile`), not per
computational level. No `update_state!` follows: the terrain reference feeds only the buoyancy and
pressure-gradient tendencies, not any diagnostic field. A no-op if the dynamics carries no reference.
"""
function AtmosphereModels.reset_reference_state!(model::TerrainCompressibleModel)
    dynamics = model.dynamics
    ref = dynamics.reference_state
    ref === nothing && return nothing

    ref_spec = terrain_reference_mean_profiles(model)
    compute_terrain_reference_state!(ref.pressure, ref.density, ref.exner_function,
                                     ref.surface_pressure, model.grid,
                                     base_pressure(dynamics),
                                     ref_spec,
                                     standard_pressure(dynamics),
                                     model.thermodynamic_constants)

    return nothing
end

# Terrain models keep the velocity pass separate (their auxiliary dynamics also builds w̃ and ρw̃).
# C7 (handoff 3af9cbdb, T/p/auxiliary part): the auxiliary thermodynamics and the temperature-and-pressure
# inversion in one launch (`TERRAIN_FUSED_AUXILIARY`, `BREEZE_TERRAIN_FUSED_AUXILIARY=false` at load: two launches).
const TERRAIN_FUSED_AUXILIARY = Ref(true)

function AtmosphereModels.compute_auxiliary_state_variables!(model::TerrainCompressibleModel)
    fold = TERRAIN_FUSED_AUXILIARY[] & VELOCITY_PASS_CONTRAVARIANT[]
    dynamics = model.dynamics
    auxiliaries = fold ? ContravariantVerticalTransport(dynamics.contravariant_vertical_velocity,
                                                        dynamics.contravariant_vertical_momentum) : nothing
    AtmosphereModels.compute_velocities!(model, auxiliaries)
    if TERRAIN_FUSED_AUXILIARY[]
        compute_terrain_fused_thermodynamics!(model)
        compute_contravariant_velocity!(model; interior = !fold)
    else
        AtmosphereModels.compute_auxiliary_thermodynamic_variables!(model)
        AtmosphereModels.compute_auxiliary_dynamics_variables!(model)
    end
    return nothing
end

# The T/p inversion reads only cell (i, j, k), whose auxiliaries the same thread has just written, so every
# interior value equals the two-launch sequence's; the halo fills of both passes follow, in their order, without
# the fill of the provisional temperature (overwritten by the final fill in the two-launch sequence too).
function compute_terrain_fused_thermodynamics!(model::TerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics
    lin = FOLDED_LINEARIZATION[] ? folded_linearization_target(model) : nothing

    fill_halo_regions!(dynamics.dry_density)
    fill_halo_regions!(prognostic_fields(model.formulation))

    kernel! = fused_thermodynamics_kernel(model)
    forwarded = kernel! === _compute_forwarded_terrain_fused_thermodynamics!
    extra = forwarded ? (fused_pass_adjustment(model.microphysics), fused_pass_density_source(model)) : ()

    launch!(arch, grid, :xyz,
            kernel!,
            model.temperature,
            dynamics.pressure,
            dynamics.dry_density,
            dynamics.total_density,
            specific_prognostic_moisture(model),
            model.formulation,
            dynamics,
            grid,
            model.thermodynamic_constants,
            model.microphysics,
            model.microphysical_fields,
            model.moisture_density,
            stored_energy_denominator(model),
            lin, extra...)

    complete_fused_pass_total_density!(model)

    fill_halo_regions!(model.microphysical_fields, AtmosphereModels.kernel_boundary_condition_args(model)...)
    fill_halo_regions!(model.formulation)
    fill_halo_regions!(model.temperature)
    fill_halo_regions!(dynamics.pressure)
    fill_folded_linearization_halos!(lin)

    return nothing
end

# The fused pass stores Πᴸ, θᴸ, γᵐRᵐᴸ of the acoustic substepper (see `FOLDED_LINEARIZATION`).
linearization_in_auxiliary_pass(model::TerrainCompressibleModel) =
    FOLDED_LINEARIZATION[] & TERRAIN_FUSED_AUXILIARY[] && !isnothing(folded_linearization_target(model))

@kernel function _compute_terrain_fused_thermodynamics!(temperature_field, pressure_field,
                                                        dry_density, total_density, specific_prognostic_moisture,
                                                        formulation, dynamics, grid, constants,
                                                        microphysics, microphysical_fields, moisture_density,
                                                        denominator, lin)
    i, j, k = @index(Global, NTuple)
    AtmosphereModels.auxiliary_thermodynamics_at!(i, j, k, grid, specific_prognostic_moisture,
                                                  formulation, dynamics, constants, microphysics,
                                                  microphysical_fields, moisture_density)
    compute_temperature_and_pressure_at!(i, j, k, temperature_field, pressure_field,
                                         dry_density, total_density, formulation, dynamics,
                                         specific_prognostic_moisture, grid, microphysics,
                                         microphysical_fields, constants, denominator)
    store_folded_linearization_at!(i, j, k, lin, pressure_field, dry_density, total_density,
                                   specific_prognostic_moisture, grid, microphysics, microphysical_fields, constants)
end

#####
##### T1 (port of handoff 88c924ae, tier A): the fused pass with its cell's fields read once
#####
##### `_compute_terrain_fused_thermodynamics!` reads ρᵈ, ρθ, ρ, ρqᵉ and the prognostic microphysical densities at its
##### cell several times — after its own stores to θ, qᵉ and the microphysical auxiliaries, which the compiler cannot
##### prove do not alias them — recomputes θ = ρθ/ρᵈ, qʳ, qˢⁿ and the moisture fractions from the reloads, and (in
##### `auxiliary_thermodynamics_at!`) inverts θˡⁱ once more for a temperature the fused kernel discards. The forwarded
##### kernel evaluates the same functions in the same order on register values (`CellValue` in place of a field read at
##### the cell; the stored auxiliaries from the microphysical state they were stored from, `stored_auxiliary_values`),
##### so every output is the same bits. Only for microphysics whose fields are read at (i, j, k) only and whose
##### prognostics are not written by the pass (`forwards_fused_pass`, true for 1M saturation adjustment, set in
##### BreezeCloudMicrophysicsExt). `BREEZE_FORWARDED_FUSED_THERMODYNAMICS=false` at load: the original kernel.
#####

const FORWARDED_FUSED_THERMODYNAMICS = Ref(true)

"""
A register in place of a field that a kernel reads at its own cell only: `c[i, j, k]` is the value it holds.
"""
struct CellValue{T}
    value :: T
end

@inline Base.getindex(c::CellValue, i, j, k) = c.value
@inline Base.eltype(::Type{CellValue{T}}) where T = T

"Whether the fused pass may hold `microphysics`' prognostic fields as cell values (see `CellValue`)."
forwards_fused_pass(microphysics) = false

"""
The auxiliary fields of `microphysics` that `grid_moisture_fractions` reads, as the values `update_microphysical_fields!`
has just stored at the cell from the microphysical state `ℳ` (a NamedTuple of names and values), or `nothing`.
"""
@inline stored_auxiliary_values(microphysics, ℳ, 𝒰) = nothing

@inline cell_prognostics(μ, ::Tuple{}, i, j, k) = μ
@inline function cell_prognostics(μ, names::Tuple, i, j, k)
    name = first(names)
    value = @inbounds getproperty(μ, name)[i, j, k]
    return cell_prognostics(merge(μ, NamedTuple{(name,)}((CellValue(value),))), Base.tail(names), i, j, k)
end

@inline with_stored_auxiliaries(μ, ::Nothing) = μ
@inline with_stored_auxiliaries(μ, ::NamedTuple{()}) = μ
@inline function with_stored_auxiliaries(μ, values::NamedTuple{names}) where names
    name = first(names)
    value = CellValue(convert(eltype(getproperty(μ, name)), first(values)))
    return with_stored_auxiliaries(merge(μ, NamedTuple{(name,)}((value,))), Base.tail(values))
end

@inline forwarded_formulation(f::LiquidIcePotentialTemperatureFormulation, ρθ) =
    LiquidIcePotentialTemperatureFormulation(CellValue(ρθ), f.potential_temperature, f.temperature_solver)

@inline forwarded_linearization(::Nothing, ρθ) = nothing
@inline forwarded_linearization(lin, ρθ) = merge(lin, (; ρθ = CellValue(ρθ)))

# The cell's total density: the stored field (F6 off), or diagnosed and stored here (`DiagnosedTotalDensity`, F6).
@inline forwarded_total_density!(::Nothing, i, j, k, total_density, microphysics, ρᵈ, ρqᵉ, μ) =
    @inbounds total_density[i, j, k]

#####
##### P07 (port of handoff 2541e95b, tier A): the forwarded pass's saturation adjustment returns the state of its last
##### secant residual (`Microphysics.forwarded_adjust_thermodynamic_state`). The microphysics is handed to the
##### adjustment wrapped in `ForwardedPartition` (method in BreezeCloudMicrophysicsExt for 1M saturation adjustment);
##### `BREEZE_FORWARDED_PARTITION=false` at load: the original adjustment.
#####

const FORWARDED_PARTITION = Ref(true)

"""
$(TYPEDEF)

The adjustment argument of the forwarded fused pass: `microphysics`, whose saturation adjustment returns the state of
its last secant residual instead of evaluating it again at the root.
"""
struct ForwardedPartition{M}
    microphysics :: M
end

Adapt.adapt_structure(to, a::ForwardedPartition) = ForwardedPartition(adapt(to, a.microphysics))

"Whether `microphysics` has a forwarded-partition adjustment (`ForwardedPartition`)."
forwards_secant_partition(microphysics) = false

fused_pass_adjustment(microphysics) =
    FORWARDED_PARTITION[] && forwards_secant_partition(microphysics) ? ForwardedPartition(microphysics) : microphysics
#####
##### F6 (port of handoff 95077e10, tier A): the forwarded fused pass diagnoses and stores the total density with
##### `_compute_total_density!`'s expression on its cell values; `update_state!` skips `compute_total_density!` and
##### the tracer conversion, which follow the pass (ρ halo fill, conversion, tracer halo refill).
##### `BREEZE_FUSED_TOTAL_DENSITY=false` at load: the separate launch.
#####

const FUSED_TOTAL_DENSITY = Ref(true)

struct DiagnosedTotalDensity end

@inline function forwarded_total_density!(::DiagnosedTotalDensity, i, j, k, total_density, microphysics, ρᵈ, ρqᵉ, μ)
    ρ = convert(eltype(total_density),
                AtmosphereModels.total_density(i, j, k, CellValue(ρᵈ), microphysics, CellValue(ρqᵉ), μ))
    @inbounds total_density[i, j, k] = ρ
    return ρ
end

AtmosphereModels.diagnoses_total_density_in_auxiliary_pass(model::TerrainCompressibleModel) =
    FUSED_TOTAL_DENSITY[] & TERRAIN_FUSED_AUXILIARY[] &&
    fused_thermodynamics_kernel(model) === _compute_forwarded_terrain_fused_thermodynamics!

fused_pass_density_source(model) =
    AtmosphereModels.diagnoses_total_density_in_auxiliary_pass(model) ? DiagnosedTotalDensity() : nothing

function complete_fused_pass_total_density!(model)
    AtmosphereModels.diagnoses_total_density_in_auxiliary_pass(model) || return nothing
    fill_halo_regions!(model.dynamics.total_density)
    AtmosphereModels.tracer_density_to_specific!(model)
    isempty(model.tracers) || fill_halo_regions!(model.tracers, AtmosphereModels.kernel_boundary_condition_args(model)...)
    return nothing
end

fused_thermodynamics_kernel(model) = _compute_terrain_fused_thermodynamics!

function fused_thermodynamics_kernel(model::AtmosphereModel{<:Any, <:LiquidIcePotentialTemperatureFormulation})
    forwarded = FORWARDED_FUSED_THERMODYNAMICS[] && forwards_fused_pass(model.microphysics)
    return forwarded ? _compute_forwarded_terrain_fused_thermodynamics! : _compute_terrain_fused_thermodynamics!
end

@kernel function _compute_forwarded_terrain_fused_thermodynamics!(temperature_field, pressure_field,
                                                                  dry_density, total_density, specific_prognostic_moisture,
                                                                  formulation, dynamics, grid, constants,
                                                                  microphysics, microphysical_fields, moisture_density,
                                                                  denominator, lin, adjustment, density_source)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵈ  = dry_density[i, j, k]                         # = dynamics_density(dynamics)
        ρθ  = formulation.potential_temperature_density[i, j, k]
        ρqᵉ = moisture_density[i, j, k]
    end
    μ = cell_prognostics(microphysical_fields, AtmosphereModels.prognostic_field_names(microphysics), i, j, k)
    ρ = forwarded_total_density!(density_source, i, j, k, total_density, microphysics, ρᵈ, ρqᵉ, μ)

    # auxiliary_thermodynamics_at!
    θ = ρθ / ρᵈ
    @inbounds formulation.potential_temperature[i, j, k] = θ
    qᵛᵉ = ρqᵉ / ρ
    @inbounds specific_prognostic_moisture[i, j, k] = qᵛᵉ
    q₀ = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, μ)
    𝒰₀ = LiquidIceDensityState(θ, q₀, standard_pressure(dynamics), ρ, formulation.temperature_solver)
    μᵖ = AtmosphereModels.extract_microphysical_prognostics(i, j, k, microphysics, μ)
    𝒰₁ = AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰₀, adjustment, qᵛᵉ, constants, μᵖ, ρ)
    AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, microphysics, ρ, 𝒰₁, constants)

    zero_velocities = (; u = zero(ρ), v = zero(ρ), w = zero(ρ))
    ℳ = AtmosphereModels.grid_microphysical_state(i, j, k, grid, microphysics, μ, ρ, 𝒰₁, zero_velocities)
    μ₁ = with_stored_auxiliaries(μ, stored_auxiliary_values(microphysics, ℳ, 𝒰₁))

    # compute_temperature_and_pressure_at! and store_folded_linearization_at! on the cell values
    f₁ = forwarded_formulation(formulation, ρθ)
    T, p = compute_temperature_and_pressure_at!(i, j, k, temperature_field, pressure_field,
                                                CellValue(ρᵈ), CellValue(ρ), f₁, dynamics,
                                                CellValue(qᵛᵉ), grid, microphysics, μ₁, constants, denominator)
    pˢ = convert(eltype(pressure_field), p)
    store_folded_linearization_at!(i, j, k, forwarded_linearization(lin, ρθ), CellValue(pˢ), CellValue(ρᵈ),
                                   CellValue(ρ), CellValue(qᵛᵉ), grid, microphysics, μ₁, constants)
end
