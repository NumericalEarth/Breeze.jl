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
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, δxᶠᶜᶜ, δyᶜᶠᶜ, Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, ∂zᶜᶜᶠ, Δzᶜᶜᶠ, Δzᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!, NormalFlowBoundaryCondition, FieldBoundaryConditions

using Breeze.TerrainFollowingDiscretization: TerrainMetrics, SlopeOutsideInterpolation,
                                              SlopeInsideInterpolation,
                                              TerrainFollowingGrid, ∂z∂x, ∂z∂y

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
$(TYPEDSIGNATURES)

Reduce a 3D `field` to a `HorizontalMeanProfile` of its horizontal mean at each
vertical level, paired with the horizontal-mean physical height of that level. On a
terrain-following grid the cell-center height `znode(i, j, k, …)` varies with `(i, j)`, so
both the value and the height are averaged over `(i, j)` (on a CPU mirror of the grid) to
give a single representative profile that can be re-evaluated per column.
"""
function horizontal_mean_profile(field)
    grid = field.grid
    Nx, Ny, Nz = size(grid)
    cpu_grid = Oceananigans.Architectures.on_architecture(Oceananigans.CPU(), grid)
    data = Array(Oceananigans.interior(field))
    FT = eltype(grid)
    c = Center()
    normalization = one(FT) / (Nx * Ny)
    heights = zeros(FT, Nz)
    values = zeros(FT, Nz)

    for k in 1:Nz
        height_sum = zero(FT)
        value_sum = zero(FT)

        for j in 1:Ny, i in 1:Nx
            height_sum += znode(i, j, k, cpu_grid, c, c, c)
            value_sum += data[i, j, k]
        end

        heights[k] = height_sum * normalization
        values[k] = value_sum * normalization
    end

    return HorizontalMeanProfile(heights, values)
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
function compute_contravariant_velocity!(model::TerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics
    w̃ = dynamics.contravariant_vertical_velocity
    ρw̃ = dynamics.contravariant_vertical_momentum

    launch!(arch, grid, :xyz,
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

    @inbounds begin
        w̃[i, j, k] = w̃_ijk
        ρw̃[i, j, k] = ρw̃_ijk
    end
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
    launch!(arch, grid, :xyz, _initialize_terrain_vertical_momentum_perturbation!,
            substepper.momentum_perturbation.w,
            Uᴸ_outer.ρu, Uᴸ_outer.ρv, Uᴸ_outer.ρw,
            model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
            grid)
    return nothing
end

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

# `slope_correction` (0 or 1) gates the horizontal slope correction slopeₓ·∂ₓ(Cᴸ(ρθ)′) +
# slopeᵧ·∂ᵧ(Cᴸ(ρθ)′) in lockstep with the MPAS first-small-step gate applied to ρu's
# perturbation PGF in `_explicit_horizontal_step!`. Because ρw̃ = ρw − slopeₓ·ρu − slopeᵧ·ρv,
# the slope-projected correction must be skipped on exactly the substeps where ρu skips
# its perturbation horizontal PGF; otherwise the two are out of phase on substep 1 of a
# multi-substep stage. The vertical ∂z(Cᴸ(ρθ)′) part is always applied — the
# vertical acoustic mode is solved implicitly every substep.
@inline function ∇ᶻp′(i, j, k, grid,
                                                dynamics::TerrainCompressibleDynamics,
                                                ρθ′, Πᴸ, γRᵐᴸ, slope_correction)
    ∂z_p′ = ∂zᶜᶜᶠ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
    correction = terrain_horizontal_linearized_pressure_gradient_correction(i, j, k, grid,
                                                                            dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    return ∂z_p′ - slope_correction * correction
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

@inline function slope_x_times_∂z_linearized_pressure(i, j, k, grid, ρθ′, Πᴸ, γRᵐᴸ)
    slope = terrain_slope_x_ccf(i, j, k, grid)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function slope_y_times_∂z_linearized_pressure(i, j, k, grid, ρθ′, Πᴸ, γRᵐᴸ)
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

@inline function terrain_vertical_transport_momentum(i, j, k, grid,
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

@inline function acoustic_recovered_vertical_momentum(i, j, k, grid,
                                                      dynamics::TerrainCompressibleDynamics,
                                                      ρuᴸ, ρvᴸ, ρwᴸ, ρu′, ρv′, ρw̃′)
    slope_x = terrain_slope_x_ccf(i, j, k, grid)
    slope_y = terrain_slope_y_ccf(i, j, k, grid)

    ρuᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, total_momentum, ρuᴸ, ρu′)
    ρvᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, total_momentum, ρvᴸ, ρv′)
    ρw̃_stage = transport_ρw(i, j, k, grid, dynamics,
                                                           ρuᴸ, ρvᴸ, ρwᴸ)
    @inbounds ρw̃ᵐ⁺ = ρw̃_stage + ρw̃′[i, j, k]

    return ρw̃ᵐ⁺ + slope_x * ρuᶜᶜᶠ + slope_y * ρvᶜᶜᶠ
end

@inline total_momentum(i, j, k, grid, mᴸ, m′) = @inbounds mᴸ[i, j, k] + m′[i, j, k]

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

    launch!(arch, grid, :xyz, _assemble_terrain_slow_vertical_momentum_tendency!,
            substepper.slow_vertical_momentum_tendency,
            Gⁿ.ρu, Gⁿ.ρv, Gⁿ.ρw,
            dynamics.pressure,
            dynamics.total_density,
            dynamics.terrain_reference_pressure,
            dynamics.terrain_reference_density,
            grid, dynamics, g, vertical_pressure_tendency_factor)

    return nothing
end

@kernel function _assemble_terrain_slow_vertical_momentum_tendency!(Gˢρw̃,
                                                                    Gⁿρu, Gⁿρv, Gⁿρw,
                                                                    pᴸ, ρᴸ, pᵣ, ρᵣ,
                                                                    grid, dynamics, g,
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

@inline function AtmosphereModels.x_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    stencil = d.terrain_metrics.pressure_gradient_stencil
    return terrain_x_pressure_gradient(i, j, k, grid, d, stencil, d.terrain_reference_pressure)
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

@inline function slope_x_times_∂z(i, j, k, grid, p)
    slope = terrain_slope_x_ccf(i, j, k, grid)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, p)
end

@inline function slope_x_times_∂z_p′(i, j, k, grid, p, pᵣ)
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
    return terrain_y_pressure_gradient(i, j, k, grid, d, stencil, d.terrain_reference_pressure)
end

##### Slope-outside-interpolation (default): use Oceananigans' generalized ∂yᶜᶠᶜ

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, ::Nothing)
    return ∂yᶜᶠᶜ(i, j, k, grid, d.pressure)
end

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, pᵣ)
    return ∂yᶜᶠᶜ(i, j, k, grid, perturbation_pressure, d.pressure, pᵣ)
end

##### Slope-inside-interpolation: ℑz(ℑy(slope * ∂z(p')))

@inline function slope_y_times_∂z(i, j, k, grid, p)
    slope = terrain_slope_y_ccf(i, j, k, grid)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, p)
end

@inline function slope_y_times_∂z_p′(i, j, k, grid, p, pᵣ)
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
    @inbounds Gρ[i, j, k] = - divᶜᶜᶜ(i, j, k, grid, momentum.ρu, momentum.ρv, ρw̃)
end

#####
##### Hook into auxiliary variable computation to compute w̃ and ρw̃
#####

function compute_terrain_temperature_and_pressure!(model::TerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics

    # Ensure halos are filled
    fill_halo_regions!(dynamics.dry_density)
    fill_halo_regions!(prognostic_fields(model.formulation))

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
            model.thermodynamic_constants)

    fill_halo_regions!(model.temperature)
    fill_halo_regions!(dynamics.pressure)

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

@inline function AtmosphereModels.z_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    ∂z_p = ∂zᶜᶜᶠ(i, j, k, grid, d.pressure)
    ∂z_pᵣ = terrain_∂z_reference_pressure(i, j, k, grid, d.terrain_reference_pressure)
    return ∂z_p - ∂z_pᵣ
end

@inline terrain_∂z_reference_pressure(i, j, k, grid, ::Nothing) = zero(grid)
@inline terrain_∂z_reference_pressure(i, j, k, grid, pᵣ) = ∂zᶜᶜᶠ(i, j, k, grid, pᵣ)

@inline function AtmosphereModels.buoyancy_forceᶜᶜᶜ(i, j, k, grid,
                                                    dynamics::TerrainCompressibleDynamics,
                                                    temperature,
                                                    specific_moisture,
                                                    microphysics,
                                                    microphysical_fields,
                                                    constants)
    ρ_field = dynamics.total_density  # total air density: gravity acts on total mass
    @inbounds ρ = ρ_field[i, j, k]
    g = constants.gravitational_acceleration
    ρᵣ = terrain_reference_density(i, j, k, dynamics.terrain_reference_density)
    return -g * (ρ - ρᵣ)
end

@inline terrain_reference_density(i, j, k, ::Nothing) = false
@inline terrain_reference_density(i, j, k, ρᵣ) = @inbounds ρᵣ[i, j, k]

#####
##### 3D terrain reference state via per-column discrete Exner integration
#####

using Breeze.Thermodynamics: evaluate_profile, hydrostatic_pressure,
                             newton_hydrostatic_pressure, moist_reference_constants
using Breeze.Solvers: FixedIterations

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

Integrate the hydrostatic equation ``∂p/∂z = \\mathrm{dpdz}(z, p)`` from the surface to
height ``z``, repeatedly doubling the number of steps until the pressure at ``z`` changes
by less than the relative `tolerance` between successive refinements. `dpdz(z, p)` returns
the local pressure gradient ``-g ρ`` given height and pressure.
"""
function converged_hydrostatic_pressure(z, p₀, dpdz;
                                        tolerance = sqrt(eps(float(typeof(p₀)))),
                                        initial_steps = 16,
                                        max_steps = 1 << 16)
    z == 0 && return p₀

    integrate(nsteps) = begin
        dz = z / nsteps
        half_dz = dz / 2
        p = p₀
        for i in 1:nsteps
            zₗ = (i - 1) * dz
            k₁ = dpdz(zₗ, p)
            k₂ = dpdz(zₗ + half_dz, p + k₁ * half_dz)
            p += k₂ * dz
        end
        return p
    end

    nsteps = initial_steps
    p_coarse = integrate(nsteps)
    while nsteps < max_steps
        nsteps *= 2
        p_fine = integrate(nsteps)
        abs(p_fine - p_coarse) ≤ tolerance * abs(p_fine) && return p_fine
        p_coarse = p_fine
    end

    return p_coarse
end

terrain_hydrostatic_pressure(z, p₀, θᵣ, ::Nothing, pˢᵗ, constants) =
    hydrostatic_pressure(z, p₀, θᵣ, pˢᵗ, constants)

function terrain_hydrostatic_pressure(z, p₀, θᵣ, qᵛᵣ, pˢᵗ, constants)
    # Compute the continuous hydrostatic pressure at physical height `z`. For
    # moist terrain columns this supplies the boundary state at the local terrain
    # surface; the first cell center and the interior levels are then obtained by
    # the same discrete-balance Newton solve.
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    g = constants.gravitational_acceleration

    @inline function dpdz(zⁿ, p)
        θⁿ = evaluate_profile(θᵣ, zⁿ)
        qᵛⁿ = evaluate_profile(qᵛᵣ, zⁿ)
        Rᵐⁿ, cᵖᵐⁿ, κᵐⁿ = moist_reference_constants(qᵛⁿ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
        Tⁿ = θⁿ * (p / pˢᵗ)^κᵐⁿ
        return -g * p / (Rᵐⁿ * Tⁿ)
    end

    return converged_hydrostatic_pressure(z, p₀, dpdz;
                                          tolerance = sqrt(eps(float(typeof(p₀)))))
end

"""
$(TYPEDSIGNATURES)

Fill the 3D fields `pᵣ` and `ρᵣ` with the hydrostatic reference pressure and
density, solving the discrete hydrostatic balance per column. On a terrain-following
grid, different columns have different physical heights at the same computational index
`k`, so the reference state varies horizontally even though the reference atmosphere is
horizontally uniform.

Moist terrain columns first assign the local terrain surface the continuous hydrostatic
state at its physical height, then use the same Newton solve for the surface-to-center
half cell and every interior face to drive the discrete hydrostatic balance. Dry terrain
columns keep the direct continuous anchor at the first cell center and use the Newton
solve only on interior faces.
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
function compute_terrain_reference_state!(pᵣ, ρᵣ, grid, p₀, ref_spec, pˢᵗ, constants)
    # The 3D reference state is filled once, at construction. Each column is an upward,
    # serial-in-`k` discrete-hydrostatic Newton solve that evaluates the (possibly
    # functional) reference θ — and qᵛ for moist columns — at the physical height of
    # every cell. We compute it on the host — into plain arrays, using a CPU mirror of
    # the grid — then bulk-copy the result into the reference Fields. This keeps
    # `reference_potential_temperature`/`reference_vapor_mass_fraction` ordinary host
    # functions (no GPU type-stability/allocation requirement) and costs a single
    # host→device transfer instead of a scalar host↔device op per cell. The CPU mirror
    # preserves the materialised terrain through `on_architecture`, so its `znode`/`Δz`
    # match the device grid exactly.
    Nx, Ny, Nz = size(grid)
    c = Center()

    θᵣ, qᵛᵣ = terrain_reference_profiles(ref_spec)

    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    g = constants.gravitational_acceleration

    cpu_grid = Oceananigans.Architectures.on_architecture(Oceananigans.CPU(), grid)
    FT = eltype(grid)
    p_host = zeros(FT, Nx, Ny, Nz)
    ρ_host = zeros(FT, Nx, Ny, Nz)

    for j in 1:Ny, i in 1:Nx
        p⁻ = zero(FT)
        ρ⁻ = zero(FT)
        Π⁻ = zero(FT)

        for k in 1:Nz
            z_phys = znode(i, j, k, cpu_grid, c, c, c)
            θₖ = evaluate_profile(θᵣ, z_phys)
            qᵛₖ = qᵛᵣ === nothing ? zero(θₖ) : evaluate_profile(qᵛᵣ, z_phys)
            Rᵐₖ, cᵖᵐₖ, κₖ = moist_reference_constants(qᵛₖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)

            if k == 1
                if qᵛᵣ === nothing
                    pₖ = terrain_hydrostatic_pressure(z_phys, p₀, θᵣ, qᵛᵣ, pˢᵗ, constants)
                else
                    z_surface = znode(i, j, 1, cpu_grid, c, c, Face())
                    θ_surface = evaluate_profile(θᵣ, z_surface)
                    qᵛ_surface = evaluate_profile(qᵛᵣ, z_surface)
                    Rᵐ_surface, cᵖᵐ_surface, κ_surface = moist_reference_constants(qᵛ_surface, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)

                    p⁻ = terrain_hydrostatic_pressure(z_surface, p₀, θᵣ, qᵛᵣ, pˢᵗ, constants)
                    Π_surface = (p⁻ / pˢᵗ)^κ_surface
                    ρ⁻ = p⁻ / (Rᵐ_surface * θ_surface * Π_surface)

                    # The surface sits at the bottom face; the first cell center is
                    # half a cell above it, so the surface-to-center balance spans Δzᶜᶜᶜ/2.
                    Δz = Δzᶜᶜᶜ(i, j, 1, cpu_grid) / 2
                    θ_face = (θₖ + θ_surface) / 2
                    Πₖ_init = Π_surface - g * Δz / (cᵖᵐₖ * θ_face)
                    pₖ = pˢᵗ * Πₖ_init^(1 / κₖ)
                    pₖ = newton_hydrostatic_pressure(p⁻, ρ⁻, θₖ, Rᵐₖ, κₖ, Δz, pˢᵗ, g, pₖ, FixedIterations(5))
                end
            else
                z_below = znode(i, j, k - 1, cpu_grid, c, c, c)
                θ_below = evaluate_profile(θᵣ, z_below)
                θ_face = (θₖ + θ_below) / 2
                Δz = Δzᶜᶜᶠ(i, j, k, cpu_grid)
                Πₖ_init = Π⁻ - g * Δz / (cᵖᵐₖ * θ_face)
                pₖ = pˢᵗ * Πₖ_init^(1 / κₖ)
                pₖ = newton_hydrostatic_pressure(p⁻, ρ⁻, θₖ, Rᵐₖ, κₖ, Δz, pˢᵗ, g, pₖ, FixedIterations(5))
            end

            Πₖ = (pₖ / pˢᵗ)^κₖ
            ρₖ = pₖ / (Rᵐₖ * θₖ * Πₖ)
            @inbounds p_host[i, j, k] = pₖ
            @inbounds ρ_host[i, j, k] = ρₖ

            p⁻ = pₖ
            ρ⁻ = ρₖ
            Π⁻ = Πₖ
        end
    end

    arch = architecture(grid)
    copyto!(Oceananigans.interior(pᵣ), Oceananigans.Architectures.on_architecture(arch, p_host))
    copyto!(Oceananigans.interior(ρᵣ), Oceananigans.Architectures.on_architecture(arch, ρ_host))
    fill_halo_regions!(pᵣ)
    fill_halo_regions!(ρᵣ)
    return nothing
end

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

Recompute the 3D terrain reference pressure/density in place from the horizontal-mean state,
via `compute_terrain_reference_state!`. Unlike the Exner/anelastic `set_to_mean!` reset, no
`update_state!` follows: the terrain reference feeds only the buoyancy and pressure-gradient
tendencies, not any diagnostic field. A no-op if the dynamics carries no terrain reference.
"""
function AtmosphereModels.reset_reference_state!(model::TerrainCompressibleModel)
    dynamics = model.dynamics
    pᵣ = dynamics.terrain_reference_pressure
    ρᵣ = dynamics.terrain_reference_density
    (pᵣ === nothing || ρᵣ === nothing) && return nothing

    ref_spec = terrain_reference_mean_profiles(model)
    compute_terrain_reference_state!(pᵣ, ρᵣ, model.grid,
                                     surface_pressure(dynamics),
                                     ref_spec,
                                     standard_pressure(dynamics),
                                     model.thermodynamic_constants)

    return nothing
end
