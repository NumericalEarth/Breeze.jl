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
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, δxᶠᶜᶜ, δyᶜᶠᶜ, Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, ∂zᶜᶜᶠ, Δzᶜᶜᶠ
using Oceananigans.BoundaryConditions: fill_halo_regions!, OpenBoundaryCondition, FieldBoundaryConditions

using Breeze.TerrainFollowingDiscretization: TerrainMetrics, SlopeOutsideInterpolation,
                                              SlopeInsideInterpolation,
                                              TerrainFollowingGrid, ∂z∂x, ∂z∂y

#####
##### Terrain-aware type alias
#####

const TerrainCompressibleDynamics = CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:TerrainMetrics}
const TerrainCompressibleModel = AtmosphereModel{<:TerrainCompressibleDynamics}

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
            grid, model.momentum, dynamics.density,
            dynamics.terrain_metrics)

    # The terrain kinematic BC (w̃ = 0 at the surface) is enforced declaratively:
    # `ρw` carries an Open bottom BC ρw|₁ = slopeₓ·ρu + slopeᵧ·ρv (see
    # `terrain_kinematic_bottom_ρw`), applied by `fill_halo_regions!(model.momentum, …)`
    # before this kernel runs. Because the slope/interpolation here matches that BC,
    # ρw̃|₁ = ρw|₁ − slope·ρu = 0 falls out automatically — no imperative bottom-face
    # zeroing (and no one-shot IC kernel) required.
    fill_halo_regions!(w̃)
    fill_halo_regions!(ρw̃)

    return nothing
end

@kernel function _compute_contravariant_velocity!(w̃, ρw̃, grid, momentum, density, metrics)
    i, j, k = @index(Global, NTuple)

    # Terrain slopes (∂z/∂x, ∂z/∂y)_r at (Center, Center, Face). On a BTF/MVD
    # grid these come from `metrics` with linear decay; on a terrain-following
    # coordinate grid they come from the grid operator (formulation decay).
    slope_x = terrain_slope_x_ccf(i, j, k, grid, metrics)
    slope_y = terrain_slope_y_ccf(i, j, k, grid, metrics)

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
            grid, model.dynamics.terrain_metrics)
    return nothing
end

@kernel function _initialize_terrain_vertical_momentum_perturbation!(ρw̃′,
                                                                     ρu_outer, ρv_outer, ρw_outer,
                                                                     ρu_stage, ρv_stage, ρw_stage,
                                                                     grid, metrics)
    i, j, k = @index(Global, NTuple)
    ρw̃_outer = terrain_vertical_transport_momentum(i, j, k, grid, metrics,
                                                    ρu_outer, ρv_outer, ρw_outer)
    ρw̃_stage = terrain_vertical_transport_momentum(i, j, k, grid, metrics,
                                                    ρu_stage, ρv_stage, ρw_stage)
    @inbounds ρw̃′[i, j, k] = ρw̃_outer - ρw̃_stage
end

@inline function x_linearized_pressure_gradient(i, j, k, grid,
                                                dynamics::TerrainCompressibleDynamics,
                                                ρθ′, Πᴸ, γRᵐᴸ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_x_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                  stencil, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function y_linearized_pressure_gradient(i, j, k, grid,
                                                dynamics::TerrainCompressibleDynamics,
                                                ρθ′, Πᴸ, γRᵐᴸ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_y_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                  stencil, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function terrain_slope_x_ccf(i, j, k, grid, metrics)
    ∂x_h_cc = ℑxᶜᵃᵃ(i, j, 1, grid, metrics.∂x_h)
    r = rnode(k, grid, Face())
    return ∂x_h_cc * (1 - r / metrics.z_top)
end

@inline function terrain_slope_y_ccf(i, j, k, grid, metrics)
    ∂y_h_cc = ℑyᵃᶜᵃ(i, j, 1, grid, metrics.∂y_h)
    r = rnode(k, grid, Face())
    return ∂y_h_cc * (1 - r / metrics.z_top)
end

# On a TerrainFollowingVerticalDiscretization grid the coordinate owns the
# slope: take (∂z/∂x)_r from the grid operator (which carries the formulation's
# decay — linear for LinearDecay, sinh for TwoLevelDecay) and interpolate the x-face
# value to (Center, Center) at the z-face. The `metrics` argument is ignored;
# σ and the slope come from the one coordinate map, so they cannot disagree.
# Use Oceananigans' stagger interpolators (`ℑxᶜᵃᵃ`/`ℑyᵃᶜᵃ`) instead of a
# manual `(idx, idx+1)/2` average: those handle Flat dimensions correctly.
# The naive form reads `∂z∂y(i, j+1, …)` which is out-of-bounds on a Flat-y
# grid (Ny = 1, no y halo) and returns uninitialised memory — which then
# propagates as NaN through `compute_contravariant_velocity!` and the rest
# of the substep. `ℑyᵃᶜᵃ` on a Flat-y grid collapses to a no-op, matching
# the MVD `terrain_slope_y_ccf` path that uses `ℑyᵃᶜᵃ(metrics.∂y_h)`.
@inline terrain_slope_x_ccf(i, j, k, grid::TerrainFollowingGrid, metrics) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂z∂x, Face())

@inline terrain_slope_y_ccf(i, j, k, grid::TerrainFollowingGrid, metrics) =
    ℑyᵃᶜᵃ(i, j, k, grid, ∂z∂y, Face())

#####
##### Bottom kinematic boundary condition for ρw (declarative, dispatch-selected)
#####
##### "No flow through the terrain surface" is, in contravariant form,
#####   ρw̃ = ρw - slopeₓ·ρu - slopeᵧ·ρv = 0   at the bottom face,
##### i.e. the Cartesian vertical momentum follows the terrain:
#####   ρw|₁ = slopeₓ·ρu + slopeᵧ·ρv .
##### We impose this as an `OpenBoundaryCondition` on `ρw` whose value is computed
##### from the live momentum at fill time. `fill_halo_regions!(model.momentum,
##### clock, fields(model))` runs every `update_state!`/substep, so the surface
##### stays kinematically balanced at the IC and throughout the run — replacing the
##### previous one-shot IC kernel and the imperative bottom-face zeroing. The slope
##### and interpolation match `compute_contravariant_velocity!`, so ρw̃|₁ = 0 to
##### machine precision. The perturbation fields keep their impenetrable defaults
##### (the substepper deliberately does not inherit this BC; see the §"acoustic
##### substep" note on issue #716).

@inline function terrain_kinematic_bottom_ρw(i, j, grid, clock, model_fields)
    slope_x = terrain_slope_x_ccf(i, j, 1, grid, nothing)
    slope_y = terrain_slope_y_ccf(i, j, 1, grid, nothing)
    ρu_ccf  = ℑzᵃᵃᶠ(i, j, 1, grid, ℑxᶜᵃᵃ, model_fields.ρu)
    ρv_ccf  = ℑzᵃᵃᶠ(i, j, 1, grid, ℑyᵃᶜᵃ, model_fields.ρv)
    return slope_x * ρu_ccf + slope_y * ρv_ccf
end

# Dispatch the ρw bottom boundary condition on the grid type: terrain-following
# grids get the kinematic Open BC; every other grid keeps the BCs it was given
# (impenetrable bottom by default). The terrain BC sets only the bottom; the top
# (impenetrable lid) and horizontal sides take their usual regularised defaults.
terrain_ρw_boundary_conditions(grid, ρw_bcs) = ρw_bcs
terrain_ρw_boundary_conditions(::TerrainFollowingGrid, ρw_bcs) =
    FieldBoundaryConditions(; west = ρw_bcs.west, east = ρw_bcs.east,
                              south = ρw_bcs.south, north = ρw_bcs.north,
                              bottom = OpenBoundaryCondition(terrain_kinematic_bottom_ρw; discrete_form = true),
                              top = ρw_bcs.top, immersed = ρw_bcs.immersed)

@inline function terrain_horizontal_pressure_gradient_correction(i, j, k, grid, dynamics)
    metrics = dynamics.terrain_metrics
    slope_x = terrain_slope_x_ccf(i, j, k, grid, metrics)
    slope_y = terrain_slope_y_ccf(i, j, k, grid, metrics)
    ∂x_p_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, AtmosphereModels.x_pressure_gradient, dynamics)
    ∂y_p_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, AtmosphereModels.y_pressure_gradient, dynamics)
    return slope_x * ∂x_p_ccf + slope_y * ∂y_p_ccf
end

@inline function terrain_horizontal_linearized_pressure_gradient_correction(i, j, k, grid,
                                                                            dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    metrics = dynamics.terrain_metrics
    slope_x = terrain_slope_x_ccf(i, j, k, grid, metrics)
    slope_y = terrain_slope_y_ccf(i, j, k, grid, metrics)
    ∂x_p′_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, x_linearized_pressure_gradient,
                       dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    ∂y_p′_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, y_linearized_pressure_gradient,
                       dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    return slope_x * ∂x_p′_ccf + slope_y * ∂y_p′_ccf
end

@inline function z_linearized_pressure_gradient(i, j, k, grid,
                                                dynamics::TerrainCompressibleDynamics,
                                                ρθ′, Πᴸ, γRᵐᴸ)
    ∂z_p′ = ∂zᶜᶜᶠ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
    correction = terrain_horizontal_linearized_pressure_gradient_correction(i, j, k, grid,
                                                                            dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    return ∂z_p′ - correction
end

@inline function terrain_x_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                        ::SlopeOutsideInterpolation,
                                                        ρθ′, Πᴸ, γRᵐᴸ)
    return ∂xᶠᶜᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function terrain_y_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                        ::SlopeOutsideInterpolation,
                                                        ρθ′, Πᴸ, γRᵐᴸ)
    return ∂yᶜᶠᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function slope_x_times_∂z_linearized_pressure(i, j, k, grid, metrics, ρθ′, Πᴸ, γRᵐᴸ)
    slope = terrain_slope_x_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function slope_y_times_∂z_linearized_pressure(i, j, k, grid, metrics, ρθ′, Πᴸ, γRᵐᴸ)
    slope = terrain_slope_y_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function terrain_x_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                        ::SlopeInsideInterpolation,
                                                        ρθ′, Πᴸ, γRᵐᴸ)
    ∂x_p′ = δxᶠᶜᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ) *
            Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_linearized_pressure,
                       dynamics.terrain_metrics, ρθ′, Πᴸ, γRᵐᴸ)
    return ∂x_p′ - correction
end

@inline function terrain_y_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                        ::SlopeInsideInterpolation,
                                                        ρθ′, Πᴸ, γRᵐᴸ)
    ∂y_p′ = δyᶜᶠᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ) *
            Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_linearized_pressure,
                       dynamics.terrain_metrics, ρθ′, Πᴸ, γRᵐᴸ)
    return ∂y_p′ - correction
end

@inline function terrain_vertical_transport_momentum(i, j, k, grid, metrics,
                                                     ρu, ρv, ρw)
    slope_x = terrain_slope_x_ccf(i, j, k, grid, metrics)
    slope_y = terrain_slope_y_ccf(i, j, k, grid, metrics)

    ρu_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ρu)
    ρv_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, ρv)
    @inbounds ρw_ccf = ρw[i, j, k]

    return ρw_ccf - slope_x * ρu_ccf - slope_y * ρv_ccf
end

@inline function acoustic_vertical_momentum_flux(i, j, k, grid,
                                                 dynamics::TerrainCompressibleDynamics,
                                                 ρu′, ρv′, ρw̃′)
    @inbounds return ρw̃′[i, j, k]
end

@inline function acoustic_stage_vertical_transport_momentum(i, j, k, grid,
                                                            dynamics::TerrainCompressibleDynamics,
                                                            ρu_stage, ρv_stage, ρw_stage)
    return terrain_vertical_transport_momentum(i, j, k, grid, dynamics.terrain_metrics,
                                               ρu_stage, ρv_stage, ρw_stage)
end

@inline function acoustic_recovered_vertical_momentum(i, j, k, grid,
                                                      dynamics::TerrainCompressibleDynamics,
                                                      ρuᴸ, ρvᴸ, ρwᴸ, ρu′, ρv′, ρw̃′)
    metrics = dynamics.terrain_metrics
    slope_x = terrain_slope_x_ccf(i, j, k, grid, metrics)
    slope_y = terrain_slope_y_ccf(i, j, k, grid, metrics)

    ρuᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, total_momentum, ρuᴸ, ρu′)
    ρvᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, total_momentum, ρvᴸ, ρv′)
    ρw̃_stage = acoustic_stage_vertical_transport_momentum(i, j, k, grid, dynamics,
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
            dynamics.density,
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

    metrics = dynamics.terrain_metrics
    slope_x = terrain_slope_x_ccf(i, j, k, grid, metrics)
    slope_y = terrain_slope_y_ccf(i, j, k, grid, metrics)
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
    ∂zᶜᶜᶠ(i, j, k, grid, p_perturbation, p, pᵣ)

@inline terrain_vertical_buoyancy_density(i, j, k, grid, ρ, ::Nothing) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ)

@inline terrain_vertical_buoyancy_density(i, j, k, grid, ρ, ρᵣ) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ_perturbation, ρ, ρᵣ)

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
##### Note: SlopeOutsideInterpolation derives the slope live from grid.z.σ/η
##### via Oceananigans' ∂x_z operators, while SlopeInsideInterpolation reads
##### pre-stored metrics.∂x_h. Both are equivalent for static terrain.

@inline function slope_x_times_∂z(i, j, k, grid, metrics, p)
    slope = terrain_slope_x_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, p)
end

@inline function slope_x_times_∂z_p′(i, j, k, grid, metrics, p, pᵣ)
    slope = terrain_slope_x_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, perturbation_pressure, p, pᵣ)
end

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, ::Nothing)
    ∂x_p = δxᶠᶜᶜ(i, j, k, grid, d.pressure) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, slope_x_times_∂z, d.terrain_metrics, d.pressure)
    return ∂x_p - correction
end

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, pᵣ)
    ∂x_p′ = δxᶠᶜᶜ(i, j, k, grid, perturbation_pressure, d.pressure, pᵣ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, slope_x_times_∂z_p′, d.terrain_metrics, d.pressure, pᵣ)
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

@inline function slope_y_times_∂z(i, j, k, grid, metrics, p)
    slope = terrain_slope_y_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, p)
end

@inline function slope_y_times_∂z_p′(i, j, k, grid, metrics, p, pᵣ)
    slope = terrain_slope_y_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, perturbation_pressure, p, pᵣ)
end

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, ::Nothing)
    ∂y_p = δyᶜᶠᶜ(i, j, k, grid, d.pressure) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, slope_y_times_∂z, d.terrain_metrics, d.pressure)
    return ∂y_p - correction
end

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, pᵣ)
    ∂y_p′ = δyᶜᶠᶜ(i, j, k, grid, perturbation_pressure, d.pressure, pᵣ) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, slope_y_times_∂z_p′, d.terrain_metrics, d.pressure, pᵣ)
    return ∂y_p′ - correction
end

#####
##### Terrain-aware density tendency
#####

function AtmosphereModels.compute_dynamics_tendency!(model::TerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    Gρ = model.timestepper.Gⁿ.ρ
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
    fill_halo_regions!(dynamics.density)
    fill_halo_regions!(prognostic_fields(model.formulation))

    # Compute temperature and pressure (same as non-terrain CompressibleModel)
    launch!(arch, grid, :xyz,
            _compute_temperature_and_pressure!,
            model.temperature,
            dynamics.pressure,
            dynamics.density,
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
    ρ_field = dynamics_density(dynamics)
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

using GPUArraysCore: @allowscalar

using Breeze.Thermodynamics: hydrostatic_pressure

"""
$(TYPEDSIGNATURES)

Fill the 3D fields `pᵣ` and `ρᵣ` with the hydrostatic reference pressure and
density computed by per-column discrete Exner integration. On a terrain-following grid,
different columns have different physical heights at the same computational index `k`,
so the reference state varies horizontally even though the reference atmosphere is
horizontally uniform.

The Exner function is integrated upward at each column using the physical vertical
spacing, ensuring that the discrete hydrostatic balance
```math
\\frac{p_{ref}[k] - p_{ref}[k-1]}{Δz} + g \\frac{ρ_{ref}[k] + ρ_{ref}[k-1]}{2} \\approx 0
```
holds to high accuracy at every interior face. This is essential for reducing the
truncation error in the vertical momentum equation (``-∂p/∂z - gρ``), which would
otherwise be dominated by the near-cancellation of two large terms.

The reference pressure is also used for the perturbation horizontal pressure gradient,
reducing the terrain-following PGF error.
"""
function compute_terrain_reference_state!(pᵣ, ρᵣ, grid, p₀, θᵣ, pˢᵗ, constants)
    Nx, Ny, Nz = size(grid)
    c = Center()
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    g = constants.gravitational_acceleration
    @allowscalar for j in 1:Ny, i in 1:Nx
        πₖ = zero(κ) # initialized at k == 1 below
        for k in 1:Nz
            z = znode(i, j, k, grid, c, c, c)
            θₖ = θᵣ isa Number ? θᵣ : θᵣ(z)

            if k == 1
                # Evaluate the continuous hydrostatic pressure at the local
                # physical height (which varies with terrain) rather than
                # forcing sea-level pressure at every column.
                p_hydro = hydrostatic_pressure(z, p₀, θᵣ, pˢᵗ, constants)
                πₖ = (p_hydro / pˢᵗ)^κ
            else
                z_below = znode(i, j, k - 1, grid, c, c, c)
                θ_below = θᵣ isa Number ? θᵣ : θᵣ(z_below)
                θ_face = (θₖ + θ_below) / 2
                Δz = Δzᶜᶜᶠ(i, j, k, grid)
                πₖ = πₖ - g * Δz / (cᵖᵈ * θ_face)
            end

            pₖ = pˢᵗ * πₖ^(1 / κ)
            ρₖ = pₖ / (Rᵈ * θₖ * πₖ)
            @inbounds pᵣ[i, j, k] = pₖ
            @inbounds ρᵣ[i, j, k] = ρₖ
        end
    end

    fill_halo_regions!(pᵣ)
    fill_halo_regions!(ρᵣ)
    return nothing
end
