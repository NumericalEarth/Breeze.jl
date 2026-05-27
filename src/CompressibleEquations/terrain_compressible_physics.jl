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
#####   w̃ = w - (∂z/∂x)_ζ · u - (∂z/∂y)_ζ · v
#####
##### The terrain-corrected horizontal pressure gradient is:
#####   (∂p/∂x)_z = (∂p/∂x)_ζ - (∂z/∂x)_ζ · (∂p/∂z)
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
using Oceananigans.BoundaryConditions: fill_halo_regions!

using Breeze.TerrainFollowingDiscretization: TerrainMetrics, SlopeOutsideInterpolation,
                                              SlopeInsideInterpolation,
                                              TFVDRG, ∂z∂x, ∂z∂y

#####
##### Terrain-aware type alias
#####

const TerrainCompressibleDynamics = CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:TerrainMetrics}
const FlatTerrainMetrics = TerrainMetrics{<:Any, <:Any, <:Any, <:Any, <:Any, Val{true}}
const FlatTerrainCompressibleDynamics = CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:FlatTerrainMetrics}
const TerrainCompressibleModel = AtmosphereModel{<:TerrainCompressibleDynamics}
const FlatTerrainCompressibleModel = AtmosphereModel{<:FlatTerrainCompressibleDynamics}

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
\\tilde{w} = w - \\left(\\frac{\\partial z}{\\partial x}\\right)_\\zeta u
                    - \\left(\\frac{\\partial z}{\\partial y}\\right)_\\zeta v
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

    # Enforce kinematic BC: w̃ = 0 at the terrain surface (bottom face).
    # The ImpenetrableBoundaryCondition sets w = 0 at the bottom, but the
    # correct terrain BC is w̃ = 0 (no flow through the terrain surface).
    # Since w̃ = w - slope·u, having w = 0 gives w̃ = -slope·u ≠ 0 which is
    # a spurious mass flux through the terrain. Setting w̃ = 0 directly here
    # ensures no transport through the bottom boundary.
    # Zero bottom face BEFORE filling halos so the BC propagates correctly.
    launch!(arch, grid, :xy, _zero_bottom_face!, w̃)
    launch!(arch, grid, :xy, _zero_bottom_face!, ρw̃)

    fill_halo_regions!(w̃)
    fill_halo_regions!(ρw̃)

    return nothing
end

function compute_contravariant_velocity!(model::FlatTerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics
    w̃ = dynamics.contravariant_vertical_velocity
    ρw̃ = dynamics.contravariant_vertical_momentum

    launch!(arch, grid, :xyz,
            _copy_flat_contravariant_velocity!,
            w̃, ρw̃,
            grid, model.velocities.w, model.momentum.ρw)

    fill_halo_regions!(w̃)
    fill_halo_regions!(ρw̃)

    return nothing
end

@kernel function _copy_flat_contravariant_velocity!(w̃, ρw̃, grid, w, ρw)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        w̃[i, j, k] = w[i, j, k] * (k > 1)
        ρw̃[i, j, k] = ρw[i, j, k] * (k > 1)
    end
end

@kernel function _zero_bottom_face!(field)
    i, j = @index(Global, NTuple)
    @inbounds field[i, j, 1] = 0
end

@kernel function _compute_contravariant_velocity!(w̃, ρw̃, grid, momentum, density, metrics)
    i, j, k = @index(Global, NTuple)

    # Terrain slopes (∂z/∂x, ∂z/∂y)_ζ at (Center, Center, Face). On a BTF/MVD
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

AtmosphereModels.transport_velocities(model::FlatTerrainCompressibleModel) = model.velocities

function outer_step_start_transport_velocities(model::TerrainCompressibleModel)
    w̃ = model.dynamics.contravariant_vertical_velocity
    u = model.velocities.u
    v = model.velocities.v
    return (; u, v, w=w̃)
end

outer_step_start_transport_velocities(model::FlatTerrainCompressibleModel) = model.velocities

function AtmosphereModels.advecting_momentum(model::TerrainCompressibleModel)
    ρw̃ = model.dynamics.contravariant_vertical_momentum
    ρu = model.momentum.ρu
    ρv = model.momentum.ρv
    return (; ρu, ρv, ρw=ρw̃)
end

function AtmosphereModels.advecting_momentum(model::FlatTerrainCompressibleModel)
    ρu = model.momentum.ρu
    ρv = model.momentum.ρv
    ρw = model.momentum.ρw
    return (; ρu, ρv, ρw)
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

@inline function acoustic_x_pressure_gradient(i, j, k, grid, dynamics::TerrainCompressibleDynamics, p)
    return AtmosphereModels.x_pressure_gradient(i, j, k, grid, dynamics)
end

@inline function acoustic_x_pressure_gradient_components(i, j, k, grid,
                                                         dynamics::TerrainCompressibleDynamics, p)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_x_pressure_gradient_components(i, j, k, grid, dynamics,
                                                  stencil,
                                                  dynamics.terrain_reference_pressure)
end

@inline function acoustic_x_pressure_gradient(i, j, k, grid, dynamics::FlatTerrainCompressibleDynamics, p)
    return ∂xᶠᶜᶜ(i, j, k, grid, p)
end

@inline function acoustic_y_pressure_gradient(i, j, k, grid, dynamics::TerrainCompressibleDynamics, p)
    return AtmosphereModels.y_pressure_gradient(i, j, k, grid, dynamics)
end

@inline function acoustic_y_pressure_gradient_components(i, j, k, grid,
                                                         dynamics::TerrainCompressibleDynamics, p)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_y_pressure_gradient_components(i, j, k, grid, dynamics,
                                                  stencil,
                                                  dynamics.terrain_reference_pressure)
end

@inline function acoustic_y_pressure_gradient(i, j, k, grid, dynamics::FlatTerrainCompressibleDynamics, p)
    return ∂yᶜᶠᶜ(i, j, k, grid, p)
end

@inline function acoustic_x_linearized_pressure_gradient(i, j, k, grid,
                                                         dynamics::FlatTerrainCompressibleDynamics,
                                                         ρθ′, Πᴸ, γRᵐᴸ)
    return ∂xᶠᶜᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function acoustic_y_linearized_pressure_gradient(i, j, k, grid,
                                                         dynamics::FlatTerrainCompressibleDynamics,
                                                         ρθ′, Πᴸ, γRᵐᴸ)
    return ∂yᶜᶠᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function acoustic_x_linearized_pressure_gradient(i, j, k, grid,
                                                         dynamics::TerrainCompressibleDynamics,
                                                         ρθ′, Πᴸ, γRᵐᴸ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_x_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                  stencil, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function acoustic_x_linearized_pressure_gradient_components(i, j, k, grid,
                                                                    dynamics::TerrainCompressibleDynamics,
                                                                    ρθ′, Πᴸ, γRᵐᴸ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_x_linearized_pressure_gradient_components(i, j, k, grid, dynamics,
                                                             stencil, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function acoustic_y_linearized_pressure_gradient(i, j, k, grid,
                                                         dynamics::TerrainCompressibleDynamics,
                                                         ρθ′, Πᴸ, γRᵐᴸ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_y_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                  stencil, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function acoustic_y_linearized_pressure_gradient_components(i, j, k, grid,
                                                                    dynamics::TerrainCompressibleDynamics,
                                                                    ρθ′, Πᴸ, γRᵐᴸ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_y_linearized_pressure_gradient_components(i, j, k, grid, dynamics,
                                                             stencil, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function acoustic_x_full_dry_pressure_gradient(i, j, k, grid,
                                                       dynamics::TerrainCompressibleDynamics,
                                                       ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_x_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                stencil, ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
end

@inline function acoustic_x_full_dry_pressure_gradient_components(i, j, k, grid,
                                                                  dynamics::TerrainCompressibleDynamics,
                                                                  ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_x_full_dry_pressure_gradient_components(i, j, k, grid, dynamics,
                                                           stencil, ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
end

@inline function acoustic_y_full_dry_pressure_gradient(i, j, k, grid,
                                                       dynamics::TerrainCompressibleDynamics,
                                                       ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_y_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                stencil, ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
end

@inline function acoustic_y_full_dry_pressure_gradient_components(i, j, k, grid,
                                                                  dynamics::TerrainCompressibleDynamics,
                                                                  ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    return terrain_y_full_dry_pressure_gradient_components(i, j, k, grid, dynamics,
                                                           stencil, ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
end

@inline reference_exner(i, j, k, pᵣ, pˢᵗ, κ) =
    @inbounds (pᵣ[i, j, k] / pˢᵗ)^κ

@inline frozen_exner_perturbation(i, j, k, grid, Πᴸ, pᵣ, pˢᵗ, κ) =
    @inbounds Πᴸ[i, j, k] - reference_exner(i, j, k, pᵣ, pˢᵗ, κ)

@inline function linearized_exner_perturbation(i, j, k, grid,
                                               ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                               γRᵐᴸ, κ, cᵖᵈ)
    @inbounds begin
        p′ = linearized_pressure_perturbation(i, j, k, grid, ρθ′, Πᴸ, γRᵐᴸ)
        return p′ / (ρᴸ[i, j, k] * cᵖᵈ * θᴸ[i, j, k])
    end
end

@inline function terrain_slope_x_ccf(i, j, k, grid, metrics)
    ∂x_h_cc = ℑxᶜᵃᵃ(i, j, 1, grid, metrics.∂x_h)
    ζ = rnode(k, grid, Face())
    return ∂x_h_cc * (1 - ζ / metrics.z_top)
end

@inline function terrain_slope_y_ccf(i, j, k, grid, metrics)
    ∂y_h_cc = ℑyᵃᶜᵃ(i, j, 1, grid, metrics.∂y_h)
    ζ = rnode(k, grid, Face())
    return ∂y_h_cc * (1 - ζ / metrics.z_top)
end

# On a TerrainFollowingVerticalDiscretization grid the coordinate owns the
# slope: take (∂z/∂x)_ζ from the grid operator (which carries the formulation's
# decay — linear for LinearDecay, sinh for SLEVE) and interpolate the x-face
# value to (Center, Center) at the z-face. The `metrics` argument is ignored;
# σ and the slope come from the one coordinate map, so they cannot disagree.
@inline terrain_slope_x_ccf(i, j, k, grid::TFVDRG, metrics) =
    (∂z∂x(i, j, k, grid, Face()) + ∂z∂x(i + 1, j, k, grid, Face())) / 2

@inline terrain_slope_y_ccf(i, j, k, grid::TFVDRG, metrics) =
    (∂z∂y(i, j, k, grid, Face()) + ∂z∂y(i, j + 1, k, grid, Face())) / 2

@inline function terrain_horizontal_pressure_gradient_correction(i, j, k, grid, dynamics, p)
    metrics = dynamics.terrain_metrics
    slope_x = terrain_slope_x_ccf(i, j, k, grid, metrics)
    slope_y = terrain_slope_y_ccf(i, j, k, grid, metrics)
    ∂x_p_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, acoustic_x_pressure_gradient, dynamics, p)
    ∂y_p_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, acoustic_y_pressure_gradient, dynamics, p)
    return slope_x * ∂x_p_ccf + slope_y * ∂y_p_ccf
end

@inline function terrain_horizontal_linearized_pressure_gradient_correction(i, j, k, grid,
                                                                            dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    metrics = dynamics.terrain_metrics
    slope_x = terrain_slope_x_ccf(i, j, k, grid, metrics)
    slope_y = terrain_slope_y_ccf(i, j, k, grid, metrics)
    ∂x_p′_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, acoustic_x_linearized_pressure_gradient,
                       dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    ∂y_p′_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, acoustic_y_linearized_pressure_gradient,
                       dynamics, ρθ′, Πᴸ, γRᵐᴸ)
    return slope_x * ∂x_p′_ccf + slope_y * ∂y_p′_ccf
end

@inline function acoustic_z_linearized_pressure_gradient(i, j, k, grid,
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

@inline function terrain_x_linearized_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                   stencil::SlopeOutsideInterpolation,
                                                                   ρθ′, Πᴸ, γRᵐᴸ)
    gradient = terrain_x_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                      stencil, ρθ′, Πᴸ, γRᵐᴸ)
    return gradient, zero(grid)
end

@inline function terrain_y_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                        ::SlopeOutsideInterpolation,
                                                        ρθ′, Πᴸ, γRᵐᴸ)
    return ∂yᶜᶠᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function terrain_y_linearized_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                   stencil::SlopeOutsideInterpolation,
                                                                   ρθ′, Πᴸ, γRᵐᴸ)
    gradient = terrain_y_linearized_pressure_gradient(i, j, k, grid, dynamics,
                                                      stencil, ρθ′, Πᴸ, γRᵐᴸ)
    return gradient, zero(grid)
end

@inline function terrain_x_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                      ::SlopeOutsideInterpolation,
                                                      ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    pᵣ = dynamics.terrain_reference_pressure
    return ∂xᶠᶜᶜ(i, j, k, grid, full_dry_acoustic_pressure_perturbation,
                 ρθ′, ρᴸ, θᴸ, pᵣ, pˢᵗ, κ, cᵖᵈ)
end

@inline function terrain_x_full_dry_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                 stencil::SlopeOutsideInterpolation,
                                                                 ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    gradient = terrain_x_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                    stencil, ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    return gradient, zero(grid)
end

@inline function terrain_y_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                      ::SlopeOutsideInterpolation,
                                                      ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    pᵣ = dynamics.terrain_reference_pressure
    return ∂yᶜᶠᶜ(i, j, k, grid, full_dry_acoustic_pressure_perturbation,
                 ρθ′, ρᴸ, θᴸ, pᵣ, pˢᵗ, κ, cᵖᵈ)
end

@inline function terrain_y_full_dry_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                 stencil::SlopeOutsideInterpolation,
                                                                 ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    gradient = terrain_y_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                    stencil, ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    return gradient, zero(grid)
end

@inline function slope_x_times_∂z_linearized_pressure(i, j, k, grid, metrics, ρθ′, Πᴸ, γRᵐᴸ)
    slope = terrain_slope_x_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function slope_x_times_∂z_full_dry_pressure(i, j, k, grid, metrics,
                                                    ρθ′, ρᴸ, θᴸ, pᵣ, pˢᵗ, κ, cᵖᵈ)
    slope = terrain_slope_x_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, full_dry_acoustic_pressure_perturbation,
                          ρθ′, ρᴸ, θᴸ, pᵣ, pˢᵗ, κ, cᵖᵈ)
end

@inline function slope_y_times_∂z_linearized_pressure(i, j, k, grid, metrics, ρθ′, Πᴸ, γRᵐᴸ)
    slope = terrain_slope_y_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
end

@inline function slope_y_times_∂z_full_dry_pressure(i, j, k, grid, metrics,
                                                    ρθ′, ρᴸ, θᴸ, pᵣ, pˢᵗ, κ, cᵖᵈ)
    slope = terrain_slope_y_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, full_dry_acoustic_pressure_perturbation,
                          ρθ′, ρᴸ, θᴸ, pᵣ, pˢᵗ, κ, cᵖᵈ)
end

@inline function slope_x_times_∂z_frozen_exner(i, j, k, grid, metrics, Πᴸ, pᵣ, pˢᵗ, κ)
    slope = terrain_slope_x_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, frozen_exner_perturbation, Πᴸ, pᵣ, pˢᵗ, κ)
end

@inline function slope_y_times_∂z_frozen_exner(i, j, k, grid, metrics, Πᴸ, pᵣ, pˢᵗ, κ)
    slope = terrain_slope_y_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, frozen_exner_perturbation, Πᴸ, pᵣ, pˢᵗ, κ)
end

@inline function slope_x_times_∂z_linearized_exner(i, j, k, grid, metrics,
                                                   ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                   γRᵐᴸ, κ, cᵖᵈ)
    slope = terrain_slope_x_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, linearized_exner_perturbation,
                          ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ, κ, cᵖᵈ)
end

@inline function slope_y_times_∂z_linearized_exner(i, j, k, grid, metrics,
                                                   ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                   γRᵐᴸ, κ, cᵖᵈ)
    slope = terrain_slope_y_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, linearized_exner_perturbation,
                          ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ, κ, cᵖᵈ)
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

@inline function terrain_x_linearized_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                   ::SlopeInsideInterpolation,
                                                                   ρθ′, Πᴸ, γRᵐᴸ)
    horizontal_gradient =
        δxᶠᶜᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ) *
        Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_linearized_pressure,
                       dynamics.terrain_metrics, ρθ′, Πᴸ, γRᵐᴸ)
    return horizontal_gradient, -correction
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

@inline function terrain_y_linearized_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                   ::SlopeInsideInterpolation,
                                                                   ρθ′, Πᴸ, γRᵐᴸ)
    horizontal_gradient =
        δyᶜᶠᶜ(i, j, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ) *
        Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_linearized_pressure,
                       dynamics.terrain_metrics, ρθ′, Πᴸ, γRᵐᴸ)
    return horizontal_gradient, -correction
end

@inline function terrain_x_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                      ::SlopeInsideInterpolation,
                                                      ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    horizontal_gradient =
        δxᶠᶜᶜ(i, j, k, grid, full_dry_acoustic_pressure_perturbation,
              ρθ′, ρᴸ, θᴸ, dynamics.terrain_reference_pressure, pˢᵗ, κ, cᵖᵈ) *
        Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_full_dry_pressure,
                       dynamics.terrain_metrics, ρθ′, ρᴸ, θᴸ,
                       dynamics.terrain_reference_pressure, pˢᵗ, κ, cᵖᵈ)
    return horizontal_gradient - correction
end

@inline function terrain_x_full_dry_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                 ::SlopeInsideInterpolation,
                                                                 ρθ′, ρᴸ, θᴸ,
                                                                 pˢᵗ, κ, cᵖᵈ)
    horizontal_gradient =
        δxᶠᶜᶜ(i, j, k, grid, full_dry_acoustic_pressure_perturbation,
              ρθ′, ρᴸ, θᴸ, dynamics.terrain_reference_pressure, pˢᵗ, κ, cᵖᵈ) *
        Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_full_dry_pressure,
                       dynamics.terrain_metrics, ρθ′, ρᴸ, θᴸ,
                       dynamics.terrain_reference_pressure, pˢᵗ, κ, cᵖᵈ)
    return horizontal_gradient, -correction
end

@inline function terrain_y_full_dry_pressure_gradient(i, j, k, grid, dynamics,
                                                      ::SlopeInsideInterpolation,
                                                      ρθ′, ρᴸ, θᴸ, pˢᵗ, κ, cᵖᵈ)
    horizontal_gradient =
        δyᶜᶠᶜ(i, j, k, grid, full_dry_acoustic_pressure_perturbation,
              ρθ′, ρᴸ, θᴸ, dynamics.terrain_reference_pressure, pˢᵗ, κ, cᵖᵈ) *
        Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_full_dry_pressure,
                       dynamics.terrain_metrics, ρθ′, ρᴸ, θᴸ,
                       dynamics.terrain_reference_pressure, pˢᵗ, κ, cᵖᵈ)
    return horizontal_gradient - correction
end

@inline function terrain_y_full_dry_pressure_gradient_components(i, j, k, grid, dynamics,
                                                                 ::SlopeInsideInterpolation,
                                                                 ρθ′, ρᴸ, θᴸ,
                                                                 pˢᵗ, κ, cᵖᵈ)
    horizontal_gradient =
        δyᶜᶠᶜ(i, j, k, grid, full_dry_acoustic_pressure_perturbation,
              ρθ′, ρᴸ, θᴸ, dynamics.terrain_reference_pressure, pˢᵗ, κ, cᵖᵈ) *
        Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_full_dry_pressure,
                       dynamics.terrain_metrics, ρθ′, ρᴸ, θᴸ,
                       dynamics.terrain_reference_pressure, pˢᵗ, κ, cᵖᵈ)
    return horizontal_gradient, -correction
end

@inline function terrain_x_frozen_exner_gradient(i, j, k, grid, dynamics,
                                                 ::SlopeOutsideInterpolation,
                                                 Πᴸ, pᵣ, pˢᵗ, κ)
    return ∂xᶠᶜᶜ(i, j, k, grid, frozen_exner_perturbation, Πᴸ, pᵣ, pˢᵗ, κ)
end

@inline function terrain_x_frozen_exner_gradient_components(i, j, k, grid, dynamics,
                                                            stencil::SlopeOutsideInterpolation,
                                                            Πᴸ, pᵣ, pˢᵗ, κ)
    gradient = terrain_x_frozen_exner_gradient(i, j, k, grid, dynamics,
                                               stencil, Πᴸ, pᵣ, pˢᵗ, κ)
    return gradient, zero(grid)
end

@inline function terrain_y_frozen_exner_gradient(i, j, k, grid, dynamics,
                                                 ::SlopeOutsideInterpolation,
                                                 Πᴸ, pᵣ, pˢᵗ, κ)
    return ∂yᶜᶠᶜ(i, j, k, grid, frozen_exner_perturbation, Πᴸ, pᵣ, pˢᵗ, κ)
end

@inline function terrain_y_frozen_exner_gradient_components(i, j, k, grid, dynamics,
                                                            stencil::SlopeOutsideInterpolation,
                                                            Πᴸ, pᵣ, pˢᵗ, κ)
    gradient = terrain_y_frozen_exner_gradient(i, j, k, grid, dynamics,
                                               stencil, Πᴸ, pᵣ, pˢᵗ, κ)
    return gradient, zero(grid)
end

@inline function terrain_x_linearized_exner_gradient(i, j, k, grid, dynamics,
                                                     ::SlopeOutsideInterpolation,
                                                     ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                     γRᵐᴸ, κ, cᵖᵈ)
    return ∂xᶠᶜᶜ(i, j, k, grid, linearized_exner_perturbation,
                 ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ, κ, cᵖᵈ)
end

@inline function terrain_x_linearized_exner_gradient_components(i, j, k, grid, dynamics,
                                                                stencil::SlopeOutsideInterpolation,
                                                                ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                                γRᵐᴸ, κ, cᵖᵈ)
    gradient = terrain_x_linearized_exner_gradient(i, j, k, grid, dynamics,
                                                   stencil, ρθ′, Πᴸ, pᴸ, ρᴸ,
                                                   θᴸ, γRᵐᴸ, κ, cᵖᵈ)
    return gradient, zero(grid)
end

@inline function terrain_y_linearized_exner_gradient(i, j, k, grid, dynamics,
                                                     ::SlopeOutsideInterpolation,
                                                     ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                     γRᵐᴸ, κ, cᵖᵈ)
    return ∂yᶜᶠᶜ(i, j, k, grid, linearized_exner_perturbation,
                 ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ, κ, cᵖᵈ)
end

@inline function terrain_y_linearized_exner_gradient_components(i, j, k, grid, dynamics,
                                                                stencil::SlopeOutsideInterpolation,
                                                                ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                                γRᵐᴸ, κ, cᵖᵈ)
    gradient = terrain_y_linearized_exner_gradient(i, j, k, grid, dynamics,
                                                   stencil, ρθ′, Πᴸ, pᴸ, ρᴸ,
                                                   θᴸ, γRᵐᴸ, κ, cᵖᵈ)
    return gradient, zero(grid)
end

@inline function terrain_x_frozen_exner_gradient(i, j, k, grid, dynamics,
                                                 ::SlopeInsideInterpolation,
                                                 Πᴸ, pᵣ, pˢᵗ, κ)
    ∂x_Πᴸ = δxᶠᶜᶜ(i, j, k, grid, frozen_exner_perturbation, Πᴸ, pᵣ, pˢᵗ, κ) *
            Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_frozen_exner,
                       dynamics.terrain_metrics, Πᴸ, pᵣ, pˢᵗ, κ)
    return ∂x_Πᴸ - correction
end

@inline function terrain_x_frozen_exner_gradient_components(i, j, k, grid, dynamics,
                                                            ::SlopeInsideInterpolation,
                                                            Πᴸ, pᵣ, pˢᵗ, κ)
    horizontal_gradient =
        δxᶠᶜᶜ(i, j, k, grid, frozen_exner_perturbation, Πᴸ, pᵣ, pˢᵗ, κ) *
        Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_frozen_exner,
                       dynamics.terrain_metrics, Πᴸ, pᵣ, pˢᵗ, κ)
    return horizontal_gradient, -correction
end

@inline function terrain_y_frozen_exner_gradient(i, j, k, grid, dynamics,
                                                 ::SlopeInsideInterpolation,
                                                 Πᴸ, pᵣ, pˢᵗ, κ)
    ∂y_Πᴸ = δyᶜᶠᶜ(i, j, k, grid, frozen_exner_perturbation, Πᴸ, pᵣ, pˢᵗ, κ) *
            Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_frozen_exner,
                       dynamics.terrain_metrics, Πᴸ, pᵣ, pˢᵗ, κ)
    return ∂y_Πᴸ - correction
end

@inline function terrain_y_frozen_exner_gradient_components(i, j, k, grid, dynamics,
                                                            ::SlopeInsideInterpolation,
                                                            Πᴸ, pᵣ, pˢᵗ, κ)
    horizontal_gradient =
        δyᶜᶠᶜ(i, j, k, grid, frozen_exner_perturbation, Πᴸ, pᵣ, pˢᵗ, κ) *
        Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_frozen_exner,
                       dynamics.terrain_metrics, Πᴸ, pᵣ, pˢᵗ, κ)
    return horizontal_gradient, -correction
end

@inline function terrain_x_linearized_exner_gradient(i, j, k, grid, dynamics,
                                                     ::SlopeInsideInterpolation,
                                                     ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                     γRᵐᴸ, κ, cᵖᵈ)
    ∂x_Π′ = δxᶠᶜᶜ(i, j, k, grid, linearized_exner_perturbation,
                  ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ, κ, cᵖᵈ) *
            Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_linearized_exner,
                       dynamics.terrain_metrics, ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                       γRᵐᴸ, κ, cᵖᵈ)
    return ∂x_Π′ - correction
end

@inline function terrain_x_linearized_exner_gradient_components(i, j, k, grid, dynamics,
                                                                ::SlopeInsideInterpolation,
                                                                ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                                γRᵐᴸ, κ, cᵖᵈ)
    horizontal_gradient =
        δxᶠᶜᶜ(i, j, k, grid, linearized_exner_perturbation,
              ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ, κ, cᵖᵈ) *
        Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_linearized_exner,
                       dynamics.terrain_metrics, ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                       γRᵐᴸ, κ, cᵖᵈ)
    return horizontal_gradient, -correction
end

@inline function terrain_y_linearized_exner_gradient(i, j, k, grid, dynamics,
                                                     ::SlopeInsideInterpolation,
                                                     ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                     γRᵐᴸ, κ, cᵖᵈ)
    ∂y_Π′ = δyᶜᶠᶜ(i, j, k, grid, linearized_exner_perturbation,
                  ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ, κ, cᵖᵈ) *
            Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_linearized_exner,
                       dynamics.terrain_metrics, ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                       γRᵐᴸ, κ, cᵖᵈ)
    return ∂y_Π′ - correction
end

@inline function terrain_y_linearized_exner_gradient_components(i, j, k, grid, dynamics,
                                                                ::SlopeInsideInterpolation,
                                                                ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                                                                γRᵐᴸ, κ, cᵖᵈ)
    horizontal_gradient =
        δyᶜᶠᶜ(i, j, k, grid, linearized_exner_perturbation,
              ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ, κ, cᵖᵈ) *
        Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_linearized_exner,
                       dynamics.terrain_metrics, ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ,
                       γRᵐᴸ, κ, cᵖᵈ)
    return horizontal_gradient, -correction
end

@inline function cm1_style_x_exner_pressure_acceleration_components(i, j, k, grid,
                                                                    dynamics::TerrainCompressibleDynamics,
                                                                    ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ,
                                                                    apply_pressure_gradient,
                                                                    pˢᵗ, κ, cᵖᵈ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    pᵣ = dynamics.terrain_reference_pressure
    frozen_horizontal_gradient, frozen_terrain_gradient =
        terrain_x_frozen_exner_gradient_components(i, j, k, grid, dynamics,
                                                   stencil, Πᴸ, pᵣ, pˢᵗ, κ)
    perturbation_horizontal_gradient, perturbation_terrain_gradient =
        terrain_x_linearized_exner_gradient_components(i, j, k, grid, dynamics,
                                                       stencil, ρθ′, Πᴸ, pᴸ, ρᴸ,
                                                       θᴸ, γRᵐᴸ, κ, cᵖᵈ)
    perturbation_factor = ifelse(apply_pressure_gradient, one(cᵖᵈ), zero(cᵖᵈ))
    θ_face = ℑxᶠᵃᵃ(i, j, k, grid, θᴸ)
    frozen_gradient = frozen_horizontal_gradient + frozen_terrain_gradient
    perturbation_gradient =
        perturbation_factor * (perturbation_horizontal_gradient + perturbation_terrain_gradient)
    frozen_acceleration = -cᵖᵈ * θ_face * frozen_gradient
    perturbation_acceleration = -cᵖᵈ * θ_face * perturbation_gradient
    frozen_horizontal_acceleration = -cᵖᵈ * θ_face * frozen_horizontal_gradient
    frozen_terrain_acceleration = -cᵖᵈ * θ_face * frozen_terrain_gradient
    perturbation_horizontal_acceleration =
        -cᵖᵈ * θ_face * perturbation_factor * perturbation_horizontal_gradient
    perturbation_terrain_acceleration =
        -cᵖᵈ * θ_face * perturbation_factor * perturbation_terrain_gradient
    horizontal_acceleration =
        frozen_horizontal_acceleration + perturbation_horizontal_acceleration
    terrain_acceleration =
        frozen_terrain_acceleration + perturbation_terrain_acceleration
    return frozen_acceleration + perturbation_acceleration,
           frozen_acceleration, perturbation_acceleration,
           horizontal_acceleration, terrain_acceleration,
           frozen_horizontal_acceleration, frozen_terrain_acceleration,
           perturbation_horizontal_acceleration, perturbation_terrain_acceleration
end

@inline function cm1_style_y_exner_pressure_acceleration_components(i, j, k, grid,
                                                                    dynamics::TerrainCompressibleDynamics,
                                                                    ρθ′, Πᴸ, pᴸ, ρᴸ, θᴸ, γRᵐᴸ,
                                                                    apply_pressure_gradient,
                                                                    pˢᵗ, κ, cᵖᵈ)
    stencil = dynamics.terrain_metrics.pressure_gradient_stencil
    pᵣ = dynamics.terrain_reference_pressure
    frozen_horizontal_gradient, frozen_terrain_gradient =
        terrain_y_frozen_exner_gradient_components(i, j, k, grid, dynamics,
                                                   stencil, Πᴸ, pᵣ, pˢᵗ, κ)
    perturbation_horizontal_gradient, perturbation_terrain_gradient =
        terrain_y_linearized_exner_gradient_components(i, j, k, grid, dynamics,
                                                       stencil, ρθ′, Πᴸ, pᴸ, ρᴸ,
                                                       θᴸ, γRᵐᴸ, κ, cᵖᵈ)
    perturbation_factor = ifelse(apply_pressure_gradient, one(cᵖᵈ), zero(cᵖᵈ))
    θ_face = ℑyᵃᶠᵃ(i, j, k, grid, θᴸ)
    frozen_gradient = frozen_horizontal_gradient + frozen_terrain_gradient
    perturbation_gradient =
        perturbation_factor * (perturbation_horizontal_gradient + perturbation_terrain_gradient)
    frozen_acceleration = -cᵖᵈ * θ_face * frozen_gradient
    perturbation_acceleration = -cᵖᵈ * θ_face * perturbation_gradient
    frozen_horizontal_acceleration = -cᵖᵈ * θ_face * frozen_horizontal_gradient
    frozen_terrain_acceleration = -cᵖᵈ * θ_face * frozen_terrain_gradient
    perturbation_horizontal_acceleration =
        -cᵖᵈ * θ_face * perturbation_factor * perturbation_horizontal_gradient
    perturbation_terrain_acceleration =
        -cᵖᵈ * θ_face * perturbation_factor * perturbation_terrain_gradient
    horizontal_acceleration =
        frozen_horizontal_acceleration + perturbation_horizontal_acceleration
    terrain_acceleration =
        frozen_terrain_acceleration + perturbation_terrain_acceleration
    return frozen_acceleration + perturbation_acceleration,
           frozen_acceleration, perturbation_acceleration,
           horizontal_acceleration, terrain_acceleration,
           frozen_horizontal_acceleration, frozen_terrain_acceleration,
           perturbation_horizontal_acceleration, perturbation_terrain_acceleration
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

@inline function acoustic_stage_vertical_transport_momentum(i, j, k, grid,
                                                            dynamics::FlatTerrainCompressibleDynamics,
                                                            ρu_stage, ρv_stage, ρw_stage)
    @inbounds return ρw_stage[i, j, k]
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

@inline function acoustic_recovered_vertical_momentum(i, j, k, grid,
                                                      dynamics::FlatTerrainCompressibleDynamics,
                                                      ρuᴸ, ρvᴸ, ρwᴸ, ρu′, ρv′, ρw′)
    @inbounds return ρwᴸ[i, j, k] + ρw′[i, j, k]
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
    horizontal_pressure_gradient = terrain_horizontal_pressure_gradient_correction(i, j, k, grid, dynamics, pᴸ)

    @inbounds Gˢρw̃[i, j, k] = (Gⁿρw[i, j, k] -
                                horizontal_slow_tendency -
                                vertical_pressure_tendency_factor * ∂z_p′ +
                                horizontal_pressure_gradient -
                                g * ρ′ᶜᶜᶠ) * (k > 1)
end

@inline terrain_vertical_pressure_gradient(i, j, k, grid, p, ::Nothing) =
    ∂zᶜᶜᶠ(i, j, k, grid, p)

@inline terrain_vertical_pressure_gradient(i, j, k, grid, p, p_ref) =
    ∂zᶜᶜᶠ(i, j, k, grid, p_perturbation, p, p_ref)

@inline terrain_vertical_buoyancy_density(i, j, k, grid, ρ, ::Nothing) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ)

@inline terrain_vertical_buoyancy_density(i, j, k, grid, ρ, ρ_ref) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ_perturbation, ρ, ρ_ref)

#####
##### Terrain-corrected pressure gradient
#####
##### The true horizontal pressure gradient at constant z is:
#####   (∂p/∂x)_z = (∂p/∂x)_ζ - (∂z/∂x)_ζ · (∂p/∂z)
#####
##### For SlopeOutsideInterpolation (default), Oceananigans' generalized ∂xᶠᶜᶜ
##### on MutableVerticalDiscretization grids computes this chain-rule correction
##### automatically. For SlopeInsideInterpolation, we use basic δx/Δx operators
##### to compute (∂p/∂x)_ζ, then multiply the slope inside the interpolation.
#####
##### When a terrain reference pressure p_ref(z_physical) is available, the PG is
##### computed using perturbation pressure p' = p - p_ref. Since p_ref depends only
##### on physical height z, its true horizontal gradient (∂p_ref/∂x)_z = 0 exactly.
##### The perturbation terms are much smaller than the full pressure terms, which
##### reduces the truncation error from the near-cancellation of the two large terms.
##### This is the standard approach for reducing PGF errors in terrain-following
##### (sigma) coordinate models (Klemp, 2011).
#####

@inline perturbation_pressure(i, j, k, grid, p, p_ref) = @inbounds p[i, j, k] - p_ref[i, j, k]

@inline function AtmosphereModels.x_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    stencil = d.terrain_metrics.pressure_gradient_stencil
    return terrain_x_pressure_gradient(i, j, k, grid, d, stencil, d.terrain_reference_pressure)
end

@inline function AtmosphereModels.x_pressure_gradient(i, j, k, grid, d::FlatTerrainCompressibleDynamics)
    return ∂xᶠᶜᶜ(i, j, k, grid, d.pressure)
end

##### Slope-outside-interpolation (default): use Oceananigans' generalized ∂xᶠᶜᶜ
##### which applies the chain-rule correction (∂p/∂x)_z = (∂p/∂x)_ζ - (∂z/∂x)_ζ · (∂p/∂z)

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, ::Nothing)
    return ∂xᶠᶜᶜ(i, j, k, grid, d.pressure)
end

@inline function terrain_x_pressure_gradient_components(i, j, k, grid, d,
                                                        stencil::SlopeOutsideInterpolation,
                                                        ::Nothing)
    gradient = terrain_x_pressure_gradient(i, j, k, grid, d, stencil, nothing)
    return gradient, zero(grid)
end

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, p_ref)
    return ∂xᶠᶜᶜ(i, j, k, grid, perturbation_pressure, d.pressure, p_ref)
end

@inline function terrain_x_pressure_gradient_components(i, j, k, grid, d,
                                                        stencil::SlopeOutsideInterpolation,
                                                        p_ref)
    gradient = terrain_x_pressure_gradient(i, j, k, grid, d, stencil, p_ref)
    return gradient, zero(grid)
end

##### Slope-inside-interpolation (CM1-like): ℑz(ℑx(slope * ∂z(p')))
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

@inline function slope_x_times_∂z_p′(i, j, k, grid, metrics, p, p_ref)
    slope = terrain_slope_x_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, perturbation_pressure, p, p_ref)
end

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, ::Nothing)
    ∂x_p = δxᶠᶜᶜ(i, j, k, grid, d.pressure) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, slope_x_times_∂z, d.terrain_metrics, d.pressure)
    return ∂x_p - correction
end

@inline function terrain_x_pressure_gradient_components(i, j, k, grid, d,
                                                        ::SlopeInsideInterpolation,
                                                        ::Nothing)
    horizontal_gradient = δxᶠᶜᶜ(i, j, k, grid, d.pressure) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z, d.terrain_metrics, d.pressure)
    return horizontal_gradient, -correction
end

@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, p_ref)
    ∂x_p′ = δxᶠᶜᶜ(i, j, k, grid, perturbation_pressure, d.pressure, p_ref) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, slope_x_times_∂z_p′, d.terrain_metrics, d.pressure, p_ref)
    return ∂x_p′ - correction
end

@inline function terrain_x_pressure_gradient_components(i, j, k, grid, d,
                                                        ::SlopeInsideInterpolation,
                                                        p_ref)
    horizontal_gradient =
        δxᶠᶜᶜ(i, j, k, grid, perturbation_pressure, d.pressure, p_ref) *
        Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ,
                       slope_x_times_∂z_p′, d.terrain_metrics, d.pressure, p_ref)
    return horizontal_gradient, -correction
end

##### Y-direction pressure gradient

@inline function AtmosphereModels.y_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    stencil = d.terrain_metrics.pressure_gradient_stencil
    return terrain_y_pressure_gradient(i, j, k, grid, d, stencil, d.terrain_reference_pressure)
end

@inline function AtmosphereModels.y_pressure_gradient(i, j, k, grid, d::FlatTerrainCompressibleDynamics)
    return ∂yᶜᶠᶜ(i, j, k, grid, d.pressure)
end

##### Slope-outside-interpolation (default): use Oceananigans' generalized ∂yᶜᶠᶜ

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, ::Nothing)
    return ∂yᶜᶠᶜ(i, j, k, grid, d.pressure)
end

@inline function terrain_y_pressure_gradient_components(i, j, k, grid, d,
                                                        stencil::SlopeOutsideInterpolation,
                                                        ::Nothing)
    gradient = terrain_y_pressure_gradient(i, j, k, grid, d, stencil, nothing)
    return gradient, zero(grid)
end

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, p_ref)
    return ∂yᶜᶠᶜ(i, j, k, grid, perturbation_pressure, d.pressure, p_ref)
end

@inline function terrain_y_pressure_gradient_components(i, j, k, grid, d,
                                                        stencil::SlopeOutsideInterpolation,
                                                        p_ref)
    gradient = terrain_y_pressure_gradient(i, j, k, grid, d, stencil, p_ref)
    return gradient, zero(grid)
end

##### Slope-inside-interpolation (CM1-like): ℑz(ℑy(slope * ∂z(p')))

@inline function slope_y_times_∂z(i, j, k, grid, metrics, p)
    slope = terrain_slope_y_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, p)
end

@inline function slope_y_times_∂z_p′(i, j, k, grid, metrics, p, p_ref)
    slope = terrain_slope_y_ccf(i, j, k, grid, metrics)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, perturbation_pressure, p, p_ref)
end

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, ::Nothing)
    ∂y_p = δyᶜᶠᶜ(i, j, k, grid, d.pressure) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, slope_y_times_∂z, d.terrain_metrics, d.pressure)
    return ∂y_p - correction
end

@inline function terrain_y_pressure_gradient_components(i, j, k, grid, d,
                                                        ::SlopeInsideInterpolation,
                                                        ::Nothing)
    horizontal_gradient = δyᶜᶠᶜ(i, j, k, grid, d.pressure) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z, d.terrain_metrics, d.pressure)
    return horizontal_gradient, -correction
end

@inline function terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, p_ref)
    ∂y_p′ = δyᶜᶠᶜ(i, j, k, grid, perturbation_pressure, d.pressure, p_ref) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, slope_y_times_∂z_p′, d.terrain_metrics, d.pressure, p_ref)
    return ∂y_p′ - correction
end

@inline function terrain_y_pressure_gradient_components(i, j, k, grid, d,
                                                        ::SlopeInsideInterpolation,
                                                        p_ref)
    horizontal_gradient =
        δyᶜᶠᶜ(i, j, k, grid, perturbation_pressure, d.pressure, p_ref) *
        Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ,
                       slope_y_times_∂z_p′, d.terrain_metrics, d.pressure, p_ref)
    return horizontal_gradient, -correction
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

function AtmosphereModels.compute_dynamics_tendency!(model::FlatTerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    Gρ = model.timestepper.Gⁿ.ρ
    momentum = model.momentum
    td = model.dynamics.time_discretization

    launch!(arch, grid, :xyz, _compute_density_tendency!, Gρ, grid, momentum, td)

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

function AtmosphereModels.compute_auxiliary_dynamics_variables!(model::FlatTerrainCompressibleModel)
    compute_terrain_temperature_and_pressure!(model)
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
##### p_ref and ρ_ref in approximate discrete hydrostatic balance, allowing
##### the vertical PG and buoyancy to be computed in perturbation form:
#####   -(∂p'/∂z) - g ρ'
##### where p' = p - p_ref and ρ' = ρ - ρ_ref are small perturbations.
#####

@inline function AtmosphereModels.z_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    ∂z_p = ∂zᶜᶜᶠ(i, j, k, grid, d.pressure)
    ∂z_pᵣ = terrain_∂z_reference_pressure(i, j, k, grid, d.terrain_reference_pressure)
    return ∂z_p - ∂z_pᵣ
end

@inline terrain_∂z_reference_pressure(i, j, k, grid, ::Nothing) = zero(grid)
@inline terrain_∂z_reference_pressure(i, j, k, grid, p_ref) = ∂zᶜᶜᶠ(i, j, k, grid, p_ref)

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
@inline terrain_reference_density(i, j, k, ρ_ref) = @inbounds ρ_ref[i, j, k]

#####
##### 3D terrain reference state via per-column discrete Exner integration
#####

using GPUArraysCore: @allowscalar

using Breeze.Thermodynamics: evaluate_profile, hydrostatic_pressure

"""
$(TYPEDSIGNATURES)

Fill the 3D fields `p_ref` and `ρ_ref` with the hydrostatic reference pressure and
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
function compute_terrain_reference_state!(p_ref, ρ_ref, grid, p₀, θᵣ, pˢᵗ, constants)
    Nx, Ny, Nz = size(grid)
    c = Center()
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ
    g = constants.gravitational_acceleration
    @allowscalar for j in 1:Ny, i in 1:Nx
        πₖ = zero(κ) # initialized at k == 1 below
        for k in 1:Nz
            z_phys = znode(i, j, k, grid, c, c, c)
            θₖ = evaluate_profile(θᵣ, z_phys)

            if k == 1
                # Evaluate the continuous hydrostatic pressure at the local
                # physical height (which varies with terrain) rather than
                # forcing sea-level pressure at every column.
                p_hydro = hydrostatic_pressure(z_phys, p₀, θᵣ, pˢᵗ, constants)
                πₖ = (p_hydro / pˢᵗ)^κ
            else
                z_below = znode(i, j, k - 1, grid, c, c, c)
                θ_below = evaluate_profile(θᵣ, z_below)
                θ_face = (θₖ + θ_below) / 2
                Δz = Δzᶜᶜᶠ(i, j, k, grid)
                πₖ = πₖ - g * Δz / (cᵖᵈ * θ_face)
            end

            pₖ = pˢᵗ * πₖ^(1 / κ)
            ρₖ = pₖ / (Rᵈ * θₖ * πₖ)
            @inbounds p_ref[i, j, k] = pₖ
            @inbounds ρ_ref[i, j, k] = ρₖ
        end
    end

    fill_halo_regions!(p_ref)
    fill_halo_regions!(ρ_ref)
    return nothing
end
