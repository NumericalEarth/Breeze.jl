#####
##### Terrain-following coordinate physics for compressible dynamics
#####
##### For terrain-following coordinates, three modifications are needed:
#####
##### 1. Contravariant vertical velocity Ω̃ replaces w in vertical transport
##### 2. Horizontal pressure gradient includes terrain correction
##### 3. Density tendency uses ρΩ̃ instead of ρw
#####
##### The contravariant vertical velocity is:
#####   Ω̃ = w - (∂z/∂x)_ζ · u - (∂z/∂y)_ζ · v
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

using Breeze.TerrainFollowingDiscretization: TerrainMetrics, SlopeOutsideInterpolation, SlopeInsideInterpolation

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

Compute the contravariant vertical velocity ``\\tilde{\\Omega}`` and
contravariant vertical momentum ``\\rho \\tilde{\\Omega}`` from the
Cartesian velocity and momentum fields.

The contravariant vertical velocity is the velocity component normal
to the terrain-following coordinate surfaces:

```math
\\tilde{\\Omega} = w - \\left(\\frac{\\partial z}{\\partial x}\\right)_\\zeta u
                    - \\left(\\frac{\\partial z}{\\partial y}\\right)_\\zeta v
```
"""
function compute_contravariant_velocity!(model::TerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics

    launch!(arch, grid, :xyz,
            _compute_contravariant_velocity!,
            dynamics.Ω̃, dynamics.ρΩ̃,
            grid, model.velocities, model.momentum,
            dynamics.terrain_metrics)

    # Enforce kinematic BC: Ω̃ = 0 at the terrain surface (bottom face).
    # The ImpenetrableBoundaryCondition sets w = 0 at the bottom, but the
    # correct terrain BC is Ω̃ = 0 (no flow through the terrain surface).
    # Since Ω̃ = w - slope·u, having w = 0 gives Ω̃ = -slope·u ≠ 0 which is
    # a spurious mass flux through the terrain. Setting Ω̃ = 0 directly here
    # ensures no transport through the bottom boundary.
    # Zero bottom face BEFORE filling halos so the BC propagates correctly.
    launch!(arch, grid, :xy, _zero_bottom_face!, dynamics.Ω̃)
    launch!(arch, grid, :xy, _zero_bottom_face!, dynamics.ρΩ̃)

    fill_halo_regions!(dynamics.Ω̃)
    fill_halo_regions!(dynamics.ρΩ̃)

    return nothing
end

@kernel function _zero_bottom_face!(field)
    i, j = @index(Global, NTuple)
    @inbounds field[i, j, 1] = 0
end

@kernel function _compute_contravariant_velocity!(Ω̃, ρΩ̃, grid, velocities, momentum, metrics)
    i, j, k = @index(Global, NTuple)

    # Terrain slope decay factor
    ζ = rnode(k, grid, Face())
    z_top = metrics.z_top
    decay = 1 - ζ / z_top

    # Terrain slopes interpolated to (Center, Center) using Oceananigans operators
    # (handles Flat topologies correctly)
    ∂x_h_cc = ℑxᶜᵃᵃ(i, j, 1, grid, metrics.∂x_h)
    ∂y_h_cc = ℑyᵃᶜᵃ(i, j, 1, grid, metrics.∂y_h)

    slope_x = ∂x_h_cc * decay
    slope_y = ∂y_h_cc * decay

    # Velocities interpolated to (Center, Center, Face).
    # u is at (Face, Center, Center) → ℑx brings to (Center, Center, Center)
    #                                 → ℑz brings to (Center, Center, Face)
    u_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, velocities.u)
    v_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, velocities.v)
    @inbounds w_ccf = velocities.w[i, j, k]

    # Contravariant vertical velocity
    Ω̃_ijk = w_ccf - slope_x * u_ccf - slope_y * v_ccf

    # Momentum interpolated to (Center, Center, Face).
    # ρu is at (Face, Center, Center) → ℑx then ℑz to (Center, Center, Face)
    ρu_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, momentum.ρu)
    ρv_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, momentum.ρv)
    @inbounds ρw_ccf = momentum.ρw[i, j, k]

    # Contravariant vertical momentum
    ρΩ̃_ijk = ρw_ccf - slope_x * ρu_ccf - slope_y * ρv_ccf

    @inbounds begin
        Ω̃[i, j, k] = Ω̃_ijk
        ρΩ̃[i, j, k] = ρΩ̃_ijk
    end
end

#####
##### Transport velocity/momentum interface for terrain-following coordinates
#####

function AtmosphereModels.transport_velocities(model::TerrainCompressibleModel)
    Ω̃ = model.dynamics.Ω̃
    u = model.velocities.u
    v = model.velocities.v
    return (; u, v, w=Ω̃)
end

function AtmosphereModels.transport_momentum(model::TerrainCompressibleModel)
    ρΩ̃ = model.dynamics.ρΩ̃
    ρu = model.momentum.ρu
    ρv = model.momentum.ρv
    return (; ρu, ρv, ρw=ρΩ̃)
end

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

@inline _perturbation_pressure(i, j, k, grid, p, p_ref) = @inbounds p[i, j, k] - p_ref[i, j, k]

@inline function AtmosphereModels.x_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    stencil = d.terrain_metrics.pressure_gradient_stencil
    return _terrain_x_pressure_gradient(i, j, k, grid, d, stencil, d.terrain_reference_pressure)
end

##### Slope-outside-interpolation (default): use Oceananigans' generalized ∂xᶠᶜᶜ
##### which applies the chain-rule correction (∂p/∂x)_z = (∂p/∂x)_ζ - (∂z/∂x)_ζ · (∂p/∂z)

@inline function _terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, ::Nothing)
    return ∂xᶠᶜᶜ(i, j, k, grid, d.pressure)
end

@inline function _terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, p_ref)
    return ∂xᶠᶜᶜ(i, j, k, grid, _perturbation_pressure, d.pressure, p_ref)
end

##### Slope-inside-interpolation (CM1-like): ℑz(ℑx(slope * ∂z(p')))
#####
##### The slope is evaluated at each (Center, Center, Face) stencil point
##### and multiplied by ∂z(p') before the 4-point average to (Face, Center, Center).
#####
##### Note: SlopeOutsideInterpolation derives the slope live from grid.z.σ/η
##### via Oceananigans' ∂x_z operators, while SlopeInsideInterpolation reads
##### pre-stored metrics.∂x_h. Both are equivalent for static terrain.

@inline function _slope_x_times_∂z(i, j, k, grid, metrics, p)
    ∂x_h_cc = ℑxᶜᵃᵃ(i, j, 1, grid, metrics.∂x_h)
    ζ = rnode(k, grid, Face())
    slope = ∂x_h_cc * (1 - ζ / metrics.z_top)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, p)
end

@inline function _slope_x_times_∂z_p′(i, j, k, grid, metrics, p, p_ref)
    ∂x_h_cc = ℑxᶜᵃᵃ(i, j, 1, grid, metrics.∂x_h)
    ζ = rnode(k, grid, Face())
    slope = ∂x_h_cc * (1 - ζ / metrics.z_top)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, _perturbation_pressure, p, p_ref)
end

@inline function _terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, ::Nothing)
    ∂x_p = δxᶠᶜᶜ(i, j, k, grid, d.pressure) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, _slope_x_times_∂z, d.terrain_metrics, d.pressure)
    return ∂x_p - correction
end

@inline function _terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, p_ref)
    ∂x_p′ = δxᶠᶜᶜ(i, j, k, grid, _perturbation_pressure, d.pressure, p_ref) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, _slope_x_times_∂z_p′, d.terrain_metrics, d.pressure, p_ref)
    return ∂x_p′ - correction
end

##### Y-direction pressure gradient

@inline function AtmosphereModels.y_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    stencil = d.terrain_metrics.pressure_gradient_stencil
    return _terrain_y_pressure_gradient(i, j, k, grid, d, stencil, d.terrain_reference_pressure)
end

##### Slope-outside-interpolation (default): use Oceananigans' generalized ∂yᶜᶠᶜ

@inline function _terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, ::Nothing)
    return ∂yᶜᶠᶜ(i, j, k, grid, d.pressure)
end

@inline function _terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, p_ref)
    return ∂yᶜᶠᶜ(i, j, k, grid, _perturbation_pressure, d.pressure, p_ref)
end

##### Slope-inside-interpolation (CM1-like): ℑz(ℑy(slope * ∂z(p')))

@inline function _slope_y_times_∂z(i, j, k, grid, metrics, p)
    ∂y_h_cc = ℑyᵃᶜᵃ(i, j, 1, grid, metrics.∂y_h)
    ζ = rnode(k, grid, Face())
    slope = ∂y_h_cc * (1 - ζ / metrics.z_top)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, p)
end

@inline function _slope_y_times_∂z_p′(i, j, k, grid, metrics, p, p_ref)
    ∂y_h_cc = ℑyᵃᶜᵃ(i, j, 1, grid, metrics.∂y_h)
    ζ = rnode(k, grid, Face())
    slope = ∂y_h_cc * (1 - ζ / metrics.z_top)
    return slope * ∂zᶜᶜᶠ(i, j, k, grid, _perturbation_pressure, p, p_ref)
end

@inline function _terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, ::Nothing)
    ∂y_p = δyᶜᶠᶜ(i, j, k, grid, d.pressure) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, _slope_y_times_∂z, d.terrain_metrics, d.pressure)
    return ∂y_p - correction
end

@inline function _terrain_y_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, p_ref)
    ∂y_p′ = δyᶜᶠᶜ(i, j, k, grid, _perturbation_pressure, d.pressure, p_ref) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, _slope_y_times_∂z_p′, d.terrain_metrics, d.pressure, p_ref)
    return ∂y_p′ - correction
end

#####
##### Terrain-aware density tendency
#####

function AtmosphereModels.compute_dynamics_tendency!(model::TerrainCompressibleModel)
    grid = model.grid
    arch = architecture(grid)
    Gρ = model.timestepper.Gⁿ.ρ
    ρΩ̃ = model.dynamics.ρΩ̃

    launch!(arch, grid, :xyz, _compute_terrain_density_tendency!, Gρ, grid, model.momentum, ρΩ̃)

    return nothing
end

@kernel function _compute_terrain_density_tendency!(Gρ, grid, momentum, ρΩ̃)
    i, j, k = @index(Global, NTuple)
    # Use ρΩ̃ (contravariant momentum) for vertical transport instead of ρw
    @inbounds Gρ[i, j, k] = - divᶜᶜᶜ(i, j, k, grid, momentum.ρu, momentum.ρv, ρΩ̃)
end

#####
##### Hook into auxiliary variable computation to compute Ω̃ and ρΩ̃
#####

function AtmosphereModels.compute_auxiliary_dynamics_variables!(model::TerrainCompressibleModel)
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
##### p_ref and ρ_ref in approximate discrete hydrostatic balance, allowing
##### the vertical PG and buoyancy to be computed in perturbation form:
#####   -(∂p'/∂z) - g ρ'
##### where p' = p - p_ref and ρ' = ρ - ρ_ref are small perturbations.
#####

@inline function AtmosphereModels.z_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    ∂z_p = ∂zᶜᶜᶠ(i, j, k, grid, d.pressure)
    ∂z_pᵣ = _terrain_∂z_reference_pressure(i, j, k, grid, d.terrain_reference_pressure)
    return ∂z_p - ∂z_pᵣ
end

@inline _terrain_∂z_reference_pressure(i, j, k, grid, ::Nothing) = zero(grid)
@inline _terrain_∂z_reference_pressure(i, j, k, grid, p_ref) = ∂zᶜᶜᶠ(i, j, k, grid, p_ref)

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
    ρᵣ = _terrain_reference_density(i, j, k, dynamics.terrain_reference_density)
    return -g * (ρ - ρᵣ)
end

@inline _terrain_reference_density(i, j, k, ::Nothing) = false
@inline _terrain_reference_density(i, j, k, ρ_ref) = @inbounds ρ_ref[i, j, k]

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
