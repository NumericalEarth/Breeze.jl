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

using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ
using Oceananigans.BoundaryConditions: fill_halo_regions!

using Breeze.TerrainFollowingDiscretization: TerrainMetrics, terrain_slope_x, terrain_slope_y

#####
##### Terrain-aware type alias
#####

const TerrainCompressibleDynamics = CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:TerrainMetrics}
const TerrainCompressibleModel = AtmosphereModel{<:TerrainCompressibleDynamics}

#####
##### Compute contravariant vertical velocity and momentum
#####

"""
    compute_contravariant_velocity!(model::TerrainCompressibleModel)

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
    arch = grid.architecture
    dynamics = model.dynamics

    launch!(arch, grid, :xyz,
            _compute_contravariant_velocity!,
            dynamics.Ω̃, dynamics.ρΩ̃,
            grid, model.velocities, model.momentum,
            dynamics.terrain_metrics)

    fill_halo_regions!(dynamics.Ω̃)
    fill_halo_regions!(dynamics.ρΩ̃)

    return nothing
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

    # Velocities interpolated to (Center, Center, Face) using Oceananigans operators
    u_ccf = ℑxᶜᵃᵃ(i, j, k, grid, velocities.u)
    v_ccf = ℑyᵃᶜᵃ(i, j, k, grid, velocities.v)
    @inbounds w_ccf = velocities.w[i, j, k]

    # Contravariant vertical velocity
    Ω̃_ijk = w_ccf - slope_x * u_ccf - slope_y * v_ccf

    # Momentum interpolated to (Center, Center, Face)
    ρu_ccf = ℑxᶜᵃᵃ(i, j, k, grid, momentum.ρu)
    ρv_ccf = ℑyᵃᶜᵃ(i, j, k, grid, momentum.ρv)
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
#####   (∂p/∂x)_z = (∂p/∂x)_ζ - (∂z/∂x)_ζ · (∂p/∂ζ)
#####
##### Since Oceananigans' ∂xᶠᶜᶜ computes (∂p/∂x)_ζ on the computational grid,
##### we need to subtract the terrain correction term.
#####

@inline function AtmosphereModels.x_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    ∂x_p = ∂xᶠᶜᶜ(i, j, k, grid, d.pressure)
    slope = terrain_slope_x(i, j, k, grid, d.terrain_metrics, Center())
    ∂z_p = ℑxᶠᵃᵃ(i, j, k, grid, ∂zᶜᶜᶠ, d.pressure)
    return ∂x_p - slope * ∂z_p
end

@inline function AtmosphereModels.y_pressure_gradient(i, j, k, grid, d::TerrainCompressibleDynamics)
    ∂y_p = ∂yᶜᶠᶜ(i, j, k, grid, d.pressure)
    slope = terrain_slope_y(i, j, k, grid, d.terrain_metrics, Center())
    ∂z_p = ℑyᵃᶠᵃ(i, j, k, grid, ∂zᶜᶜᶠ, d.pressure)
    return ∂y_p - slope * ∂z_p
end

#####
##### Terrain-aware density tendency
#####

function AtmosphereModels.compute_dynamics_tendency!(model::TerrainCompressibleModel)
    grid = model.grid
    arch = grid.architecture
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
    arch = grid.architecture
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
            model.specific_moisture,
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
