#####
##### Diagnostics of the hydrostatic column
#####
#
# Everything here is diagnosed from the prognostic state (pˢ, u, v, ρθ), in the order
#
#   1. layer mass ρ and Exner functions Π, Πᶠ   from pˢ
#   2. geopotential Φ                           from θ = ρθ / ρ, Π and Πᶠ
#   3. vertical mass flux ρw                    from pˢ, u and v, by continuity
#
# The z-face `k` is the lower interface of layer `k`, and the z-face `k + 1` its upper one, so
# the pressure decreases with `k`.
#
# Halos. Every diagnostic is computed in the columns of the interior, region by region on a
# multi-region grid, and the halos of those that the horizontal stencils read (ρ, p, Π, Φ, ρw
# and w) are then filled like those of any field. On a `YinYangGrid` the halo columns are
# therefore interpolated, as scalars, from the diagnostics of the other panel: they are not
# diagnosed again from the interpolated surface pressure and potential temperature. The two
# differ, by the interpolation error, wherever a diagnostic is not linear in the prognostic
# variables. Interpolating the diagnostics is what keeps the resting isentropic atmosphere at
# rest over terrain that lies across the seam: the pressure-gradient force is the gradient of
# Φ + cᵖᵈ θ₀ Π, which is uniform, and an interpolation whose weights sum to one returns it
# unchanged, whereas Π of an interpolated surface pressure is not the interpolated Π. The layer
# mass is linear in pˢ, so its interpolated halos are those of the interpolated pˢ. The Exner
# function on the z-faces is read in the interior columns only, and its halos are not filled.

#####
##### Layer mass and Exner functions
#####

# Pressure on the z-face `k`: p = A + B pˢ
@inline interface_pressure(i, j, k, A, B, pˢ) = @inbounds A[i, j, k] + B[i, j, k] * pˢ[i, j, 1]

# Pressure thickness of layer `k`: Δp = ΔA + ΔB pˢ, linear in pˢ
@inline function layer_pressure_thickness(i, j, k, A, B, pˢ)
    ΔA = @inbounds A[i, j, k] - A[i, j, k+1]
    ΔB = @inbounds B[i, j, k] - B[i, j, k+1]
    return @inbounds ΔA + ΔB * pˢ[i, j, 1]
end

#####
##### Pressure on the coordinate surfaces, for radiative transfer
#####

# The pressures A + B pˢ of the z-faces of every column: the level pressures of a radiative
# transfer model. Level 1 is the ground, at pˢ, and level Nz + 1 the model top, at the top
# pressure, so the pressure thicknesses of the layers sum to pˢ - pᵗ: the whole column, where
# averages of the layer pressures would leave out half of the lowest and of the highest layer.
struct HybridLevelPressure{A, B, P}
    pressure_coefficient :: A          # A at z-faces
    surface_pressure_coefficient :: B  # B at z-faces
    surface_pressure :: P              # pˢ
end

Adapt.adapt_structure(to, pressure::HybridLevelPressure) =
    HybridLevelPressure(adapt(to, pressure.pressure_coefficient),
                        adapt(to, pressure.surface_pressure_coefficient),
                        adapt(to, pressure.surface_pressure))

AtmosphereModels.level_pressure_source(dynamics::HydrostaticDynamics) =
    HybridLevelPressure(dynamics.pressure_coefficient, dynamics.surface_pressure_coefficient, dynamics.surface_pressure)

@inline AtmosphereModels.level_pressure(i, j, k, grid, p, pressure::HybridLevelPressure) =
    interface_pressure(i, j, k, pressure.pressure_coefficient, pressure.surface_pressure_coefficient,
                       pressure.surface_pressure)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Diagnose the layer mass `ρ = Δp / (g Δz)` of `HydrostaticDynamics` from the surface pressure,
with the Exner function on the z-faces, the layer Exner function, and the layer pressure.

The layer mass is the density that converts the prognostic densities to specific quantities, so
it is computed here, at the start of `update_state!`.
"""
function AtmosphereModels.compute_total_density!(model::HydrostaticModel)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration

    @apply_regionally launch!(arch, grid, :xy, _compute_layer_mass_and_exner_functions!,
                              dynamics.density,
                              dynamics.pressure,
                              dynamics.exner_function,
                              dynamics.interface_exner_function,
                              grid,
                              dynamics.pressure_coefficient,
                              dynamics.surface_pressure_coefficient,
                              dynamics.surface_pressure,
                              dynamics.standard_pressure, κ, g)

    fill_halo_regions!(dynamics.density)
    fill_halo_regions!(dynamics.pressure)
    fill_halo_regions!(dynamics.exner_function)

    validate_layer_mass(dynamics)

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Throw an error unless every layer of `HydrostaticDynamics` has a positive mass, that is a
positive pressure thickness `Δp = ΔA + ΔB pˢ`.

The hybrid coefficients are validated at the standard pressure when the model is built, but
`Δp` depends on the surface pressure of each column: coefficients that are valid at 10⁵ Pa can
give a layer of negative mass under a lower surface pressure. The layer mass is therefore
checked every time it is diagnosed, which is at the start of every `update_state!`: when the
state is set, and after every stage of a time step. A layer whose mass is not positive, or not
a number, stops the integration, because every later result would be meaningless.
"""
function validate_layer_mass(dynamics::HydrostaticDynamics)
    ρ = minimum(dynamics.density)

    ρ > 0 ||
        throw(DomainError(ρ, "HydrostaticDynamics has a layer whose mass is not positive: its pressure " *
                             "thickness Δp = ΔA + ΔB pˢ ≤ 0 (or is not a number) for the surface pressure of " *
                             "some column. Either the hybrid coefficients do not suit the surface pressure, " *
                             "or the integration has failed."))

    return nothing
end

# The layer Exner function is the mass-weighted mean of (p / pˢᵗ)^κ over the layer
# (Phillips 1974),
#
#   Π = [p⁻ Πᶠ⁻ - p⁺ Πᶠ⁺] / [(1 + κ) Δp] ,
#
# with ⁻ and ⁺ the lower and upper interfaces, so that Σ Δp Π telescopes to a function of pˢ
# alone. The layer pressure is the pressure whose Exner function is Π.
@kernel function _compute_layer_mass_and_exner_functions!(ρ, p, Π, Πᶠ, grid, A, B, pˢ, pˢᵗ, κ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        p⁻ = interface_pressure(i, j, 1, A, B, pˢ)
        Πᶠ⁻ = (p⁻ / pˢᵗ)^κ
        Πᶠ[i, j, 1] = Πᶠ⁻

        for k in 1:Nz
            p⁺ = interface_pressure(i, j, k+1, A, B, pˢ)
            Πᶠ⁺ = (p⁺ / pˢᵗ)^κ
            Δp = layer_pressure_thickness(i, j, k, A, B, pˢ)
            Πᵏ = (p⁻ * Πᶠ⁻ - p⁺ * Πᶠ⁺) / ((1 + κ) * Δp)

            ρ[i, j, k] = coordinate_layer_density(i, j, k, grid, ρ, Δp, g)
            Π[i, j, k] = Πᵏ
            p[i, j, k] = pˢᵗ * Πᵏ^(1 / κ)
            Πᶠ[i, j, k+1] = Πᶠ⁺

            p⁻ = p⁺
            Πᶠ⁻ = Πᶠ⁺
        end
    end
end

#####
##### Geopotential
#####

# Hydrostatic balance ∂Φ/∂Π = -cᵖᵈ θ, integrated exactly for a layer of uniform θ: from the
# ground up, Φ rises by cᵖᵈ θ (Πᶠ⁻ - Π) to the layer and by cᵖᵈ θ (Πᶠ⁻ - Πᶠ⁺) across it.
@kernel function _compute_geopotential!(Φ, grid, Φˢ, θ, Π, Πᶠ, cᵖᵈ)
    i, j = @index(Global, NTuple)
    compute_column_geopotential!(i, j, grid, Φ, Φˢ, θ, Π, Πᶠ, cᵖᵈ)
end

@inline function compute_column_geopotential!(i, j, grid, Φ, Φˢ, θ, Π, Πᶠ, cᵖᵈ)
    Nz = size(grid, 3)

    @inbounds begin
        Φᶠ = Φˢ[i, j, 1]

        for k in 1:Nz
            Φ[i, j, k] = Φᶠ + cᵖᵈ * θ[i, j, k] * (Πᶠ[i, j, k] - Π[i, j, k])
            Φᶠ += cᵖᵈ * θ[i, j, k] * (Πᶠ[i, j, k] - Πᶠ[i, j, k+1])
        end
    end

    return nothing
end

#####
##### Mass fluxes and continuity
#####

# Mass fluxes through the x- and y-faces of a cell, in kg s⁻¹. These are the fluxes that
# `div_ρUc` transports scalars with, so that a uniform scalar stays uniform.
@inline mass_flux_x(i, j, k, grid, ρ, u) = @inbounds ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Axᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
@inline mass_flux_y(i, j, k, grid, ρ, v) = @inbounds ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Ayᶜᶠᶜ(i, j, k, grid) * v[i, j, k]

# On a pressure-level grid the mass flux is written in the layer mass Δp / g, so that the
# telescoping and positivity identities hold exactly in Δp rather than up to the rounding of ρ Δz
@inline mass_flux_x(i, j, k, grid::PressureLevelGrid, ρ, u) =
    @inbounds ℑxᶠᵃᵃ(i, j, k, grid, layer_mass) * Δyᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
@inline mass_flux_y(i, j, k, grid::PressureLevelGrid, ρ, v) =
    @inbounds ℑyᵃᶠᵃ(i, j, k, grid, layer_mass) * Δxᶜᶠᶜ(i, j, k, grid) * v[i, j, k]

# Horizontal divergence of the mass flux out of layer `k`, per unit area, in kg m⁻² s⁻¹
@inline horizontal_mass_flux_divergence(i, j, k, grid, ρ, u, v) =
    Az⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, mass_flux_x, ρ, u) +
                               δyᵃᶜᵃ(i, j, k, grid, mass_flux_y, ρ, v))

# Column sum of the horizontal mass flux divergence: the rate at which the column loses mass
@inline function column_mass_flux_divergence(i, j, grid, ρ, u, v)
    ΣD = zero(grid)
    for k in 1:size(grid, 3)
        ΣD += horizontal_mass_flux_divergence(i, j, k, grid, ρ, u, v)
    end
    return ΣD
end

# Continuity of layer `k`, whose mass per unit area is (ΔA + ΔB pˢ) / g, reads
#
#   ΔB ∂ₜpˢ / g + D + ρw⁺ - ρw⁻ = 0 ,       ∂ₜpˢ / g = - ΣD ,
#
# with D the horizontal divergence of the layer and ΣD its column sum. No mass crosses the
# ground, so ρw is accumulated upward from zero. It also vanishes at the model top, because ΔB
# sums to one, and is set to zero there exactly so that no rounding error leaks through.
#
# `w` is the velocity through the coordinate surfaces that gives the mass flux ρw with the
# interpolated density of `div_ρUc`.
@kernel function _compute_vertical_mass_flux!(ρw, w, grid, B, ρ, u, v)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    ΣD = column_mass_flux_divergence(i, j, grid, ρ, u, v)

    @inbounds begin
        ρw[i, j, 1] = zero(grid)
        w[i, j, 1] = zero(grid)

        for k in 1:Nz-1
            ΔB = B[i, j, k] - B[i, j, k+1]
            D = horizontal_mass_flux_divergence(i, j, k, grid, ρ, u, v)
            ρw[i, j, k+1] = ρw[i, j, k] - D + ΔB * ΣD
            w[i, j, k+1] = ρw[i, j, k+1] / ℑzᵃᵃᶠ(i, j, k+1, grid, ρ)
        end

        ρw[i, j, Nz+1] = zero(grid)
        w[i, j, Nz+1] = zero(grid)
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Diagnose the geopotential of `HydrostaticDynamics` from hydrostatic balance, and the mass flux
and velocity through the coordinate surfaces from continuity.

These diagnostics are those of dry air, so the state is first checked to be dry: this runs in
every `update_state!`, that is when the state is set and after every stage of a time step.
"""
function AtmosphereModels.compute_auxiliary_dynamics_variables!(model::HydrostaticModel)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    u, v, w = model.velocities
    θ = model.formulation.potential_temperature
    cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity

    validate_dry_state(model)

    @apply_regionally launch!(arch, grid, :xy, _compute_geopotential!,
                              dynamics.geopotential, grid, dynamics.surface_geopotential,
                              θ, dynamics.exner_function, dynamics.interface_exner_function, cᵖᵈ)

    rediagnose_layer_density!(grid, model, θ)

    @apply_regionally launch!(arch, grid, :xy, _compute_vertical_mass_flux!,
                              dynamics.vertical_mass_flux, w, grid,
                              dynamics.surface_pressure_coefficient, dynamics.density, u, v)

    fill_halo_regions!(dynamics.geopotential)
    fill_halo_regions!(dynamics.vertical_mass_flux)
    fill_halo_regions!(w)

    compute_hydrostatic_heights!(dynamics.height_coordinate, grid, dynamics, θ, model.thermodynamic_constants)

    return nothing
end

#####
##### The density of a pressure-level grid
#####

# The layer mass ρ that the coordinate kernels write. On the label grid it is diagnosed from the
# pressure thickness and the fixed spacing; on a pressure-level grid ρ is the physical density of
# the grid, which sets its depths and is diagnosed from the geopotential instead
# (`rediagnose_layer_density!`): it is kept here.
@inline coordinate_layer_density(i, j, k, grid, ρ, Δp, g) = Δp / (g * Δzᶜᶜᶜ(i, j, k, grid))
@inline coordinate_layer_density(i, j, k, grid::PressureLevelGrid, ρ, Δp, g) = @inbounds ρ[i, j, k]

# The scalars that are densities: every prognostic field but the surface pressure, the velocities
# and the tracers, which `update_state!` holds as specific values while the density is diagnosed
# (`tracer_density_to_specific!`) and converts back with the new density
hydrostatic_scalar_densities(model) =
    Tuple(field for (name, field) in pairs(prognostic_fields(model)) if name ∉ (:pˢ, :u, :v, keys(model.tracers)...))

@inline scale_densities!(i, j, k, ::Tuple{}, factor) = nothing

@inline function scale_densities!(i, j, k, fields::Tuple, factor)
    @inbounds first(fields)[i, j, k] *= factor
    scale_densities!(i, j, k, Base.tail(fields), factor)
    return nothing
end

# ρ = Δp / (g Δz) with Δz = cᵖᵈ θᵨ ΔΠᶠ / g the depth of the layer from the hydrostatic integral, the
# depth of the geopotential. The densities are carried with ρ at a fixed specific value.
@kernel function _rediagnose_layer_density!(ρ, grid, densities, θᵨ, Πᶠ, cᵖᵈ, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Δz = physical_layer_depth(i, j, k, θᵨ, Πᶠ, cᵖᵈ, g)
        ρ⁺ = grid_layer_pressure_thickness(i, j, k, grid) / (g * Δz)
        scale_densities!(i, j, k, densities, ρ⁺ / ρ[i, j, k])
        ρ[i, j, k] = ρ⁺
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

On a [`PressureLevelGrid`](@ref Breeze.PressureLevelDiscretization.PressureLevelGrid), diagnose the
density of the grid, which sets its physical depths, from the hydrostatic integral with the
density potential temperature `θᵨ`: `ρ = Δp / (g Δz)` with `Δz = cᵖᵈ θᵨ ΔΠᶠ / g`. Every scalar
density of `model` (and `extra_densities`) is carried with it at a fixed specific value. Nothing on
any other grid, whose layer mass is diagnosed with the surface pressure.

The specific values change by a rounding: `(ρ⁺ / ρ) ρc / ρ⁺` differs from `ρc / ρ` in the last place.
"""
rediagnose_layer_density!(grid, model, θᵨ, extra_densities=()) = nothing

function rediagnose_layer_density!(grid::PressureLevelGrid, model, θᵨ, extra_densities=())
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    cᵖᵈ = constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration
    densities = (hydrostatic_scalar_densities(model)..., extra_densities...)

    launch!(grid.architecture, grid, :xyz, _rediagnose_layer_density!,
            dynamics.density, grid, densities, θᵨ, dynamics.interface_exner_function, cᵖᵈ, g)

    fill_halo_regions!(dynamics.density)
    foreach(fill_halo_regions!, densities)

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The velocities that transport the scalars of `HydrostaticDynamics` on `grid`: `(u, v, w)`, or on a
[`PressureLevelGrid`](@ref Breeze.PressureLevelDiscretization.PressureLevelGrid) `(u, v, W)` with
`W` a [`CoordinateMassFlux`](@ref): the velocity `w` through the coordinate surfaces with the mass
flux `ρw` per unit area that the vertical fluxes of the pressure-level grid carry in place of `ℑz(ρ) w`.
"""
coordinate_transport_velocities(grid, u, v, w, ρw) = (; u, v, w)
coordinate_transport_velocities(grid::PressureLevelGrid, u, v, w, ρw) = (; u, v, w = CoordinateMassFlux(w, ρw))

"""
$(TYPEDEF)

The vertical transport through the coordinate surfaces of a pressure-level grid: indexed, the
velocity `w` through them, which Courant numbers and the explicit scale of adaptive implicit
vertical advection read; its `mass_flux`, ρw per unit area from continuity, is what the vertical
fluxes of the scalars carry (methods in Breeze.Advection), so that they are the mass flux exactly.
"""
struct CoordinateMassFlux{FT, W, M} <: AbstractArray{FT, 3}
    velocity :: W   # w through the coordinate surfaces, m s⁻¹
    mass_flux :: M  # ρw through them, per unit area, kg m⁻² s⁻¹

    CoordinateMassFlux(w::W, ρw::M) where {W, M} = new{eltype(w), W, M}(w, ρw)
end

Base.size(W::CoordinateMassFlux) = size(W.velocity)
Base.@propagate_inbounds Base.getindex(W::CoordinateMassFlux, i, j, k) = W.velocity[i, j, k]
Adapt.adapt_structure(to, W::CoordinateMassFlux) = CoordinateMassFlux(Adapt.adapt(to, W.velocity), Adapt.adapt(to, W.mass_flux))

AtmosphereModels.transport_velocities(model::HydrostaticModel) =
    coordinate_transport_velocities(model.grid, model.velocities.u, model.velocities.v, model.velocities.w,
                                    model.dynamics.vertical_mass_flux)

# The horizontal velocities are prognostic: only their halos need filling
function AtmosphereModels.compute_velocities!(model::HydrostaticModel)
    fill_horizontal_velocity_halos!(model.momentum, boundary_condition_args(model)...)
    return nothing
end
