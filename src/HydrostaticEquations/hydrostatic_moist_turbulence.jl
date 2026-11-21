"""
    HydrostaticMoistTurbulence

Adapter of [`TKEBasedTurbulenceClosure`](@ref Breeze.TurbulenceClosures.TKEBasedTurbulenceClosure)
with its static stability (`MoistStaticStability` by default) to moist `HydrostaticDynamics`.

Nothing of the closure is redefined here. Its kernels and kernel functions (static stability,
mixing length, stability functions, diffusivities, the sink rate of the turbulent kinetic
energy) are called as they are. What this module adds is

1. a geometry, [`PhysicalHeightGrid`](@ref): the vertical spacings of the hydrostatic column in
   meters, `ΔΦ / g`, behind the operators that the closure calls on its `grid`. The grid of
   `HydrostaticDynamics` labels coordinate surfaces, and its `Δz` is not a height.
2. the conservative insertion of the turbulent fluxes in the budgets of the total-mass coordinate:
   one backward Euler step per time step and per column for the water that does not fall, the
   enthalpy, the velocities and the turbulent kinetic energy, each a tridiagonal solve.
3. the heat that closes the energy budget: the enthalpy of a layer changes by the divergence of
   the enthalpy flux, minus the energy the turbulent kinetic energy receives from buoyancy, plus
   what it dissipates.
4. the insertion of surface fluxes, given as fields of stress, sensible heat flux and vapor
   flux (the quantities a coupler computes with its bulk formula), and a filler of those fields
   that evaluates the existing bulk boundary conditions of Breeze.

The derivation is `DERIVATION.md` of the moist closure workstream.
"""
module HydrostaticMoistTurbulence

export MoistTurbulentMixing, moist_turbulent_mixing_step!, BulkSurfaceExchange, PhysicalHeightGrid, MoistColumnPhysics

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, Center, Face, CenterField, XFaceField, YFaceField, ZFaceField, Field
using Oceananigans.BoundaryConditions: fill_halo_regions!, getbc, Bottom
using Oceananigans.Grids: AbstractGrid, topology
using Oceananigans.Models: boundary_condition_args
using Oceananigans.Operators: Operators, Δzᶜᶜᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.TimeSteppers: update_state!, time_discretization
using Oceananigans.TurbulenceClosures: TurbulenceClosures as OceanTurbulenceClosures,
                                       VerticallyImplicitTimeDiscretization, build_closure_fields
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: ℑbzᵃᵃᶜ
using Oceananigans.Utils: KernelParameters, launch!, @apply_regionally

using Breeze.AtmosphereModels: AtmosphereModelBuoyancy, specific_prognostic_moisture, dynamics_thermodynamic_fields,
                               log_density_potential_temperature
using Breeze.BoundaryConditions: BoundaryConditions as AtmosphereBoundaryConditions
using Breeze.HydrostaticEquations: HydrostaticPressureGrid, HydrostaticHeightCoordinate, height_grid_or,
                                   MoistHydrostaticModel, column_physics_in_time_step, warm_rain_column_physics!, hydrostatic_water_fields,
                                   water_densities, sum_at, layer_pressure_thickness,
                                   interface_potential_temperature, height_between_layers, physical_layer_depth,
                                   layer_mass_per_area, diffuse_column!, face_conductance, _diffuse_velocity!,
                                   change_column_surface_pressure!, column_mass_densities,
                                   from_x_faces, from_y_faces, specific_kinetic_energy, surface_pressure_kinetic_energy
using Breeze.Thermodynamics: dry_air_gas_constant, absolute_zero_latent_heat
using Breeze.TurbulenceClosures: TKEBasedTurbulenceClosure, TKE_NAME, buoyancy_productionᶜᶜᶠ,
                                 log_dry_potential_temperatureᶜᶜᶜ,
                                 clip_negative_halo_diffusivities!,
                                 _compute_tke_static_stability!, _compute_mixing_length!,
                                 _compute_tke_closure_fields!, _compute_tke_implicit_linear_coefficient!
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS, @regional

using DocStringExtensions: TYPEDEF

#####
##### Physical heights behind the operators of a grid
#####

"""
$(TYPEDEF)

The grid of `HydrostaticDynamics` with the vertical spacings of its hydrostatic column, in
meters, in place of the spacings of its coordinate: `Δzᶜᶜᶜ` is the depth `ΔΦ / g` of a layer,
`Δzᶜᶜᶠ` the height between two layers, and `height_above_bottomᶜᶜᶠ` the height of a coordinate
surface above the ground. Kernel functions written for a height coordinate, called with this
grid, differentiate and integrate in height. The horizontal operators and the masks are those
of the underlying grid.

The three fields are diagnosed from the state by `compute_physical_heights!` and are valid for
that state only. The grid is built where a kernel is launched, from the grid of a region and the
fields of that region.
"""
struct PhysicalHeightGrid{FT, TX, TY, TZ, Arch, G, C, F} <: AbstractGrid{FT, TX, TY, TZ, Arch, Nothing}
    architecture :: Arch
    underlying_grid :: G
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Hx :: Int
    Hy :: Int
    Hz :: Int
    layer_depth :: C        # depth of the layers, at cell centers, m
    interface_spacing :: F  # height between two layers, at z-faces, m
    interface_height :: F   # height of the z-faces above the ground, m

    function PhysicalHeightGrid(arch::Arch, grid::G, Δhᶜ::C, Δhᶠ::F, hᶠ::F) where {Arch, G, C, F}
        FT = eltype(grid)
        TX, TY, TZ = topology(grid)
        return new{FT, TX, TY, TZ, Arch, G, C, F}(arch, grid, grid.Nx, grid.Ny, grid.Nz, grid.Hx, grid.Hy, grid.Hz,
                                                  Δhᶜ, Δhᶠ, hᶠ)
    end
end

PhysicalHeightGrid(grid, heights) =
    PhysicalHeightGrid(grid.architecture, grid, heights.layer_depth, heights.interface_spacing, heights.interface_height)

height_fields(grid) = (layer_depth = CenterField(grid),          # depth of the layers, m
                       interface_spacing = ZFaceField(grid),     # height between two layers, m
                       interface_height = ZFaceField(grid))      # height of the z-faces above the ground, m

Adapt.adapt_structure(to, grid::PhysicalHeightGrid) =
    PhysicalHeightGrid(nothing, adapt(to, grid.underlying_grid), adapt(to, grid.layer_depth),
                       adapt(to, grid.interface_spacing), adapt(to, grid.interface_height))

Base.summary(grid::PhysicalHeightGrid) = string("PhysicalHeightGrid of ", summary(grid.underlying_grid))
Base.show(io::IO, grid::PhysicalHeightGrid) = print(io, summary(grid))

@inline Operators.Δzᶜᶜᶜ(i, j, k, grid::PhysicalHeightGrid) = @inbounds grid.layer_depth[i, j, k]
@inline Operators.Δzᶜᶜᶠ(i, j, k, grid::PhysicalHeightGrid) = @inbounds grid.interface_spacing[i, j, k]
@inline Operators.Δzᶠᶜᶜ(i, j, k, grid::PhysicalHeightGrid) = ℑxᶠᵃᵃ(i, j, k, grid.underlying_grid, grid.layer_depth)
@inline Operators.Δzᶜᶠᶜ(i, j, k, grid::PhysicalHeightGrid) = ℑyᵃᶠᵃ(i, j, k, grid.underlying_grid, grid.layer_depth)
@inline Operators.Δzᶠᶜᶠ(i, j, k, grid::PhysicalHeightGrid) = ℑxᶠᵃᵃ(i, j, k, grid.underlying_grid, grid.interface_spacing)
@inline Operators.Δzᶜᶠᶠ(i, j, k, grid::PhysicalHeightGrid) = ℑyᵃᶠᵃ(i, j, k, grid.underlying_grid, grid.interface_spacing)

@inline Operators.Δz⁻¹ᶜᶜᶜ(i, j, k, grid::PhysicalHeightGrid) = 1 / Δzᶜᶜᶜ(i, j, k, grid)
@inline Operators.Δz⁻¹ᶜᶜᶠ(i, j, k, grid::PhysicalHeightGrid) = 1 / Operators.Δzᶜᶜᶠ(i, j, k, grid)
@inline Operators.Δz⁻¹ᶠᶜᶜ(i, j, k, grid::PhysicalHeightGrid) = 1 / Operators.Δzᶠᶜᶜ(i, j, k, grid)
@inline Operators.Δz⁻¹ᶜᶠᶜ(i, j, k, grid::PhysicalHeightGrid) = 1 / Operators.Δzᶜᶠᶜ(i, j, k, grid)
@inline Operators.Δz⁻¹ᶠᶜᶠ(i, j, k, grid::PhysicalHeightGrid) = 1 / Operators.Δzᶠᶜᶠ(i, j, k, grid)
@inline Operators.Δz⁻¹ᶜᶠᶠ(i, j, k, grid::PhysicalHeightGrid) = 1 / Operators.Δzᶜᶠᶠ(i, j, k, grid)

# As for a height coordinate (Oceananigans.TurbulenceClosures): the height of a face is not less
# than the depth of the layer below it, nor the height of a center less than half its layer.
@inline function OceanTurbulenceClosures.height_above_bottomᶜᶜᶠ(i, j, k, grid::PhysicalHeightGrid)
    h = @inbounds grid.interface_height[i, j, k]
    return max(Δzᶜᶜᶜ(i, j, k-1, grid), h)
end

@inline function OceanTurbulenceClosures.height_above_bottomᶜᶜᶜ(i, j, k, grid::PhysicalHeightGrid)
    h = @inbounds (grid.interface_height[i, j, k] + grid.interface_height[i, j, k+1]) / 2
    return max(Δzᶜᶜᶜ(i, j, k, grid) / 2, h)
end

# The grids on which the kernels of the closure differentiate in height: the wrapper above, built
# where a kernel is launched, or the physical height grid of a model that carries one
# (`HydrostaticDynamics(physical_height_grid=true)`), which is used as it is.
const GridOfHeights = Union{PhysicalHeightGrid, HydrostaticPressureGrid}

closure_height_grid(grid, heights, ::Nothing) = PhysicalHeightGrid(grid, heights)
closure_height_grid(grid, heights, coordinate::HydrostaticHeightCoordinate) = coordinate.grid

#####
##### Differences of logarithms between two layers
#####

# ln(a / b) for a and b of one sign. Where the two are close it is the series of 2 atanh(x) in
# x = (a - b) / (a + b), whose first operation is a difference that floating point takes exactly:
# the relative error of the result is then that of the arithmetic, where ln(a) - ln(b) loses the
# digits the two logarithms share (a relative error of about ε |ln a| / |ln(a / b)|, a thousand
# times the rounding ε for the potential temperatures of two adjacent layers). The series is
# truncated after x⁸ / 9, with |x| < 1/20: a relative truncation error below 1e-14.
@inline function log_ratio(a, b)
    x = (a - b) / (a + b)
    x² = x^2
    series = 2 * x * (1 + x² * (1//3 + x² * (1//5 + x² * (1//7 + x² * 1//9))))
    return ifelse(abs(x) < 1//20, series, log(a / b))
end

# The closure differentiates ln θᵨ (unsaturated branch) and ln θ of the dry potential temperature
# (saturated branch) between two layers. The same derivatives, of the same quantities, are taken
# here from fields that the hydrostatic core carries, without a difference of logarithms and
# without raising the pressure to a power again: θᵨ is `moisture.density_potential_temperature`,
# equal to (Rᵐ / Rᵈ) T / Π, and ln θ = ln T - ln Π with Π the Exner function of the layer.
@inline function Operators.∂zᶜᶜᶠ(i, j, k, grid::GridOfHeights, ::typeof(log_density_potential_temperature),
                                 buoyancy, T, qᵛᵉ)
    θᵨ = buoyancy.dynamics.moisture.density_potential_temperature
    return @inbounds log_ratio(θᵨ[i, j, k], θᵨ[i, j, k-1]) * Operators.Δz⁻¹ᶜᶜᶠ(i, j, k, grid)
end

@inline function Operators.∂zᶜᶜᶠ(i, j, k, grid::GridOfHeights, ::typeof(log_dry_potential_temperatureᶜᶜᶜ),
                                 buoyancy, T, qᵛᵉ)
    Π = buoyancy.dynamics.exner_function
    δ_log_θ = @inbounds log_ratio(T[i, j, k], T[i, j, k-1]) - log_ratio(Π[i, j, k], Π[i, j, k-1])
    return δ_log_θ * Operators.Δz⁻¹ᶜᶜᶠ(i, j, k, grid)
end

# Depth of the layers, height between them and height of the coordinate surfaces above the
# ground, from the hydrostatic integral with the density potential temperature. The spacings of
# the first and last face, through which nothing is exchanged, are half a layer.
@kernel function _compute_physical_heights!(Δhᶜ, Δhᶠ, hᶠ, grid, θᵨ, Π, Πᶠ, cᵖᵈ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        hᶠ[i, j, 1] = zero(grid)

        for k in 1:Nz
            Δhᶜ[i, j, k] = physical_layer_depth(i, j, k, θᵨ, Πᶠ, cᵖᵈ, g)
            hᶠ[i, j, k+1] = hᶠ[i, j, k] + Δhᶜ[i, j, k]
        end

        Δhᶠ[i, j, 1] = Δhᶜ[i, j, 1] / 2

        for k in 2:Nz
            Δhᶠ[i, j, k] = height_between_layers(i, j, k, grid, θᵨ, Π, Πᶠ, cᵖᵈ, g)
        end

        Δhᶠ[i, j, Nz+1] = Δhᶜ[i, j, Nz] / 2
    end
end

function launch_physical_heights!(heights, grid, dynamics, constants)
    launch!(grid.architecture, grid, :xy, _compute_physical_heights!,
            heights.layer_depth, heights.interface_spacing, heights.interface_height, grid,
            dynamics.moisture.density_potential_temperature, dynamics.exner_function,
            dynamics.interface_exner_function, constants.dry_air.heat_capacity, constants.gravitational_acceleration)
    return nothing
end

# The heights of a model with a physical height grid are those of its last `update_state!`
compute_physical_heights!(heights, model) = compute_physical_heights!(heights, model, model.dynamics.height_coordinate)
compute_physical_heights!(heights, model, ::HydrostaticHeightCoordinate) = nothing

function compute_physical_heights!(heights, model, ::Nothing)
    @apply_regionally launch_physical_heights!(heights, model.grid, model.dynamics, model.thermodynamic_constants)
    fill_halo_regions!(heights.layer_depth)
    fill_halo_regions!(heights.interface_spacing)
    fill_halo_regions!(heights.interface_height)
    return nothing
end

#####
##### From velocity points to cell centers
#####

# `from_x_faces` and `from_y_faces` of `HydrostaticEquations` bring a quantity per unit area at
# the velocity points to cell centers with the areas of the faces for weights.

#####
##### Surface fluxes: what a bulk formula hands to the atmosphere
#####

# Fluxes through the ground, positive upward (into the atmosphere), and the temperature at
# which the vapor leaves the surface. A coupler fills them; `BulkSurfaceExchange` does from the
# bulk boundary conditions of Breeze.
function surface_flux_fields(grid)
    return (x_stress = Field{Face, Center, Nothing}(grid),              # N m⁻², at u points
            y_stress = Field{Center, Face, Nothing}(grid),              # N m⁻², at v points
            sensible_heat_flux = Field{Center, Center, Nothing}(grid),  # W m⁻²
            vapor_flux = Field{Center, Center, Nothing}(grid),          # kg m⁻² s⁻¹
            surface_temperature = Field{Center, Center, Nothing}(grid)) # K
end

# Accumulated over the integration, per unit area
function turbulence_ledger(grid)
    return (surface_vapor = Field{Center, Center, Nothing}(grid),         # kg m⁻²: vapor through the ground
            surface_energy = Field{Center, Center, Nothing}(grid),        # J m⁻²: 𝒬 + Jᵛ (cᵖᵛ Tˢ + Φˢ) + kinetic
            surface_vapor_kinetic_energy = Field{Center, Center, Nothing}(grid), # J m⁻²: the kinetic part, Jᵛ Σ ΔB 𝒦
            surface_stress_work = Field{Center, Center, Nothing}(grid),   # J m⁻²: kinetic energy removed by the stress
            shear_production = Field{Center, Center, Nothing}(grid),      # J m⁻²: kinetic energy → TKE
            buoyancy_source = Field{Center, Center, Nothing}(grid),       # J m⁻²: enthalpy → TKE (B > 0)
            turbulent_kinetic_energy_sinks = Field{Center, Center, Nothing}(grid), # J m⁻²: TKE → enthalpy
            water_kinetic_energy = Field{Center, Center, Nothing}(grid))  # J m⁻²: enthalpy → TKE of the vapor of the ground
end

"""
$(TYPEDEF)

The existing bulk boundary conditions of Breeze (`BulkDrag`, `BulkSensibleHeatFlux`,
`BulkVaporFlux`), materialized for the ground of a hydrostatic model. Their formulas and their
coefficients are evaluated as they are, by `getbc`, on the state of the lowest layer;
[`MoistTurbulentMixing`](@ref) converts what they return to the stress, the sensible heat flux
and the vapor flux that it inserts in the lowest layer.
"""
struct BulkSurfaceExchange{U, V, H, Q}
    x_drag :: U
    y_drag :: V
    sensible_heat :: H
    vapor :: Q
end

Adapt.adapt_structure(to, exchange::BulkSurfaceExchange) =
    BulkSurfaceExchange(adapt(to, exchange.x_drag), adapt(to, exchange.y_drag),
                        adapt(to, exchange.sensible_heat), adapt(to, exchange.vapor))

@regional BulkSurfaceExchange

materialize_bulk_condition(::Nothing, location, model, closure) = nothing

function materialize_bulk_condition(bc, location, model, closure)
    coefficient = bc.condition.coefficient
    (coefficient isa Number || !isnothing(model.dynamics.height_coordinate)) ||
        throw(ArgumentError("BulkSurfaceExchange supports constant transfer coefficients only: the stability " *
                            "functions of $(summary(coefficient)) take heights from the grid, which on " *
                            "HydrostaticDynamics are labels of the coordinate unless it is built with " *
                            "physical_height_grid=true."))

    isnothing(bc.condition.surface_temperature) &&
        throw(ArgumentError("The bulk boundary conditions of BulkSurfaceExchange need a surface_temperature."))

    materialized = AtmosphereBoundaryConditions.materialize_atmosphere_boundary_condition(
        bc, Bottom(), location, height_grid_or(model.grid, model.dynamics.height_coordinate), model.dynamics, model.microphysics, model.thermodynamic_constants, closure)

    formulation = AtmosphereBoundaryConditions.sensible_heat_flux_formulation(Val(:ρθ), model.formulation)
    condition = AtmosphereBoundaryConditions.set_sensible_heat_formulation(materialized, formulation).condition
    return with_float_type(eltype(model.grid), condition)
end

# The bulk conditions keep the numbers they are given: a surface temperature written `300` stays
# an integer and a coefficient written `1.2e-3` a `Float64`, and either promotes the flux of a
# `Float32` model to `Float64` inside the kernel, which a GPU without double precision cannot
# compile. Every number of a materialized condition is converted to the float type of the grid.
@inline in_float_type(FT, x::Number) = convert(FT, x)
@inline in_float_type(FT, x) = x

with_float_type(FT, df::AtmosphereBoundaryConditions.BulkDragFunction) =
    AtmosphereBoundaryConditions.BulkDragFunction(df.direction, df.side, in_float_type(FT, df.coefficient),
                                                  in_float_type(FT, df.gustiness), in_float_type(FT, df.surface_temperature),
                                                  df.filtered_velocities, df.thermodynamic_constants)

with_float_type(FT, bf::AtmosphereBoundaryConditions.BulkSensibleHeatFluxFunction) =
    AtmosphereBoundaryConditions.BulkSensibleHeatFluxFunction(bf.side, in_float_type(FT, bf.coefficient),
                                                              in_float_type(FT, bf.gustiness),
                                                              in_float_type(FT, bf.surface_temperature),
                                                              in_float_type(FT, bf.standard_pressure),
                                                              bf.thermodynamic_constants, bf.formulation,
                                                              bf.filtered_velocities, bf.filtered_scalar, bf.microphysics)

with_float_type(FT, bf::AtmosphereBoundaryConditions.BulkVaporFluxFunction) =
    AtmosphereBoundaryConditions.BulkVaporFluxFunction(bf.side, in_float_type(FT, bf.coefficient),
                                                       in_float_type(FT, bf.gustiness),
                                                       in_float_type(FT, bf.surface_temperature),
                                                       in_float_type(FT, bf.surface_relative_humidity),
                                                       bf.thermodynamic_constants, bf.surface,
                                                       in_float_type(FT, bf.moisture_availability),
                                                       bf.filtered_velocities, bf.filtered_scalar)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Materialize bulk boundary conditions of Breeze for the ground of `model`, to be evaluated by
[`MoistTurbulentMixing`](@ref) with `closure` (which a default gustiness takes the turbulent
kinetic energy from). Each of `drag`, `sensible_heat` and `vapor` is a boundary condition built
by `BulkDrag`, `BulkSensibleHeatFlux` or `BulkVaporFlux` with a constant coefficient and a
surface temperature, or `nothing`.
"""
function BulkSurfaceExchange(model, closure; drag=nothing, sensible_heat=nothing, vapor=nothing)
    return BulkSurfaceExchange(materialize_bulk_condition(drag, (Face(), Center(), Center()), model, closure),
                               materialize_bulk_condition(drag, (Center(), Face(), Center()), model, closure),
                               materialize_bulk_condition(sensible_heat, (Center(), Center(), Center()), model, closure),
                               materialize_bulk_condition(vapor, (Center(), Center(), Center()), model, closure))
end

@inline bulk_flux(::Nothing, i, j, grid, clock, fields, dynamics_fields) = zero(grid)
@inline bulk_flux(condition, i, j, grid, clock, fields, dynamics_fields) =
    getbc(condition, i, j, grid, clock, fields, dynamics_fields)

@inline bulk_surface_temperature(::Nothing, i, j, grid, clock, T) = @inbounds T[i, j, 1]
@inline bulk_surface_temperature(condition, i, j, grid, clock, T) =
    AtmosphereBoundaryConditions.wall_value(i, j, grid, Bottom(), condition.surface_temperature, clock)

# The bulk conditions return fluxes of velocity, of potential temperature and of vapor times
# a density in kg m⁻³: a stress in N m⁻², a flux J of θ per unit mass of air, and a vapor
# flux in kg m⁻² s⁻¹. The flux of θ enters the lowest layer at fixed composition and pressure,
# where the enthalpy per unit mass is ĉᵖ Π θ with ĉᵖ = qᵈ cᵖᵈ + qᵗ cᵖᵛ: it is ĉᵖ Π J in W m⁻².
@kernel function _compute_bulk_surface_fluxes!(fluxes, grid, exchange, clock, fields, dynamics_fields, ρ, ρqˣ, Π, T, cᵖᵈ, cᵖᵛ)
    i, j = @index(Global, NTuple)

    @inbounds begin
        fluxes.x_stress[i, j, 1] = bulk_flux(exchange.x_drag, i, j, grid, clock, fields, dynamics_fields)
        fluxes.y_stress[i, j, 1] = bulk_flux(exchange.y_drag, i, j, grid, clock, fields, dynamics_fields)

        J = bulk_flux(exchange.sensible_heat, i, j, grid, clock, fields, dynamics_fields)
        ĉᵖ = cᵖᵈ + sum_at(i, j, 1, ρqˣ) / ρ[i, j, 1] * (cᵖᵛ - cᵖᵈ)
        fluxes.sensible_heat_flux[i, j, 1] = ĉᵖ * Π[i, j, 1] * J

        fluxes.vapor_flux[i, j, 1] = bulk_flux(exchange.vapor, i, j, grid, clock, fields, dynamics_fields)
        fluxes.surface_temperature[i, j, 1] = bulk_surface_temperature(exchange.vapor, i, j, grid, clock, T)
    end
end

compute_bulk_surface_fluxes!(fluxes, ::Nothing, model) = nothing

function launch_bulk_surface_fluxes!(fluxes, exchange, model)
    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants

    launch!(grid.architecture, grid, :xy, _compute_bulk_surface_fluxes!,
            fluxes, height_grid_or(grid, dynamics.height_coordinate), exchange, model.clock, Oceananigans.fields(model), dynamics_thermodynamic_fields(dynamics),
            dynamics.density, water_densities(model), dynamics.exner_function, model.temperature,
            constants.dry_air.heat_capacity, constants.vapor.heat_capacity)

    return nothing
end

function compute_bulk_surface_fluxes!(fluxes, exchange::BulkSurfaceExchange, model)
    @apply_regionally launch_bulk_surface_fluxes!(fluxes, exchange, model)
    return nothing
end

#####
##### The mixing
#####

struct MoistTurbulentMixing{C, CF, G, F, X, S, L}
    closure :: C           # the `TKEBasedTurbulenceClosure`, as it is
    closure_fields :: CF   # its fields: Kᵘ, Kᶜ, Kᵉ, N², ℓ, Lᵉ, in m² s⁻¹, s⁻², m and s⁻¹
    heights :: G           # the heights of the column, for the `PhysicalHeightGrid` of each launch
    surface_fluxes :: F    # stress, sensible heat flux, vapor flux and surface temperature
    surface_exchange :: X  # `BulkSurfaceExchange` that fills them before every step, or nothing
    scratch :: S
    ledger :: L            # accumulated exchanges of water and energy (two-dimensional)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the vertical turbulent mixing of `model`, an `AtmosphereModel` with moist
`HydrostaticDynamics` and the tracer `ρe`, by `closure`, a
[`TKEBasedTurbulenceClosure`](@ref Breeze.TurbulenceClosures.TKEBasedTurbulenceClosure) with a
vertically implicit time discretization. It is applied by [`moist_turbulent_mixing_step!`](@ref),
or as a callback, `Callback(mixing, IterationInterval(1))`, once per time step.

The closure is not a `closure` of the model: its diffusivities are computed here, in the heights
of the hydrostatic column, and the fluxes enter the budgets of the total-mass coordinate. The
tracer `ρe` is the turbulent kinetic energy per unit area and unit of the grid's coordinate,
`e` times the total density, which the model advects.

Keyword arguments
=================

- `closure`: the closure. Default `TKEBasedTurbulenceClosure(eltype(model.grid))`.
- `surface_exchange`: `nothing` (default), for which the fields of `mixing.surface_fluxes` are
  used as they are (zero unless set), or a [`BulkSurfaceExchange`](@ref) that fills them from
  the state before every step.

On a multi-region grid the kernels are launched in each region, the halos of the velocities and
of the stress are filled as pairs, and conductances that the interpolation of the halos of a
`YinYangGrid` makes negative are clipped there, as the closure clips its diffusivities. The
budgets of water, enthalpy and turbulent kinetic energy are those of each column on any grid.
The kinetic energy that the mixing removes at velocity points reaches the cell centers with the
areas of the faces for weights, which closes the energy of a single grid to rounding; across
the seams of a `YinYangGrid`, where velocity points are interpolated, it does not.

A prescribed stress must vanish on walls, where the velocity normal to the wall is zero.
"""
function MoistTurbulentMixing(model::MoistHydrostaticModel;
                              closure = TKEBasedTurbulenceClosure(eltype(model.grid)),
                              surface_exchange = nothing)
    grid = model.grid

    size(grid, 3) ≥ 2 || throw(ArgumentError("MoistTurbulentMixing requires at least two layers."))

    closure isa TKEBasedTurbulenceClosure ||
        throw(ArgumentError("MoistTurbulentMixing adapts TKEBasedTurbulenceClosure, got $(summary(closure))."))

    time_discretization(closure) isa VerticallyImplicitTimeDiscretization ||
        throw(ArgumentError("MoistTurbulentMixing requires a TKEBasedTurbulenceClosure with a " *
                            "VerticallyImplicitTimeDiscretization: the sinks of the turbulent kinetic energy " *
                            "are solved with its diffusion."))

    haskey(model.tracers, TKE_NAME) ||
        throw(ArgumentError("MoistTurbulentMixing requires the tracer :$TKE_NAME, the density of turbulent " *
                            "kinetic energy: build the model with tracers = :$TKE_NAME."))

    closure_fields = build_closure_fields(grid, model.clock, tuple(TKE_NAME), NamedTuple(), closure)

    scratch = (specific_turbulent_kinetic_energy = CenterField(grid),
               eliminated_diagonal = CenterField(grid),
               implicit_mass = CenterField(grid),
               mixing_ratio = CenterField(grid),
               total_density = CenterField(grid),
               initial_water_density = CenterField(grid),
               kinetic_energy = CenterField(grid),
               heating = CenterField(grid),
               potential_temperature = CenterField(grid),
               x_velocity = XFaceField(grid),
               y_velocity = YFaceField(grid),
               vapor_exchange = Field{Center, Center, Nothing}(grid),
               conductances = (scalar = ZFaceField(grid), momentum = ZFaceField(grid), energy = ZFaceField(grid)))

    return MoistTurbulentMixing(closure, closure_fields, height_fields(grid), surface_flux_fields(grid),
                                surface_exchange, scratch, turbulence_ledger(grid))
end

@regional MoistTurbulentMixing

# TODO: the step ends with `update_state!(model; compute_tendencies=true)`, whose tendency kernels
# use the advection schemes of the model. A model on a `Float32` grid built without explicit schemes
# gets the defaults in `Oceananigans.defaults.FloatType` (`Centered()` is `Float64`), which promote
# those kernels to `Float64` and cannot be compiled on a GPU without double precision (Metal). This
# belongs in Oceananigans (or in the `AtmosphereModel` constructor): convert default advection
# schemes to the float type of the grid. Until then build such a model with schemes in that type,
# as in `scalar_advection = WENO(Float32; order=5)`.

Base.summary(mixing::MoistTurbulentMixing) = string("MoistTurbulentMixing(", summary(mixing.closure), ")")
Base.show(io::IO, mixing::MoistTurbulentMixing) = print(io, summary(mixing))

# The ledger accumulates over the integration, so a restart carries it on
Oceananigans.prognostic_state(mixing::MoistTurbulentMixing) = Oceananigans.prognostic_state(mixing.ledger)

function Oceananigans.restore_prognostic_state!(mixing::MoistTurbulentMixing, from::NamedTuple)
    Oceananigans.restore_prognostic_state!(mixing.ledger, from)
    return mixing
end

Oceananigans.restore_prognostic_state!(mixing::MoistTurbulentMixing, ::Nothing) = mixing

(mixing::MoistTurbulentMixing)(simulation) =
    moist_turbulent_mixing_step!(mixing, simulation.model, simulation.model.clock.last_Δt)


#####
##### The closure fields, in the heights of the column
#####


# e = ρe / ρᵗ, and the kinetic energy of the layer per unit mass at cell centers, with the
# kinetic energy of the velocity points brought there by the areas of their faces
@kernel function _compute_specific_energies!(e, 𝒦, grid, ρe, ρᵗ, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        e[i, j, k] = ρe[i, j, k] / ρᵗ[i, j, k]
        𝒦[i, j, k] = specific_kinetic_energy(i, j, k, grid, u, v)
    end
end

function launch_specific_energies!(mixing, model)
    grid = model.grid
    u, v, _ = model.velocities
    launch!(grid.architecture, grid, :xyz, _compute_specific_energies!,
            mixing.scratch.specific_turbulent_kinetic_energy, mixing.scratch.kinetic_energy, grid,
            model.tracers[TKE_NAME], model.dynamics.density, u, v)
    return nothing
end

# The kernels of the closure, launched as they are with the heights of the column for a grid.
# The static stability is computed on the faces between two layers; it stays zero on the ground
# and on the model top, where the closure masks its diffusivities. The conductances follow.
function launch_closure_fields_in_height!(mixing, model)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    moisture = dynamics.moisture
    constants = model.thermodynamic_constants
    physical_grid = closure_height_grid(grid, mixing.heights, dynamics.height_coordinate)
    closure = mixing.closure
    closure_fields = mixing.closure_fields
    C = mixing.scratch.conductances
    e = mixing.scratch.specific_turbulent_kinetic_energy
    Nx, Ny, Nz = size(grid)

    buoyancy = AtmosphereModelBuoyancy(dynamics, model.formulation, constants, model.microphysics, model.microphysical_fields)
    tracers = NamedTuple{(TKE_NAME, :T, :qᵛᵉ)}((e, model.temperature, specific_prognostic_moisture(model)))

    launch!(arch, grid, KernelParameters(1:Nx, 1:Ny, 2:Nz), _compute_tke_static_stability!,
            closure_fields.N², physical_grid, closure, tracers, buoyancy)

    launch!(arch, grid, :xy, _compute_mixing_length!,
            closure_fields.ℓᵐ, closure_fields.ℓˢ, physical_grid, closure, e, closure_fields.N²)

    launch!(arch, grid, :xyz, _compute_tke_closure_fields!,
            closure_fields, physical_grid, closure, model.velocities, tracers)

    launch!(arch, grid, :xyz, _compute_tke_implicit_linear_coefficient!,
            closure_fields.Lᵉ, physical_grid, closure, closure_fields, model.velocities, tracers)

    launch!(arch, grid, :xy, _compute_moist_conductances!,
            C.scalar, C.momentum, C.energy, grid, closure_fields.Kᶜ, closure_fields.Kᵘ, closure_fields.Kᵉ,
            moisture.density_potential_temperature, dynamics.exner_function, dynamics.interface_exner_function,
            moisture.interface_pressure,
            dry_air_gas_constant(constants), constants.dry_air.heat_capacity, constants.gravitational_acceleration)

    return nothing
end

#####
##### Conductances
#####

# C = ρᵃ K / Δh on the z-faces between two layers, in kg m⁻² s⁻¹, with ρᵃ = p / (Rᵈ Tᵨ) the
# density of the air on the face at its total pressure and density temperature Tᵨ = Πᶠ θᵨ, and
# Δh the height between the layers. Momentum, turbulent kinetic energy and the scalars, which are
# per unit total mass, are all carried by the air and its water, the mass of the coordinate.
# Without water these are the conductances of the dry `VerticalDiffusion`.
@kernel function _compute_moist_conductances!(Cᶜ, Cᵘ, Cᵉ, grid, Kᶜ, Kᵘ, Kᵉ, θᵨ, Π, Πᶠ, pᶠ, Rᵈ, cᵖᵈ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        Cᶜ[i, j, 1] = zero(grid)
        Cᵘ[i, j, 1] = zero(grid)
        Cᵉ[i, j, 1] = zero(grid)

        for k in 2:Nz
            Tᵨ = Πᶠ[i, j, k] * interface_potential_temperature(i, j, k, grid, θᵨ, Π, Πᶠ)
            ρᵃ = pᶠ[i, j, k] / (Rᵈ * Tᵨ)
            Δh = height_between_layers(i, j, k, grid, θᵨ, Π, Πᶠ, cᵖᵈ, g)

            Cᶜ[i, j, k] = ρᵃ * Kᶜ[i, j, k] / Δh
            Cᵘ[i, j, k] = ρᵃ * Kᵘ[i, j, k] / Δh
            Cᵉ[i, j, k] = ρᵃ * Kᵉ[i, j, k] / Δh
        end

        Cᶜ[i, j, Nz+1] = zero(grid)
        Cᵘ[i, j, Nz+1] = zero(grid)
        Cᵉ[i, j, Nz+1] = zero(grid)
    end
end

#####
##### Water that does not fall
#####

# One backward Euler step of the mass fraction r = ρqᵉ / ρ of the water that does not fall, per
# unit total mass, with the total mass of the layers for weights. The vapor flux through the
# ground enters the lowest layer; a flux out of the air (dew) is limited to the water of that
# layer, and the ledger holds what was exchanged. `ρᵗ⁺` is the density of the mass the layers
# then hold, which differs from that of the coordinate by the vapor of the lowest layer until the
# surface pressure takes it up (`launch_energy_mixing!`), and `ρqᵗ` the density of all water
# before the step.
@kernel function _mix_water!(ρqᵉ, r, γ, ρᵗ⁺, ρqᵗ, exchanged, ledger, grid, ρ, C, Jᵛ, ρqˣ, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz
            r[i, j, k] = ρqᵉ[i, j, k] / ρ[i, j, k]
            ρqᵗ[i, j, k] = sum_at(i, j, k, ρqˣ)
        end

        μ = layer_mass_per_area(i, j, 1, grid, ρ)
        vapor = max(Δt * Jᵛ[i, j, 1], - max(0, μ * r[i, j, 1]))
        r[i, j, 1] += vapor / μ
        exchanged[i, j, 1] = vapor
        ledger.surface_vapor[i, j, 1] += vapor
    end

    diffuse_column!(r, γ, i, j, grid, Δt, ρ, C, layer_mass_per_area, face_conductance)

    @inbounds for k in 1:Nz
        ρqᵉ[i, j, k] = ρ[i, j, k] * r[i, j, k]
        ρᵗ⁺[i, j, k] = ρ[i, j, k]
    end

    @inbounds ρᵗ⁺[i, j, 1] = ρ[i, j, 1] * (1 + exchanged[i, j, 1] / layer_mass_per_area(i, j, 1, grid, ρ))
end

function launch_water_mixing!(mixing, model, Δt)
    grid = model.grid
    scratch = mixing.scratch
    launch!(grid.architecture, grid, :xy, _mix_water!,
            model.moisture_density, scratch.mixing_ratio, scratch.eliminated_diagonal, scratch.total_density,
            scratch.initial_water_density, scratch.vapor_exchange, mixing.ledger, grid, model.dynamics.density, scratch.conductances.scalar,
            mixing.surface_fluxes.vapor_flux, water_densities(model), Δt)
    return nothing
end

#####
##### Velocities
#####

# The stress of the ground accelerates the lowest layer, of total mass μ per unit area at the
# velocity point; `velocity★` is the column of velocities that the diffusion starts from.
@kernel function _apply_surface_stress!(velocity, velocity★, grid, ρᵗ, τ, Δt, interpolate)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        μ = interpolate(i, j, 1, grid, layer_mass_per_area, ρᵗ)
        velocity[i, j, 1] += Δt * τ[i, j, 1] / μ

        for k in 1:Nz
            velocity★[i, j, k] = velocity[i, j, k]
        end
    end
end

# The kinetic energy that the stress removed from the lowest layer, μ (uⁿ² - u★²) / 2 per unit
# area at the velocity point, with uⁿ = u★ - Δt τ / μ the velocity before it
@inline function stress_work(i, j, k, grid, u★, τ, ρᵗ, Δt, interpolate)
    μ = interpolate(i, j, 1, grid, layer_mass_per_area, ρᵗ)
    u = @inbounds u★[i, j, 1]
    uⁿ = @inbounds u - Δt * τ[i, j, 1] / μ
    return μ * (uⁿ^2 - u^2) / 2
end

@inline x_stress_work(i, j, k, grid, u★, τ, ρᵗ, Δt) = stress_work(i, j, k, grid, u★, τ, ρᵗ, Δt, ℑxᶠᵃᵃ)
@inline y_stress_work(i, j, k, grid, v★, τ, ρᵗ, Δt) = stress_work(i, j, k, grid, v★, τ, ρᵗ, Δt, ℑyᵃᶠᵃ)

# The kinetic energy that the backward Euler step from u★ to u⁺ removes from the layer `k`, per
# unit area at the velocity point:
#
#   d[k] = Δt (C⁻ δu⁻² + C⁺ δu⁺²) / 2 + μ (u⁺ - u★)² / 2 ≥ 0 ,
#
# half the dissipation C δu² of each of its two faces, at the new velocities, and the
# dissipation of the time discretization. The sum over the column is the kinetic energy lost,
# Σ μ (u★² - u⁺²) / 2, because the conductance vanishes on the first and the last face.
@inline function velocity_dissipation(i, j, k, grid, u⁺, u★, ρᵗ, C, Δt, interpolate)
    μ = interpolate(i, j, k, grid, layer_mass_per_area, ρᵗ)
    C⁻ = interpolate(i, j, k, grid, C)
    C⁺ = interpolate(i, j, k+1, grid, C)

    @inbounds begin
        δu⁻ = u⁺[i, j, k] - u⁺[i, j, k-1]
        δu⁺ = u⁺[i, j, k+1] - u⁺[i, j, k]
        return Δt * (C⁻ * δu⁻^2 + C⁺ * δu⁺^2) / 2 + μ * (u⁺[i, j, k] - u★[i, j, k])^2 / 2
    end
end

@inline x_velocity_dissipation(i, j, k, grid, u⁺, u★, ρᵗ, C, Δt) = velocity_dissipation(i, j, k, grid, u⁺, u★, ρᵗ, C, Δt, ℑxᶠᵃᵃ)
@inline y_velocity_dissipation(i, j, k, grid, v⁺, v★, ρᵗ, C, Δt) = velocity_dissipation(i, j, k, grid, v⁺, v★, ρᵗ, C, Δt, ℑyᵃᶠᵃ)

function launch_surface_stress!(mixing, model, Δt)
    grid = model.grid
    arch = grid.architecture
    scratch = mixing.scratch
    fluxes = mixing.surface_fluxes
    u, v, _ = model.velocities
    ρᵗ⁺ = scratch.total_density

    launch!(arch, grid, :xy, _apply_surface_stress!, u, scratch.x_velocity, grid, ρᵗ⁺, fluxes.x_stress, Δt, ℑxᶠᵃᵃ)
    launch!(arch, grid, :xy, _apply_surface_stress!, v, scratch.y_velocity, grid, ρᵗ⁺, fluxes.y_stress, Δt, ℑyᵃᶠᵃ)

    return nothing
end

function launch_velocity_mixing!(mixing, model, Δt)
    grid = model.grid
    arch = grid.architecture
    scratch = mixing.scratch
    u, v, _ = model.velocities
    ρᵗ⁺ = scratch.total_density
    C = scratch.conductances.momentum
    γ = scratch.eliminated_diagonal

    launch!(arch, grid, :xy, _diffuse_velocity!, u, γ, grid, ρᵗ⁺, C, Δt, ℑxᶠᵃᵃ)
    launch!(arch, grid, :xy, _diffuse_velocity!, v, γ, grid, ρᵗ⁺, C, Δt, ℑyᵃᶠᵃ)

    return nothing
end

#####
##### Turbulent kinetic energy
#####

@inline stored_mass(i, j, k, grid, μ) = @inbounds μ[i, j, k]

# One backward Euler step of e, per unit total mass:
#
#   μ (e⁺ - eⁿ) = Δt [Cᵉ⁺ (e⁺[k+1] - e⁺[k]) - Cᵉ⁻ (e⁺[k] - e⁺[k-1])] + 𝒫 + Δt μ (B⁺ + Lᵉ e⁺) ,
#
# with μ the total mass of the layer per unit area after the water has been mixed, 𝒫 the kinetic
# energy that the mixing of the velocities removed from the layer, B⁺ the positive part of the
# buoyancy flux - Kᶜ N² of the closure and Lᵉ ≤ 0 its sink rate (dissipation, the negative part
# of the buoyancy flux, the damping of negative e): the sources explicit and the sinks implicit,
# as the closure has them. The sinks are a linear term of the tridiagonal system, solved by
# `diffuse_column!` with the mass μ′ = μ (1 - Δt Lᵉ).
#
# `heating` is what the enthalpy of the layer receives, in J m⁻²: minus the energy that the
# turbulent kinetic energy takes from buoyancy, plus what its sinks return, minus the turbulent
# kinetic energy that the lowest layer gains with the vapor of the ground, which `ρe` keeps through
# the change of the surface pressure. The kinetic energy that the vapor brings is not taken from
# the enthalpy: it crosses the ground with the vapor and is booked with the surface fluxes
# (`_express_enthalpy_as_potential_temperature!`).
@kernel function _step_turbulent_kinetic_energy!(ρe, e, μ′, γ, heating, ledger, grid, ρᵗ⁺, ρᵗ, Cᵉ, Cᵘ, Lᵉ, Kᶜ, N²,
                                                 u, v, u★, v★, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    production = zero(grid)
    buoyancy_source = zero(grid)
    sinks = zero(grid)
    water_energy = zero(grid)

    @inbounds for k in 1:Nz
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        μ = ρᵗ⁺[i, j, k] * Δz
        δμ = (ρᵗ⁺[i, j, k] - ρᵗ[i, j, k]) * Δz
        eⁿ = e[i, j, k]

        𝒫 = (from_x_faces(i, j, k, grid, x_velocity_dissipation, u, u★, ρᵗ⁺, Cᵘ, Δt) +
             from_y_faces(i, j, k, grid, y_velocity_dissipation, v, v★, ρᵗ⁺, Cᵘ, Δt))
        B = ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, Kᶜ, N²)
        ℬ = Δt * μ * max(0, B)

        μ′[i, j, k] = μ * (1 - Δt * Lᵉ[i, j, k])
        e[i, j, k] = (μ * eⁿ + 𝒫 + ℬ) / μ′[i, j, k]
        heating[i, j, k] = - ℬ - δμ * eⁿ

        production += 𝒫
        buoyancy_source += ℬ
        water_energy += δμ * eⁿ
    end

    diffuse_column!(e, γ, i, j, grid, Δt, μ′, Cᵉ, stored_mass, face_conductance)

    @inbounds for k in 1:Nz
        μ = ρᵗ⁺[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
        sink = - Δt * μ * Lᵉ[i, j, k] * e[i, j, k]
        heating[i, j, k] += sink
        sinks += sink
        ρe[i, j, k] = ρᵗ⁺[i, j, k] * e[i, j, k]
    end

    @inbounds begin
        ledger.shear_production[i, j, 1] += production
        ledger.buoyancy_source[i, j, 1] += buoyancy_source
        ledger.turbulent_kinetic_energy_sinks[i, j, 1] += sinks
        ledger.water_kinetic_energy[i, j, 1] += water_energy
    end
end

#####
##### Enthalpy
#####

# Enthalpy of the air of a layer per unit total mass over its potential temperature, ĉᵖ Π with
# ĉᵖ = cᵖᵈ + qᵗ (cᵖᵛ - cᵖᵈ), from the density `ρqᵗ` of all its water: the enthalpy of the layer is
# Δp ĉᵖ Π θ / g.
@inline enthalpy_factor(i, j, k, ρ, ρqᵗ, Π, cᵖᵈ, cᵖᵛ) =
    @inbounds (cᵖᵈ + ρqᵗ[i, j, k] / ρ[i, j, k] * (cᵖᵛ - cᵖᵈ)) * Π[i, j, k]

# `ρqᵖ = (rain = ρqʳ, snow = ρqˢ)`: the densities of the water that falls, liquid and ice (a
# `ZeroField` for a phase that the scheme does not have)

# Mass of the air and of the water that does not fall, per unit total mass: 1 - qʳ - qˢ
@inline nonprecipitating_loading(i, j, k, ρ, ρqᵖ) = @inbounds 1 - (ρqᵖ.rain[i, j, k] + ρqᵖ.snow[i, j, k]) / ρ[i, j, k]

# Enthalpy of the falling water of a layer per unit total mass, qʳ (cˡ T - ℒˡ₀) + qˢ (cⁱ T - ℒⁱ₀),
# which the turbulence does not carry: each phase with its own heat capacity and latent energy
@inline function precipitation_enthalpy(i, j, k, ρ, ρqᵖ, T, constants)
    cˡ = constants.liquid.heat_capacity
    cⁱ = constants.ice.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    return @inbounds (ρqᵖ.rain[i, j, k] * (cˡ * T[i, j, k] - ℒˡ₀) + ρqᵖ.snow[i, j, k] * (cⁱ * T[i, j, k] - ℒⁱ₀)) / ρ[i, j, k]
end

# The flux of enthalpy through the z-face `k`, between the layers `k - 1` and `k`, is down the
# gradient of the static energy of the air and of the water that does not fall, per unit total mass:
#
#   F = - Cᶜ [χ[k] - χ[k-1] + (1 - q̄ᵖ) g Δh - (χʳ[k] - χʳ[k-1])] ,    χ = ĉᵖ Π θ ,
#
# with g Δh = cᵖᵈ [θᵨ[k-1] (Π[k-1] - Πᶠ) + θᵨ[k] (Πᶠ - Π[k])] the geopotential between the two
# layers and χʳ the enthalpy of the falling water (`precipitation_enthalpy`). With θᵨ = ϑ θ it is linear in θ,
#
#   F = - Cᶜ (a⁺ θ[k] - a⁻ θ[k-1] + R) ,
#   a⁺ = ĉᵖ Π[k] + (1 - q̄ᵖ) cᵖᵈ ϑ[k] (Πᶠ - Π[k]) ,    a⁻ = ĉᵖ Π[k-1] - (1 - q̄ᵖ) cᵖᵈ ϑ[k-1] (Π[k-1] - Πᶠ) ,
#
# and without water a⁺ = a⁻ = cᵖᵈ Πᶠ: the flux of the dry `VerticalDiffusion` times cᵖᵈ Πᶠ.
@inline function upper_enthalpy_coefficient(i, j, k, ρ, ρqᵗ, ρqᵖ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
    loading = (nonprecipitating_loading(i, j, k, ρ, ρqᵖ) + nonprecipitating_loading(i, j, k-1, ρ, ρqᵖ)) / 2
    ϑ = @inbounds θᵨ[i, j, k] / θ[i, j, k]
    return @inbounds enthalpy_factor(i, j, k, ρ, ρqᵗ, Π, cᵖᵈ, cᵖᵛ) + loading * cᵖᵈ * ϑ * (Πᶠ[i, j, k] - Π[i, j, k])
end

@inline function lower_enthalpy_coefficient(i, j, k, ρ, ρqᵗ, ρqᵖ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
    loading = (nonprecipitating_loading(i, j, k, ρ, ρqᵖ) + nonprecipitating_loading(i, j, k-1, ρ, ρqᵖ)) / 2
    ϑ = @inbounds θᵨ[i, j, k-1] / θ[i, j, k-1]
    return @inbounds enthalpy_factor(i, j, k-1, ρ, ρqᵗ, Π, cᵖᵈ, cᵖᵛ) - loading * cᵖᵈ * ϑ * (Π[i, j, k-1] - Πᶠ[i, j, k])
end

@inline precipitation_enthalpy_difference(i, j, k, ρ, ρqᵖ, T, constants) =
    precipitation_enthalpy(i, j, k-1, ρ, ρqᵖ, T, constants) - precipitation_enthalpy(i, j, k, ρ, ρqᵖ, T, constants)

# One backward Euler step of the enthalpy of the layers, Δp ĉᵖ Π θ / g, with the composition and
# the pressure of the start of the step in the coefficients, by the Thomas algorithm for the
# unknown θ̃. Row `k`, multiplied by g:
#
#   Δp ĉᵖ Π (θ̃[k] - θ[k]) = g Δt {Cᶜ⁺ (a⁺⁺ θ̃[k+1] - a⁻⁺ θ̃[k] + R⁺) - Cᶜ⁻ (a⁺⁻ θ̃[k] - a⁻⁻ θ̃[k-1] + R⁻)} + g S ,
#
# with S the sources in J m⁻²: the fluxes of the ground into the lowest layer, the heating of
# `_step_turbulent_kinetic_energy!` and the work of the surface stress. Every column of the
# matrix sums to Δp ĉᵖ Π > 0 and its off-diagonal entries are not positive.
#
# The enthalpy Δp ĉᵖ Π θ̃ / g of each layer is then expressed as the potential temperature of the
# layer with the water it now holds, at the pressure of the surface that has taken up the vapor
# of the ground (`change_column_surface_pressure!`): the column loses or gains enthalpy by its
# sources only.
#
# The step is three kernels, each with few arrays: a device may limit the number of arrays of a
# kernel (31 on Metal), and the grid brings its own.
@kernel function _add_surface_energy_sources!(heating, surface_energy, surface_stress_work, grid, ρᵗ⁺, Φˢ,
                                              𝒬, Jᵛ_applied, Tˢ, u★, v★, τˣ, τʸ, cᵖᵛ, Δt)
    i, j = @index(Global, NTuple)

    @inbounds begin
        # Sources of the lowest layer, in J m⁻²
        vapor = Jᵛ_applied[i, j, 1]
        work = (from_x_faces(i, j, 1, grid, x_stress_work, u★, τˣ, ρᵗ⁺, Δt) +
                from_y_faces(i, j, 1, grid, y_stress_work, v★, τʸ, ρᵗ⁺, Δt))
        surface = Δt * 𝒬[i, j, 1] + vapor * cᵖᵛ * Tˢ[i, j, 1]

        surface_energy[i, j, 1] += surface + vapor * Φˢ[i, j, 1]
        surface_stress_work[i, j, 1] += work
        heating[i, j, 1] = heating[i, j, 1] + surface + work
    end
end

@kernel function _solve_enthalpy!(θ̃, γ, grid, ρ, ρqᵗ, ρqᵖ, A, B, pˢ, θ, θᵨ, Π, Πᶠ, T, C, heating, constants, g, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity

    @inbounds begin
        # Forward elimination
        C⁺ = C[i, j, 2]
        a⁺ = upper_enthalpy_coefficient(i, j, 2, ρ, ρqᵗ, ρqᵖ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
        a⁻ = lower_enthalpy_coefficient(i, j, 2, ρ, ρqᵗ, ρqᵖ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
        R⁺ = precipitation_enthalpy_difference(i, j, 2, ρ, ρqᵖ, T, constants)

        Δp = layer_pressure_thickness(i, j, 1, A, B, pˢ)
        M = Δp * enthalpy_factor(i, j, 1, ρ, ρqᵗ, Π, cᵖᵈ, cᵖᵛ)
        S = heating[i, j, 1]

        β = M + g * Δt * C⁺ * a⁻
        upper = - g * Δt * C⁺ * a⁺
        θ̃[i, j, 1] = (M * θ[i, j, 1] + g * S + g * Δt * C⁺ * R⁺) / β

        for k in 2:Nz
            C⁻ = C⁺
            a⁺⁻ = a⁺
            a⁻⁻ = a⁻
            R⁻ = R⁺

            last = k == Nz
            m = ifelse(last, k, k + 1)
            C⁺ = C[i, j, k+1]
            a⁺ = upper_enthalpy_coefficient(i, j, m, ρ, ρqᵗ, ρqᵖ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
            a⁻ = lower_enthalpy_coefficient(i, j, m, ρ, ρqᵗ, ρqᵖ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
            R⁺ = precipitation_enthalpy_difference(i, j, m, ρ, ρqᵖ, T, constants)

            Δp = layer_pressure_thickness(i, j, k, A, B, pˢ)
            M = Δp * enthalpy_factor(i, j, k, ρ, ρqᵗ, Π, cᵖᵈ, cᵖᵛ)
            lower = - g * Δt * C⁻ * a⁻⁻
            diagonal = M + g * Δt * (C⁺ * a⁻ + C⁻ * a⁺⁻)
            right = M * θ[i, j, k] + g * heating[i, j, k] + g * Δt * (C⁺ * R⁺ - C⁻ * R⁻)

            γ[i, j, k] = upper / β
            β = diagonal - lower * γ[i, j, k]
            θ̃[i, j, k] = (right - lower * θ̃[i, j, k-1]) / β
            upper = - g * Δt * C⁺ * a⁺
        end

        for k in Nz-1:-1:1
            θ̃[i, j, k] -= γ[i, j, k+1] * θ̃[i, j, k+1]
        end
    end
end

# The enthalpy of each layer per unit of the coordinate, given to `change_column_surface_pressure!`,
# which expresses it as the potential temperature at the new pressure and with the new water.
#
# The vapor δm of the ground enters with the kinetic energy of the air it joins. The mixing of
# the velocities has given it to the lowest layer at the velocity of the start of the step, δm 𝒦ⁿ₁,
# and the change of the surface pressure then gives every layer its share ΔB δm at the velocity
# it now has, while the lowest layer gives its δm back: the kinetic energy of the column changes
# by δm 𝒦ⁿ₁ + δm (Σ ΔB 𝒦⁺ - 𝒦⁺₁). Both terms cross the ground with the vapor, and the ledger of
# the surface energy books them, so that no energy is taken from the enthalpy for them.
@kernel function _express_enthalpy_as_potential_temperature!(ρθ, column_enthalpy, surface_energy, vapor_kinetic_energy,
                                                             grid, θ̃, ρqᵗ, scaled, ρqˣ, vapor_exchange, 𝒦ⁿ, u, v,
                                                             dynamics, constants, κ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    ρ = dynamics.density
    Π = dynamics.exner_function
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity

    @inbounds for k in 1:Nz
        column_enthalpy[i, j, k] = ρ[i, j, k] * enthalpy_factor(i, j, k, ρ, ρqᵗ, Π, cᵖᵈ, cᵖᵛ) * θ̃[i, j, k]
    end

    @inbounds begin
        δm = vapor_exchange[i, j, 1]
        δpˢ = g * δm
        B = dynamics.surface_pressure_coefficient
        kinetic = δm * (𝒦ⁿ[i, j, 1] - specific_kinetic_energy(i, j, 1, grid, u, v)) +
                  surface_pressure_kinetic_energy(i, j, grid, δpˢ, B, u, v, g)
        surface_energy[i, j, 1] += kinetic
        vapor_kinetic_energy[i, j, 1] += kinetic
    end

    change_column_surface_pressure!(i, j, grid, δpˢ, ρθ, column_enthalpy, scaled, ρqˣ, dynamics, constants, κ, g)
end

function launch_energy_mixing!(mixing, model, Δt)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    moisture = dynamics.moisture
    constants = model.thermodynamic_constants
    closure_fields = mixing.closure_fields
    scratch = mixing.scratch
    fluxes = mixing.surface_fluxes
    C = scratch.conductances
    γ = scratch.eliminated_diagonal
    u, v, _ = model.velocities
    water = hydrostatic_water_fields(model.microphysics, model.microphysical_fields)
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity

    ledger = mixing.ledger
    conversions = (shear_production = ledger.shear_production,
                   buoyancy_source = ledger.buoyancy_source,
                   turbulent_kinetic_energy_sinks = ledger.turbulent_kinetic_energy_sinks,
                   water_kinetic_energy = ledger.water_kinetic_energy)

    launch!(arch, grid, :xy, _step_turbulent_kinetic_energy!,
            model.tracers[TKE_NAME], scratch.specific_turbulent_kinetic_energy, scratch.implicit_mass, γ,
            scratch.heating, conversions, grid, scratch.total_density, dynamics.density, C.energy, C.momentum,
            closure_fields.Lᵉ, closure_fields.Kᶜ, closure_fields.N², u, v, scratch.x_velocity, scratch.y_velocity, Δt)

    ρ = dynamics.density
    ρqᵗ = scratch.initial_water_density
    A = dynamics.pressure_coefficient
    B = dynamics.surface_pressure_coefficient
    pˢ = dynamics.surface_pressure
    Π = dynamics.exner_function
    θ̃ = scratch.potential_temperature
    g = constants.gravitational_acceleration
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity

    launch!(arch, grid, :xy, _add_surface_energy_sources!,
            scratch.heating, ledger.surface_energy, ledger.surface_stress_work, grid, scratch.total_density,
            dynamics.surface_geopotential, fluxes.sensible_heat_flux, scratch.vapor_exchange, fluxes.surface_temperature,
            scratch.x_velocity, scratch.y_velocity, fluxes.x_stress, fluxes.y_stress, cᵖᵛ, Δt)

    launch!(arch, grid, :xy, _solve_enthalpy!,
            θ̃, γ, grid, ρ, ρqᵗ, (rain = water.ρqʳ, snow = water.ρqˢ), A, B, pˢ, model.formulation.potential_temperature,
            moisture.density_potential_temperature, Π, dynamics.interface_exner_function, model.temperature,
            C.scalar, scratch.heating, constants, g, Δt)

    launch!(arch, grid, :xy, _express_enthalpy_as_potential_temperature!,
            model.formulation.potential_temperature_density, moisture.column_enthalpy, ledger.surface_energy,
            ledger.surface_vapor_kinetic_energy, grid, θ̃, ρqᵗ, column_mass_densities(model), water_densities(model),
            scratch.vapor_exchange, scratch.kinetic_energy, u, v, dynamics, constants, κ, g)

    return nothing
end

#####
##### One step
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance `model` by one backward Euler step `Δt` of the vertical turbulent `mixing`, on the
state the model is in (first-order sequential splitting after the dynamics):

1. the surface fluxes, if `mixing` has a [`BulkSurfaceExchange`](@ref), and the heights of the
   column;
2. the static stability, the mixing length, the diffusivities and the sink rate of the
   closure, from its own kernels;
3. the water that does not fall, with the vapor flux of the ground;
4. the velocities, with the stress of the ground;
5. the turbulent kinetic energy, which receives the kinetic energy the velocities lost;
6. the enthalpy, with the heat fluxes of the ground and the heat that the turbulent kinetic
   energy returns, then the potential temperature of that enthalpy.

Rain is not mixed. The state is diagnosed again at the end, which is where the phases of the
water adjust to the mixed state.
"""
# The halos of the diagnosed state that the mixing reads across columns (its closure fields are
# computed one cell into the horizontal halos, and the bulk exchange and the conductances read
# neighbouring cells): the temperature, the partition of the water, the layer pressure and the
# total surface pressure. `update_state!` of a moist hydrostatic model does not fill them, because
# the kernels of the dynamics read them in their own columns only (`fill_auxiliary_thermodynamic_halos!`).
function fill_mixing_halos!(model)
    dynamics = model.dynamics
    fill_halo_regions!(model.temperature)
    fill_halo_regions!(model.microphysical_fields, boundary_condition_args(model)...)
    fill_halo_regions!(dynamics.pressure)
    fill_halo_regions!(dynamics.surface_pressure)
    return nothing
end

function moist_turbulent_mixing_step!(mixing::MoistTurbulentMixing, model, Δt)
    (isfinite(Δt) && Δt > 0) || return nothing

    grid = model.grid
    scratch = mixing.scratch
    fluxes = mixing.surface_fluxes
    C = scratch.conductances
    u, v, _ = model.velocities
    Δt = convert(eltype(grid), Δt)

    update_state!(model; compute_tendencies=false)
    fill_mixing_halos!(model)

    compute_bulk_surface_fluxes!(fluxes, mixing.surface_exchange, model)
    fill_halo_regions!((fluxes.x_stress, fluxes.y_stress))

    compute_physical_heights!(mixing.heights, model)

    @apply_regionally launch_specific_energies!(mixing, model)
    fill_halo_regions!(scratch.specific_turbulent_kinetic_energy)

    @apply_regionally launch_closure_fields_in_height!(mixing, model)
    fill_halo_regions!(C.momentum)
    clip_negative_halo_diffusivities!(tuple(C.momentum), grid)

    @apply_regionally launch_water_mixing!(mixing, model, Δt)
    fill_halo_regions!(scratch.total_density)

    @apply_regionally launch_surface_stress!(mixing, model, Δt)
    fill_halo_regions!((scratch.x_velocity, scratch.y_velocity))

    @apply_regionally launch_velocity_mixing!(mixing, model, Δt)
    fill_halo_regions!((u, v))

    @apply_regionally launch_energy_mixing!(mixing, model, Δt)

    update_state!(model; compute_tendencies=true)

    return nothing
end

#####
##### The column physics of a time step, in one dispatch
#####

struct MoistColumnPhysics{M}
    mixing :: M               # MoistTurbulentMixing
    warm_rain_substeps :: Int # applications of the warm rain per time step
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a callable for `Callback(MoistColumnPhysics(mixing), IterationInterval(1))` that applies
the column physics of a moist hydrostatic model once after every time step, over the step just
taken and each on the state the previous one leaves: the warm rain of the microphysics
(`warm_rain_column_physics!`, in `warm_rain_substeps` applications), then the turbulent `mixing`
([`moist_turbulent_mixing_step!`](@ref)).

The time stepper must not apply the column physics as well: call
`column_physics_in_time_step!(model, false)` first, without which the callable throws an
`ArgumentError`. With `warm_rain_substeps = 1` the result is that of the time stepper applying
the warm rain and of `mixing` as a callback after it.
"""
MoistColumnPhysics(mixing::MoistTurbulentMixing; warm_rain_substeps=1) = MoistColumnPhysics(mixing, warm_rain_substeps)

# The ledger of the mixing is its only state
Oceananigans.prognostic_state(physics::MoistColumnPhysics) = Oceananigans.prognostic_state(physics.mixing)

function Oceananigans.restore_prognostic_state!(physics::MoistColumnPhysics, from::NamedTuple)
    Oceananigans.restore_prognostic_state!(physics.mixing, from)
    return physics
end

Oceananigans.restore_prognostic_state!(physics::MoistColumnPhysics, ::Nothing) = physics

function (physics::MoistColumnPhysics)(simulation)
    model = simulation.model

    column_physics_in_time_step(model) &&
        throw(ArgumentError("MoistColumnPhysics would apply the warm rain a second time: the time stepper " *
                            "applies it at the end of every step. Call " *
                            "column_physics_in_time_step!(model, false) before the first step."))

    warm_rain_column_physics!(model.microphysics, model; substeps=physics.warm_rain_substeps)
    moist_turbulent_mixing_step!(physics.mixing, model, model.clock.last_Δt)

    return nothing
end

end # module
