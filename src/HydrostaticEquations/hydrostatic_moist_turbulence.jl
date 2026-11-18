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
2. the conservative insertion of the turbulent fluxes in the budgets of the dry-mass coordinate:
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

export MoistTurbulentMixing, moist_turbulent_mixing_step!, BulkSurfaceExchange, PhysicalHeightGrid

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, Center, Face, CenterField, XFaceField, YFaceField, ZFaceField, Field
using Oceananigans.BoundaryConditions: fill_halo_regions!, getbc, Bottom
using Oceananigans.Grids: AbstractGrid, topology
using Oceananigans.MultiRegion: MultiRegion
using Oceananigans.Operators: Operators, Δzᶜᶜᶜ, ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ
using Oceananigans.TimeSteppers: update_state!, time_discretization
using Oceananigans.TurbulenceClosures: TurbulenceClosures as OceanTurbulenceClosures,
                                       VerticallyImplicitTimeDiscretization, build_closure_fields
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: ℑbzᵃᵃᶜ
using Oceananigans.Utils: KernelParameters, launch!

using Breeze.AtmosphereModels: AtmosphereModelBuoyancy, specific_prognostic_moisture, dynamics_thermodynamic_fields
using Breeze.BoundaryConditions: BoundaryConditions as AtmosphereBoundaryConditions
using Breeze.HydrostaticEquations: MoistHydrostaticModel, hydrostatic_water_fields,
                                   water_densities, sum_at, interface_pressure, layer_pressure_thickness,
                                   interface_potential_temperature, height_between_layers, physical_layer_depth,
                                   layer_mass_per_area, diffuse_column!, face_conductance, _diffuse_velocity!
using Breeze.Thermodynamics: dry_air_gas_constant, absolute_zero_latent_heat
using Breeze.TurbulenceClosures: TKEBasedTurbulenceClosure, TKE_NAME, buoyancy_productionᶜᶜᶠ,
                                 _compute_tke_static_stability!, _compute_mixing_length!,
                                 _compute_tke_closure_fields!, _compute_tke_implicit_linear_coefficient!
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS

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
that state only.
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

PhysicalHeightGrid(grid) = PhysicalHeightGrid(grid.architecture, grid, CenterField(grid), ZFaceField(grid), ZFaceField(grid))

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

function compute_physical_heights!(physical_grid::PhysicalHeightGrid, model)
    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants

    launch!(grid.architecture, grid, :xy, _compute_physical_heights!,
            physical_grid.layer_depth, physical_grid.interface_spacing, physical_grid.interface_height, grid,
            dynamics.moisture.density_potential_temperature, dynamics.exner_function,
            dynamics.interface_exner_function, constants.dry_air.heat_capacity, constants.gravitational_acceleration)

    fill_halo_regions!(physical_grid.layer_depth)
    fill_halo_regions!(physical_grid.interface_spacing)
    fill_halo_regions!(physical_grid.interface_height)

    return nothing
end

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
            surface_energy = Field{Center, Center, Nothing}(grid),        # J m⁻²: 𝒬 + Jᵛ (cᵖᵛ Tˢ + Φˢ)
            surface_stress_work = Field{Center, Center, Nothing}(grid),   # J m⁻²: kinetic energy removed by the stress
            shear_production = Field{Center, Center, Nothing}(grid),      # J m⁻²: kinetic energy → TKE
            buoyancy_source = Field{Center, Center, Nothing}(grid),       # J m⁻²: enthalpy → TKE (B > 0)
            turbulent_kinetic_energy_sinks = Field{Center, Center, Nothing}(grid), # J m⁻²: TKE → enthalpy
            water_kinetic_energy = Field{Center, Center, Nothing}(grid))  # J m⁻²: enthalpy → (kinetic + TKE) of water changing layer
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

materialize_bulk_condition(::Nothing, location, model, closure) = nothing

function materialize_bulk_condition(bc, location, model, closure)
    coefficient = bc.condition.coefficient
    coefficient isa Number ||
        throw(ArgumentError("BulkSurfaceExchange supports constant transfer coefficients only: the stability " *
                            "functions of $(summary(coefficient)) take heights from the grid, which on " *
                            "HydrostaticDynamics are labels of the coordinate."))

    isnothing(bc.condition.surface_temperature) &&
        throw(ArgumentError("The bulk boundary conditions of BulkSurfaceExchange need a surface_temperature."))

    materialized = AtmosphereBoundaryConditions.materialize_atmosphere_boundary_condition(
        bc, Bottom(), location, model.grid, model.dynamics, model.microphysics, model.thermodynamic_constants, closure)

    formulation = AtmosphereBoundaryConditions.sensible_heat_flux_formulation(Val(:ρθ), model.formulation)
    return AtmosphereBoundaryConditions.set_sensible_heat_formulation(materialized, formulation).condition
end

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
@kernel function _compute_bulk_surface_fluxes!(fluxes, grid, exchange, clock, fields, dynamics_fields, ρ, ρᵗ, Π, T, cᵖᵈ, cᵖᵛ)
    i, j = @index(Global, NTuple)

    @inbounds begin
        fluxes.x_stress[i, j, 1] = bulk_flux(exchange.x_drag, i, j, grid, clock, fields, dynamics_fields)
        fluxes.y_stress[i, j, 1] = bulk_flux(exchange.y_drag, i, j, grid, clock, fields, dynamics_fields)

        J = bulk_flux(exchange.sensible_heat, i, j, grid, clock, fields, dynamics_fields)
        ĉᵖ = (cᵖᵈ * ρ[i, j, 1] + cᵖᵛ * (ρᵗ[i, j, 1] - ρ[i, j, 1])) / ρᵗ[i, j, 1]
        fluxes.sensible_heat_flux[i, j, 1] = ĉᵖ * Π[i, j, 1] * J

        fluxes.vapor_flux[i, j, 1] = bulk_flux(exchange.vapor, i, j, grid, clock, fields, dynamics_fields)
        fluxes.surface_temperature[i, j, 1] = bulk_surface_temperature(exchange.vapor, i, j, grid, clock, T)
    end
end

compute_bulk_surface_fluxes!(fluxes, ::Nothing, model) = nothing

function compute_bulk_surface_fluxes!(fluxes, exchange::BulkSurfaceExchange, model)
    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants

    launch!(grid.architecture, grid, :xy, _compute_bulk_surface_fluxes!,
            fluxes, grid, exchange, model.clock, Oceananigans.fields(model), dynamics_thermodynamic_fields(dynamics),
            dynamics.density, dynamics.moisture.total_density, dynamics.exner_function, model.temperature,
            constants.dry_air.heat_capacity, constants.vapor.heat_capacity)

    fill_halo_regions!(fluxes.x_stress)
    fill_halo_regions!(fluxes.y_stress)

    return nothing
end

#####
##### The mixing
#####

struct MoistTurbulentMixing{C, CF, G, F, X, S, L}
    closure :: C           # the `TKEBasedTurbulenceClosure`, as it is
    closure_fields :: CF   # its fields: Kᵘ, Kᶜ, Kᵉ, N², ℓ, Lᵉ, in m² s⁻¹, s⁻², m and s⁻¹
    physical_grid :: G     # the heights of the column behind the operators of the grid
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
of the hydrostatic column, and the fluxes enter the budgets of the dry-mass coordinate. The
tracer `ρe` is the turbulent kinetic energy per unit area and unit of the grid's coordinate,
`e` times the total density, which the model advects.

Keyword arguments
=================

- `closure`: the closure. Default `TKEBasedTurbulenceClosure(eltype(model.grid))`.
- `surface_exchange`: `nothing` (default), for which the fields of `mixing.surface_fluxes` are
  used as they are (zero unless set), or a [`BulkSurfaceExchange`](@ref) that fills them from
  the state before every step.
"""
function MoistTurbulentMixing(model::MoistHydrostaticModel;
                              closure = TKEBasedTurbulenceClosure(eltype(model.grid)),
                              surface_exchange = nothing)
    grid = model.grid

    grid isa MultiRegion.MultiRegionGrids &&
        throw(ArgumentError("MoistTurbulentMixing does not support multi-region grids yet."))

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
               kinetic_energy = CenterField(grid),
               heating = CenterField(grid),
               potential_temperature = CenterField(grid),
               x_velocity = XFaceField(grid),
               y_velocity = YFaceField(grid),
               x_dissipation = XFaceField(grid),
               y_dissipation = YFaceField(grid),
               vapor_exchange = Field{Center, Center, Nothing}(grid),
               x_stress_work = Field{Face, Center, Nothing}(grid),
               y_stress_work = Field{Center, Face, Nothing}(grid),
               conductances = (scalar = ZFaceField(grid), momentum = ZFaceField(grid), energy = ZFaceField(grid)))

    return MoistTurbulentMixing(closure, closure_fields, PhysicalHeightGrid(grid), surface_flux_fields(grid),
                                surface_exchange, scratch, turbulence_ledger(grid))
end

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

# e = ρe / ρᵗ, and the kinetic energy of the layer per unit mass at cell centers
@kernel function _compute_specific_energies!(e, 𝒦, grid, ρe, ρᵗ, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        e[i, j, k] = ρe[i, j, k] / ρᵗ[i, j, k]
        𝒦[i, j, k] = (ℑxᶜᵃᵃ(i, j, k, grid, squared, u) + ℑyᵃᶜᵃ(i, j, k, grid, squared, v)) / 2
    end
end

@inline squared(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

# The kernels of the closure, launched as they are with the heights of the column for a grid.
# The static stability is computed on the faces between two layers; it stays zero on the ground
# and on the model top, where the closure masks its diffusivities.
function compute_closure_fields_in_height!(mixing, model)
    grid = model.grid
    arch = grid.architecture
    physical_grid = mixing.physical_grid
    closure = mixing.closure
    closure_fields = mixing.closure_fields
    e = mixing.scratch.specific_turbulent_kinetic_energy
    Nx, Ny, Nz = size(grid)

    buoyancy = AtmosphereModelBuoyancy(model.dynamics, model.formulation, model.thermodynamic_constants,
                                       model.microphysics, model.microphysical_fields)

    tracers = NamedTuple{(TKE_NAME, :T, :qᵛᵉ)}((e, model.temperature, specific_prognostic_moisture(model)))

    launch!(arch, grid, KernelParameters(1:Nx, 1:Ny, 2:Nz), _compute_tke_static_stability!,
            closure_fields.N², physical_grid, closure, tracers, buoyancy)

    launch!(arch, grid, :xy, _compute_mixing_length!,
            closure_fields.ℓᵐ, closure_fields.ℓˢ, physical_grid, closure, e, closure_fields.N²)

    launch!(arch, grid, :xyz, _compute_tke_closure_fields!,
            closure_fields, physical_grid, closure, model.velocities, tracers)

    launch!(arch, grid, :xyz, _compute_tke_implicit_linear_coefficient!,
            closure_fields.Lᵉ, physical_grid, closure, closure_fields, model.velocities, tracers)

    return nothing
end

#####
##### Conductances
#####

# C = ρᵃ K / Δh on the z-faces between two layers, in kg m⁻² s⁻¹, with ρᵃ = p / (Rᵈ Tᵨ) the
# density of the air on the face at its total pressure and density temperature Tᵨ = Πᶠ θᵨ, and
# Δh the height between the layers. Momentum and turbulent kinetic energy are carried by the
# air and its water; the scalars by the dry air alone, ρᵈ = qᵈ ρᵃ, so that the exchange leaves
# the dry mass of every layer as the coordinate has it. Without water these are the
# conductances of the dry `VerticalDiffusion`.
@kernel function _compute_moist_conductances!(Cᶜ, Cᵘ, Cᵉ, grid, Kᶜ, Kᵘ, Kᵉ, θᵨ, Π, Πᶠ, pᶠ, ρ, ρᵗ, Rᵈ, cᵖᵈ, g)
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
            qᵈ = (ρ[i, j, k] / ρᵗ[i, j, k] + ρ[i, j, k-1] / ρᵗ[i, j, k-1]) / 2

            Cᶜ[i, j, k] = ρᵃ * Kᶜ[i, j, k] / Δh * qᵈ
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

# One backward Euler step of the mixing ratio r = ρqᵉ / ρ of the water that does not fall, per
# unit dry mass, with the dry mass of the layers for weights. The vapor flux through the ground
# enters the lowest layer; a flux out of the air (dew) is limited to the water of that layer,
# and the ledger holds what was exchanged.
@kernel function _mix_water!(ρqᵉ, r, γ, ρᵗ⁺, exchanged, ledger, grid, ρ, C, Jᵛ, ρqˣ, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz
            r[i, j, k] = ρqᵉ[i, j, k] / ρ[i, j, k]
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
        ρᵗ⁺[i, j, k] = ρ[i, j, k] + sum_at(i, j, k, ρqˣ)
    end
end

#####
##### Velocities
#####

# The stress of the ground accelerates the lowest layer, of total mass μ per unit area at the
# velocity point. `work` is the kinetic energy it removes, μ (uⁿ² - u★²) / 2.
@kernel function _apply_surface_stress!(velocity, work, grid, ρᵗ, τ, Δt, interpolate)
    i, j = @index(Global, NTuple)

    @inbounds begin
        μ = interpolate(i, j, 1, grid, layer_mass_per_area, ρᵗ)
        uⁿ = velocity[i, j, 1]
        u★ = uⁿ + Δt * τ[i, j, 1] / μ
        work[i, j, 1] = μ * (uⁿ^2 - u★^2) / 2
        velocity[i, j, 1] = u★
    end
end

@kernel function _copy_field!(destination, source)
    i, j, k = @index(Global, NTuple)
    @inbounds destination[i, j, k] = source[i, j, k]
end

# The kinetic energy that the backward Euler step from u★ to u⁺ removes from the layer `k`, per
# unit area at the velocity point:
#
#   d[k] = Δt (C⁻ δu⁻² + C⁺ δu⁺²) / 2 + μ (u⁺ - u★)² / 2 ≥ 0 ,
#
# half the dissipation C δu² of each of its two faces, at the new velocities, and the
# dissipation of the time discretization. The sum over the column is the kinetic energy lost,
# Σ μ (u★² - u⁺²) / 2, because the conductance vanishes on the first and the last face.
@kernel function _compute_velocity_dissipation!(d, grid, u⁺, u★, ρᵗ, C, Δt, interpolate)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        μ = interpolate(i, j, k, grid, layer_mass_per_area, ρᵗ)
        C⁻ = interpolate(i, j, k, grid, C)
        C⁺ = interpolate(i, j, k+1, grid, C)
        δu⁻ = u⁺[i, j, k] - u⁺[i, j, k-1]
        δu⁺ = u⁺[i, j, k+1] - u⁺[i, j, k]
        d[i, j, k] = Δt * (C⁻ * δu⁻^2 + C⁺ * δu⁺^2) / 2 + μ * (u⁺[i, j, k] - u★[i, j, k])^2 / 2
    end
end

function mix_velocity!(velocity, velocity★, dissipation, work, stress, interpolate, mixing, model, Δt)
    grid = model.grid
    arch = grid.architecture
    ρᵗ⁺ = mixing.scratch.total_density
    C = mixing.scratch.conductances.momentum
    γ = mixing.scratch.eliminated_diagonal

    launch!(arch, grid, :xy, _apply_surface_stress!, velocity, work, grid, ρᵗ⁺, stress, Δt, interpolate)
    launch!(arch, grid, :xyz, _copy_field!, velocity★, velocity)
    launch!(arch, grid, :xy, _diffuse_velocity!, velocity, γ, grid, ρᵗ⁺, C, Δt, interpolate)
    launch!(arch, grid, :xyz, _compute_velocity_dissipation!, dissipation, grid, velocity, velocity★, ρᵗ⁺, C, Δt, interpolate)

    fill_halo_regions!(dissipation)
    fill_halo_regions!(work)

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
# turbulent kinetic energy takes from buoyancy, plus what its sinks return, minus the kinetic
# and turbulent kinetic energy that the layer gains with the water it gains.
@kernel function _step_turbulent_kinetic_energy!(ρe, e, μ′, γ, heating, ledger, grid, ρᵗ⁺, ρᵗ, Cᵉ, Lᵉ, Kᶜ, N²,
                                                 dᵘ, dᵛ, 𝒦, Δt)
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

        𝒫 = ℑxᶜᵃᵃ(i, j, k, grid, dᵘ) + ℑyᵃᶜᵃ(i, j, k, grid, dᵛ)
        B = ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, Kᶜ, N²)
        ℬ = Δt * μ * max(0, B)

        μ′[i, j, k] = μ * (1 - Δt * Lᵉ[i, j, k])
        e[i, j, k] = (μ * eⁿ + 𝒫 + ℬ) / μ′[i, j, k]
        heating[i, j, k] = - ℬ - δμ * (𝒦[i, j, k] + eⁿ)

        production += 𝒫
        buoyancy_source += ℬ
        water_energy += δμ * (𝒦[i, j, k] + eⁿ)
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

# Enthalpy of the air of a layer per unit dry mass over its potential temperature, c̃ Π with
# c̃ = cᵖᵈ + rᵗ cᵖᵛ: the enthalpy of the layer is Δπ c̃ Π θ / g.
@inline enthalpy_factor(i, j, k, ρ, ρᵗ, Π, cᵖᵈ, cᵖᵛ) =
    @inbounds (cᵖᵈ + (ρᵗ[i, j, k] - ρ[i, j, k]) / ρ[i, j, k] * cᵖᵛ) * Π[i, j, k]

# Mass of the air and of the water that does not fall, per unit dry mass: 1 + rᵉ
@inline nonprecipitating_loading(i, j, k, ρ, ρᵗ, ρqʳ) = @inbounds (ρᵗ[i, j, k] - ρqʳ[i, j, k]) / ρ[i, j, k]

# Enthalpy of the rain of a layer per unit dry mass, rʳ (cˡ T - ℒˡ₀), which the turbulence does
# not carry
@inline rain_enthalpy(i, j, k, ρ, ρqʳ, T, cˡ, ℒˡ₀) = @inbounds ρqʳ[i, j, k] / ρ[i, j, k] * (cˡ * T[i, j, k] - ℒˡ₀)

# The flux of enthalpy through the z-face `k`, between the layers `k - 1` and `k`, is down the
# gradient of the static energy of the air and of the water that does not fall, per unit dry mass:
#
#   F = - Cᶜ [χ[k] - χ[k-1] + (1 + r̄ᵉ) g Δh - (χʳ[k] - χʳ[k-1])] ,    χ = c̃ Π θ ,
#
# with g Δh = cᵖᵈ [θᵨ[k-1] (Π[k-1] - Πᶠ) + θᵨ[k] (Πᶠ - Π[k])] the geopotential between the two
# layers and χʳ the enthalpy of the rain. With θᵨ = ϑ θ it is linear in θ,
#
#   F = - Cᶜ (a⁺ θ[k] - a⁻ θ[k-1] + R) ,
#   a⁺ = c̃ Π[k] + (1 + r̄ᵉ) cᵖᵈ ϑ[k] (Πᶠ - Π[k]) ,    a⁻ = c̃ Π[k-1] - (1 + r̄ᵉ) cᵖᵈ ϑ[k-1] (Π[k-1] - Πᶠ) ,
#
# and without water a⁺ = a⁻ = cᵖᵈ Πᶠ: the flux of the dry `VerticalDiffusion` times cᵖᵈ Πᶠ.
@inline function upper_enthalpy_coefficient(i, j, k, ρ, ρᵗ, ρqʳ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
    loading = (nonprecipitating_loading(i, j, k, ρ, ρᵗ, ρqʳ) + nonprecipitating_loading(i, j, k-1, ρ, ρᵗ, ρqʳ)) / 2
    ϑ = @inbounds θᵨ[i, j, k] / θ[i, j, k]
    return @inbounds enthalpy_factor(i, j, k, ρ, ρᵗ, Π, cᵖᵈ, cᵖᵛ) + loading * cᵖᵈ * ϑ * (Πᶠ[i, j, k] - Π[i, j, k])
end

@inline function lower_enthalpy_coefficient(i, j, k, ρ, ρᵗ, ρqʳ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
    loading = (nonprecipitating_loading(i, j, k, ρ, ρᵗ, ρqʳ) + nonprecipitating_loading(i, j, k-1, ρ, ρᵗ, ρqʳ)) / 2
    ϑ = @inbounds θᵨ[i, j, k-1] / θ[i, j, k-1]
    return @inbounds enthalpy_factor(i, j, k-1, ρ, ρᵗ, Π, cᵖᵈ, cᵖᵛ) - loading * cᵖᵈ * ϑ * (Π[i, j, k-1] - Πᶠ[i, j, k])
end

@inline rain_enthalpy_difference(i, j, k, ρ, ρqʳ, T, cˡ, ℒˡ₀) =
    rain_enthalpy(i, j, k-1, ρ, ρqʳ, T, cˡ, ℒˡ₀) - rain_enthalpy(i, j, k, ρ, ρqʳ, T, cˡ, ℒˡ₀)

# One backward Euler step of the enthalpy of the layers, Δπ c̃ Π θ / g, with the composition and
# the pressure of the start of the step in the coefficients, by the Thomas algorithm for the
# unknown θ̃. Row `k`, multiplied by g:
#
#   Δπ c̃ Π (θ̃[k] - θ[k]) = g Δt {Cᶜ⁺ (a⁺⁺ θ̃[k+1] - a⁻⁺ θ̃[k] + R⁺) - Cᶜ⁻ (a⁺⁻ θ̃[k] - a⁻⁻ θ̃[k-1] + R⁻)} + g S ,
#
# with S the sources in J m⁻²: the fluxes of the ground into the lowest layer, the heating of
# `_step_turbulent_kinetic_energy!` and the work of the surface stress. Every column of the
# matrix sums to Δπ c̃ Π > 0 and its off-diagonal entries are not positive.
#
# The enthalpy Δπ c̃ Π θ̃ / g of each layer is then expressed as the potential temperature of the
# layer with the water it now holds, under the pressure of the water now above it, as
# `_sediment_rain!` does: the column loses or gains enthalpy by its sources only.
@kernel function _mix_enthalpy!(ρθ, θ̃, γ, ledger, grid, ρ, ρᵗ, ρqʳ, ρqˣ, A, B, πˢ, Φˢ, θ, θᵨ, Π, Πᶠ, T, C, heating,
                                𝒬, Jᵛ_applied, Tˢ, wˣ, wʸ, constants, pˢᵗ, κ, g, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)

    @inbounds begin
        # Sources of the lowest layer, in J m⁻²
        vapor = Jᵛ_applied[i, j, 1]
        stress_work = ℑxᶜᵃᵃ(i, j, 1, grid, wˣ) + ℑyᵃᶜᵃ(i, j, 1, grid, wʸ)
        surface = Δt * 𝒬[i, j, 1] + vapor * cᵖᵛ * Tˢ[i, j, 1]

        ledger.surface_energy[i, j, 1] += surface + vapor * Φˢ[i, j, 1]
        ledger.surface_stress_work[i, j, 1] += stress_work

        # Forward elimination
        C⁺ = C[i, j, 2]
        a⁺ = upper_enthalpy_coefficient(i, j, 2, ρ, ρᵗ, ρqʳ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
        a⁻ = lower_enthalpy_coefficient(i, j, 2, ρ, ρᵗ, ρqʳ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
        R⁺ = rain_enthalpy_difference(i, j, 2, ρ, ρqʳ, T, cˡ, ℒˡ₀)

        Δπ = layer_pressure_thickness(i, j, 1, A, B, πˢ)
        M = Δπ * enthalpy_factor(i, j, 1, ρ, ρᵗ, Π, cᵖᵈ, cᵖᵛ)
        S = heating[i, j, 1] + surface + stress_work

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
            a⁺ = upper_enthalpy_coefficient(i, j, m, ρ, ρᵗ, ρqʳ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
            a⁻ = lower_enthalpy_coefficient(i, j, m, ρ, ρᵗ, ρqʳ, θ, θᵨ, Π, Πᶠ, cᵖᵈ, cᵖᵛ)
            R⁺ = rain_enthalpy_difference(i, j, m, ρ, ρqʳ, T, cˡ, ℒˡ₀)

            Δπ = layer_pressure_thickness(i, j, k, A, B, πˢ)
            M = Δπ * enthalpy_factor(i, j, k, ρ, ρᵗ, Π, cᵖᵈ, cᵖᵛ)
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

        # The potential temperature of the enthalpy of each layer, at its new pressure and with
        # its new water
        ΣW = zero(grid)
        p⁺ = interface_pressure(i, j, Nz+1, A, B, πˢ)
        Πᶠ⁺ = (p⁺ / pˢᵗ)^κ

        for k in Nz:-1:1
            Δπ = layer_pressure_thickness(i, j, k, A, B, πˢ)
            𝓗 = Δπ * enthalpy_factor(i, j, k, ρ, ρᵗ, Π, cᵖᵈ, cᵖᵛ) * θ̃[i, j, k]

            W = g * Δzᶜᶜᶜ(i, j, k, grid) * sum_at(i, j, k, ρqˣ)
            ΣW += W
            p⁻ = interface_pressure(i, j, k, A, B, πˢ) + ΣW
            Πᶠ⁻ = (p⁻ / pˢᵗ)^κ
            Π⁺ = (p⁻ * Πᶠ⁻ - p⁺ * Πᶠ⁺) / ((1 + κ) * (Δπ + W))

            ρθ[i, j, k] = ρ[i, j, k] * 𝓗 / ((Δπ * cᵖᵈ + W * cᵖᵛ) * Π⁺)

            p⁺ = p⁻
            Πᶠ⁺ = Πᶠ⁻
        end
    end
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
function moist_turbulent_mixing_step!(mixing::MoistTurbulentMixing, model, Δt)
    (isfinite(Δt) && Δt > 0) || return nothing

    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    moisture = dynamics.moisture
    constants = model.thermodynamic_constants
    closure_fields = mixing.closure_fields
    scratch = mixing.scratch
    fluxes = mixing.surface_fluxes
    ledger = mixing.ledger
    water = hydrostatic_water_fields(model.microphysics, model.microphysical_fields)
    C = scratch.conductances
    γ = scratch.eliminated_diagonal
    e = scratch.specific_turbulent_kinetic_energy
    ρᵗ⁺ = scratch.total_density
    u, v, _ = model.velocities
    ρ = dynamics.density
    ρᵗ = moisture.total_density
    θ = model.formulation.potential_temperature
    θᵨ = moisture.density_potential_temperature
    Π = dynamics.exner_function
    Πᶠ = dynamics.interface_exner_function
    ρe = model.tracers[TKE_NAME]
    Δt = convert(eltype(grid), Δt)
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = Rᵈ / cᵖᵈ

    update_state!(model; compute_tendencies=false)

    compute_bulk_surface_fluxes!(fluxes, mixing.surface_exchange, model)
    compute_physical_heights!(mixing.physical_grid, model)

    launch!(arch, grid, :xyz, _compute_specific_energies!, e, scratch.kinetic_energy, grid, ρe, ρᵗ, u, v)
    fill_halo_regions!(e)

    compute_closure_fields_in_height!(mixing, model)

    launch!(arch, grid, :xy, _compute_moist_conductances!,
            C.scalar, C.momentum, C.energy, grid, closure_fields.Kᶜ, closure_fields.Kᵘ, closure_fields.Kᵉ,
            θᵨ, Π, Πᶠ, moisture.interface_pressure, ρ, ρᵗ, Rᵈ, cᵖᵈ, g)

    fill_halo_regions!(C.momentum)

    # Water. The vapor exchanged with the ground in this step is kept for the enthalpy it carries.
    launch!(arch, grid, :xy, _mix_water!,
            model.moisture_density, scratch.mixing_ratio, γ, ρᵗ⁺, scratch.vapor_exchange, ledger, grid, ρ, C.scalar,
            fluxes.vapor_flux, water_densities(model), Δt)

    fill_halo_regions!(ρᵗ⁺)

    # Velocities
    mix_velocity!(u, scratch.x_velocity, scratch.x_dissipation, scratch.x_stress_work, fluxes.x_stress, ℑxᶠᵃᵃ,
                  mixing, model, Δt)
    mix_velocity!(v, scratch.y_velocity, scratch.y_dissipation, scratch.y_stress_work, fluxes.y_stress, ℑyᵃᶠᵃ,
                  mixing, model, Δt)

    # Turbulent kinetic energy
    launch!(arch, grid, :xy, _step_turbulent_kinetic_energy!,
            ρe, e, scratch.implicit_mass, γ, scratch.heating, ledger, grid, ρᵗ⁺, ρᵗ, C.energy,
            closure_fields.Lᵉ, closure_fields.Kᶜ, closure_fields.N², scratch.x_dissipation, scratch.y_dissipation,
            scratch.kinetic_energy, Δt)

    # Enthalpy
    launch!(arch, grid, :xy, _mix_enthalpy!,
            model.formulation.potential_temperature_density, scratch.potential_temperature, γ, ledger, grid,
            ρ, ρᵗ, water.ρqʳ, water_densities(model),
            dynamics.pressure_coefficient, dynamics.surface_pressure_coefficient, dynamics.surface_pressure,
            dynamics.surface_geopotential, θ, θᵨ, Π, Πᶠ, model.temperature, C.scalar, scratch.heating,
            fluxes.sensible_heat_flux, scratch.vapor_exchange, fluxes.surface_temperature,
            scratch.x_stress_work, scratch.y_stress_work, constants, dynamics.standard_pressure, κ, g, Δt)

    update_state!(model; compute_tendencies=true)

    return nothing
end

end # module
