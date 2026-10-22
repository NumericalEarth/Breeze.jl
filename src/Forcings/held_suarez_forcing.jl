using ..AtmosphereModels: AtmosphereModels
using Oceananigans: Oceananigans
using Oceananigans.Grids: Center, Face, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, φnode, znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.MultiRegion: MultiRegionGrid
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Units: day
using Adapt: Adapt

struct HeldSuarezParameters{FT}
    boundary_layer_top :: FT                         # σᵇ
    friction_rate :: FT                              # kᶠ
    atmosphere_relaxation_rate :: FT                 # kᵃ
    surface_relaxation_rate :: FT                    # kˢ
    equator_to_pole_temperature_difference :: FT     # ΔTʸ
    vertical_potential_temperature_difference :: FT  # Δθᶻ
    equatorial_surface_temperature :: FT             # Tᵉ₀
    minimum_equilibrium_temperature :: FT            # Tᵐⁱⁿ
    reference_pressure :: FT                         # p₀
    exner_exponent :: FT                             # κ
end

HeldSuarezParameters(FT = Oceananigans.defaults.FloatType;
                     boundary_layer_top = 0.7,
                     friction_rate = 1 / day,
                     atmosphere_relaxation_rate = 1 / 40day,
                     surface_relaxation_rate = 1 / 4day,
                     equator_to_pole_temperature_difference = 60,
                     vertical_potential_temperature_difference = 10,
                     equatorial_surface_temperature = 315,
                     minimum_equilibrium_temperature = 200,
                     reference_pressure = 1e5,
                     exner_exponent = 2/7) =
    HeldSuarezParameters{FT}(boundary_layer_top,
                             friction_rate,
                             atmosphere_relaxation_rate,
                             surface_relaxation_rate,
                             equator_to_pole_temperature_difference,
                             vertical_potential_temperature_difference,
                             equatorial_surface_temperature,
                             minimum_equilibrium_temperature,
                             reference_pressure,
                             exner_exponent)

HeldSuarezParameters{FT}(parameters::HeldSuarezParameters) where FT =
    HeldSuarezParameters{FT}(ntuple(n -> getfield(parameters, n), fieldcount(HeldSuarezParameters))...)

struct HeldSuarezForcing{V, P, A, R, D, C}
    variable :: V                 # Val(:ρu), Val(:ρv) or Val(:ρE)
    parameters :: P
    pressure :: A                 # p
    total_density :: R            # ρ
    coupling_density :: D         # ρᵈ
    thermodynamic_constants :: C
end

HeldSuarezForcing(variable, parameters) = HeldSuarezForcing(variable, parameters, nothing, nothing, nothing, nothing)

Adapt.adapt_structure(to, forcing::HeldSuarezForcing) =
    HeldSuarezForcing(forcing.variable,
                      forcing.parameters,
                      Adapt.adapt(to, forcing.pressure),
                      Adapt.adapt(to, forcing.total_density),
                      Adapt.adapt(to, forcing.coupling_density),
                      forcing.thermodynamic_constants)

@regional HeldSuarezForcing

const XMomentumHeldSuarezForcing = HeldSuarezForcing{Val{:ρu}}
const YMomentumHeldSuarezForcing = HeldSuarezForcing{Val{:ρv}}
const EnergyHeldSuarezForcing = HeldSuarezForcing{Val{:ρE}}

"""
$(TYPEDSIGNATURES)

Return `(; ρu, ρv, ρE)`, the Rayleigh friction and the Newtonian temperature relaxation of
[Held and Suarez (1994)](@cite HeldSuarez1994), to pass as `forcing` to
[`AtmosphereModel`](@ref Breeze.AtmosphereModels.AtmosphereModel) on a grid with latitude, alone or merged with other forcings.

Keyword arguments
=================

The defaults are the values of the paper.

- `boundary_layer_top`: top of the frictional boundary layer in `σ = p / pˢ`. Default: 0.7.
- `friction_rate`: Rayleigh friction rate at the surface. Default: 1 / day.
- `atmosphere_relaxation_rate`: relaxation rate of the free atmosphere. Default: 1 / 40 days.
- `surface_relaxation_rate`: relaxation rate at the equatorial surface. Default: 1 / 4 days.
- `equator_to_pole_temperature_difference`: of the equilibrium temperature. Default: 60 K.
- `vertical_potential_temperature_difference`: static stability of the equilibrium temperature.
  Default: 10 K.
- `equatorial_surface_temperature`: equilibrium temperature there. Default: 315 K.
- `minimum_equilibrium_temperature`: stratospheric floor of the equilibrium temperature. Default: 200 K.
- `reference_pressure`: reference pressure of the equilibrium temperature. Default: 10⁵ Pa.
- `exner_exponent`: exponent of `p / reference_pressure` in the equilibrium temperature. Default: 2/7.

```jldoctest
using Breeze

held_suarez_forcings()

# output
(ρu = HeldSuarezForcing on ρu, ρv = HeldSuarezForcing on ρv, ρE = HeldSuarezForcing on ρE)
```
"""
function held_suarez_forcings(FT = Oceananigans.defaults.FloatType; kw...)
    parameters = HeldSuarezParameters(FT; kw...)
    return (; ρu = HeldSuarezForcing(Val(:ρu), parameters),
              ρv = HeldSuarezForcing(Val(:ρv), parameters),
              ρE = HeldSuarezForcing(Val(:ρE), parameters))
end

# The kernels return density-weighted tendencies, so a specific key would multiply by ρ twice
AtmosphereModels.is_density_tendency_forcing(::HeldSuarezForcing) = true

#####
##### Pointwise functions
#####

@inline function boundary_layer_fraction(parameters, σ)
    σᵇ = parameters.boundary_layer_top
    return max(0, (σ - σᵇ) / (1 - σᵇ))
end

# The equilibrium temperature Tᵉ at latitude φ (degrees) and pressure p
@inline function equilibrium_temperature(parameters, φ, p)
    Tᵉ₀ = parameters.equatorial_surface_temperature
    ΔTʸ = parameters.equator_to_pole_temperature_difference
    Δθᶻ = parameters.vertical_potential_temperature_difference
    Tᵐⁱⁿ = parameters.minimum_equilibrium_temperature
    p₀ = parameters.reference_pressure
    κ = parameters.exner_exponent

    # Radians, because `sind` and `cosd` promote Float32 to Float64, which Metal cannot compile
    φʳ = deg2rad(φ)
    sin²φ = sin(φʳ)^2
    cos²φ = cos(φʳ)^2
    Tᵉ = (Tᵉ₀ - ΔTʸ * sin²φ - Δθᶻ * log(p / p₀) * cos²φ) * (p / p₀)^κ

    return max(Tᵐⁱⁿ, Tᵉ)
end

# The thermal relaxation rate kᵀ at latitude φ (degrees) and σ = p / pˢ
@inline function thermal_relaxation_rate(parameters, φ, σ)
    kᵃ = parameters.atmosphere_relaxation_rate
    kˢ = parameters.surface_relaxation_rate
    return kᵃ + (kˢ - kᵃ) * boundary_layer_fraction(parameters, σ) * cos(deg2rad(φ))^4
end

# The Rayleigh friction rate kᵘ at σ = p / pˢ
@inline momentum_damping_rate(parameters, σ) = parameters.friction_rate * boundary_layer_fraction(parameters, σ)

# The surface pressure below column (i, j), extrapolated from the lowest cell center to the bottom face
# in hydrostatic balance at fixed temperature. The index k is ignored, so that the function can be used
# in a `KernelFunctionOperation{Center, Center, Nothing}`.
@inline function hydrostatic_surface_pressure(i, j, k, grid, pressure, density, g)
    @inbounds p₁ = pressure[i, j, 1]
    @inbounds ρ₁ = density[i, j, 1]
    z₁ = znode(i, j, 1, grid, Center(), Center(), Center())
    zˢ = znode(i, j, 1, grid, Center(), Center(), Face())
    return p₁ * exp(g * ρ₁ * (z₁ - zˢ) / p₁)
end

@inline function σᶜᶜᶜ(i, j, k, grid, forcing)
    g = forcing.thermodynamic_constants.gravitational_acceleration
    pˢ = hydrostatic_surface_pressure(i, j, k, grid, forcing.pressure, forcing.total_density, g)
    @inbounds p = forcing.pressure[i, j, k]
    return p / pˢ
end

#####
##### Kernels
#####

@inline function (forcing::XMomentumHeldSuarezForcing)(i, j, k, grid, clock, fields)
    σ = ℑxᶠᵃᵃ(i, j, k, grid, σᶜᶜᶜ, forcing)
    kᵘ = momentum_damping_rate(forcing.parameters, σ)
    @inbounds ρu = fields.ρu[i, j, k]
    return - kᵘ * ρu
end

@inline function (forcing::YMomentumHeldSuarezForcing)(i, j, k, grid, clock, fields)
    σ = ℑyᵃᶠᵃ(i, j, k, grid, σᶜᶜᶜ, forcing)
    kᵘ = momentum_damping_rate(forcing.parameters, σ)
    @inbounds ρv = fields.ρv[i, j, k]
    return - kᵘ * ρv
end

# The enthalpy tendency ρᵈ cᵖᵈ ∂ₜT, which the thermodynamic tendency converts for the prognostic variable
@inline function (forcing::EnergyHeldSuarezForcing)(i, j, k, grid, clock, fields)
    φ = φnode(i, j, k, grid, Center(), Center(), Center())
    σ = σᶜᶜᶜ(i, j, k, grid, forcing)
    cᵖᵈ = forcing.thermodynamic_constants.dry_air.heat_capacity

    @inbounds begin
        p = forcing.pressure[i, j, k]
        ρᵈ = forcing.coupling_density[i, j, k]
        T = fields.T[i, j, k]
    end

    Tᵉ = equilibrium_temperature(forcing.parameters, φ, p)
    kᵀ = thermal_relaxation_rate(forcing.parameters, φ, σ)

    return - ρᵈ * cᵖᵈ * kᵀ * (T - Tᵉ)
end

#####
##### Materialization
#####

validate_latitude(::Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid}) = nothing
validate_latitude(grid::ImmersedBoundaryGrid) = validate_latitude(grid.underlying_grid)
validate_latitude(grid::MultiRegionGrid) = validate_latitude(first(grid))
validate_latitude(grid) = throw(ArgumentError("HeldSuarezForcing requires a grid with latitude, not a $(nameof(typeof(grid)))."))

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::HeldSuarezForcing, field, name,
                                                               model_field_names, context::NamedTuple)
    validate_latitude(field.grid)
    FT = eltype(field.grid)

    return HeldSuarezForcing(forcing.variable,
                             HeldSuarezParameters{FT}(forcing.parameters),
                             context.pressure,
                             context.total_density,
                             context.coupling_density,
                             context.thermodynamic_constants)
end

Base.summary(::HeldSuarezForcing{Val{variable}}) where variable = string("HeldSuarezForcing on ", variable)
Base.show(io::IO, forcing::HeldSuarezForcing) = print(io, summary(forcing))
