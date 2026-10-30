# NCAR-style "isothermal" emulation of adiabatic displacement in a rising Boussinesq frame.
#
# The NCAR DNS cools air displaced by w′ relative to the frame at the dry-adiabatic rate,
# ∂T′/∂t = - (g / cᵖ) w′, with no vapor gradient. In terms of potential temperature this is the
# background gradient Gᶿ = g / (cᵖᵐ Π), Gᵛ = 0. It neglects the pressure part of the adiabatic
# supersaturation source A₁ w′ and overestimates it by about 23% near 280 K, which is why Breeze's
# built-in `AdiabaticExpansion` uses the moist-adiabatic background instead. This file reproduces
# the NCAR term as a user forcing for comparisons:
#
#     include("isothermal_forcing.jl")
#     dynamics = BoussinesqDynamics(grid; frame_vertical_velocity=W, adiabatic_expansion=nothing)
#     model = AtmosphereModel(grid; dynamics, forcing=(; θ=IsothermalExpansion()), ...)
#
# The gradient is refreshed from the frame before every tendency evaluation through
# `Breeze.AtmosphereModels.compute_forcing!`.

using Adapt: Adapt
using Breeze
using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.Thermodynamics: MoistureMassFractions, mixture_heat_capacity, potential_temperature_from_temperature
using Oceananigans.Operators: ℑzᵃᵃᶜ

mutable struct IsothermalExpansion{FT}
    potential_temperature_gradient :: FT
end

IsothermalExpansion() = IsothermalExpansion(0.0)

struct IsothermalExpansionKernel{FT}
    gradient :: FT
end

Adapt.adapt_structure(to, f::IsothermalExpansion) = IsothermalExpansionKernel(f.potential_temperature_gradient)

@inline (f::IsothermalExpansionKernel)(i, j, k, grid, clock, fields) = - f.gradient * ℑzᵃᵃᶜ(i, j, k, grid, fields.w)
@inline (f::IsothermalExpansion)(i, j, k, grid, clock, fields) =
    - f.potential_temperature_gradient * ℑzᵃᵃᶜ(i, j, k, grid, fields.w)

function AtmosphereModels.compute_forcing!(f::IsothermalExpansion, model)
    reference_state = model.dynamics.reference_state
    frame = reference_state.frame
    constants = model.thermodynamic_constants
    q = MoistureMassFractions(frame.vapor_mass_fraction)
    Π = 1 / potential_temperature_from_temperature(1, frame.pressure, reference_state.standard_pressure, constants, q)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    f.potential_temperature_gradient = frame.gravitational_acceleration / (cᵖᵐ * Π)
    return nothing
end
