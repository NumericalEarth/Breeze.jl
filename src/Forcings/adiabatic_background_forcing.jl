using Oceananigans: Oceananigans
using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.Thermodynamics: MoistureMassFractions, PlanarLiquidSurface, potential_temperature_from_temperature,
                             supersaturation, mixture_gas_constant, mixture_heat_capacity, liquid_latent_heat,
                             reference_value
using Oceananigans.Operators: ℑzᵃᵃᶜ
using Adapt: Adapt

#####
##### Background vertical gradients for vertically periodic (rising-frame) boxes
#####

"""
    BackgroundGradients

The uniform background vertical gradients ``Gᶿ = ∂θ̄/∂z`` and ``Gᵛ = ∂q̄ᵛ/∂z`` of an
[`AdiabaticBackgroundForcing`](@ref), refreshed from the reference (frame) state before every
tendency evaluation. Also stores the supersaturation source per unit displacement implied by
the gradients, ``𝒮_z = -(∂𝒮/∂θ Gᶿ + ∂𝒮/∂qᵛ Gᵛ)``, and the adiabatic-parcel coefficient
``A₁ = -∂𝒮/∂p ρ̄ g`` for diagnostics.
"""
mutable struct BackgroundGradients{FT}
    potential_temperature :: FT
    vapor_mass_fraction :: FT
    supersaturation_source :: FT
    A₁ :: FT
end

struct MoistAdiabaticBackground end
struct IsothermalBackground end

background_kind(kind::Union{MoistAdiabaticBackground, IsothermalBackground}) = kind
background_kind(kind::Symbol) = kind === :moist ? MoistAdiabaticBackground() :
                                kind === :isothermal ? IsothermalBackground() :
                                throw(ArgumentError("background must be :moist or :isothermal, got :$kind"))

"""
    AdiabaticBackgroundForcing

Forcing that represents a uniform background vertical gradient of potential temperature and
vapor in a vertically periodic box that rises with a [`BoussinesqReferenceState`](@ref)
`MovingFrame`. See the constructor for the derivation.
"""
struct AdiabaticBackgroundForcing{K, V, G, FT}
    kind :: K
    variable :: V # nothing before materialization; Val(:θ) or Val(:qᵛ) after
    gradients :: G
    gravitational_acceleration :: FT
end

"""
$(TYPEDSIGNATURES)

Return a forcing that represents vertical displacement within a vertically periodic box,
such as a box riding a rising [`BoussinesqReferenceState`](@ref) frame, by a uniform background gradient
of each conserved variable. Supply the same object under both `θ` and `qᵛ`:
`forcing = (θ=background, qᵛ=background)`.

A uniform reference pressure means that, without this forcing, air moving with velocity `w′`
relative to the frame neither expands nor cools. Following the standard treatment of stratified
homogeneous turbulence, each conserved variable ``φ ∈ \\{θ, qᵛ\\}`` is split into the frame mean,
a background with uniform gradient ``Gᵠ``, and a periodic fluctuation,

```math
φ = φᶠ(t) + Gᵠ (z - zᶠ) + φ″ ,
```

Since ``D(z - zᶠ)/Dt = w′``, the background enters the prognostic equation for the fluctuation
only through the forcing ``-Gᵠ w′``, which has zero domain mean. Thermodynamics is evaluated
from ``φᶠ + φ″`` at the frame pressure, which drops the background and the hydrostatic pressure
deviation ``-ρ̄ g (z - zᶠ)``. That is consistent only if dropping them leaves the supersaturation
unchanged,

```math
∂𝒮/∂θ \\, Gᶿ + ∂𝒮/∂qᵛ \\, Gᵛ - ∂𝒮/∂p \\, ρ̄ g = 0 ,
```

in which case every parcel's supersaturation responds to its displacement at the adiabatic-parcel
rate ``A₁ = -∂𝒮/∂p ρ̄ g``. The `background` options are

- `:moist` (default): the condition above together with moist neutrality, ``cᵖ Π Gᶿ = -ℒ Gᵛ``,
  so that the box samples a saturated adiabatic cloud core and the background adds no
  stratification to buoyancy.
- `:isothermal`: ``Gᶿ = g / (cᵖ Π)`` and ``Gᵛ = 0``, the dry-adiabatic cooling ``-(g/cᵖ) w′`` of
  temperature alone. It neglects the pressure part of ``A₁`` and overestimates the supersaturation
  source by about 23% near 280 K.

The gradients are evaluated from the reference (frame) pressure, temperature, and vapor
before every tendency evaluation. `gravitational_acceleration` defaults to the model's.

This forcing is opt-in: models without it have no background gradient.

Examples
========

```jldoctest
julia> using Breeze

julia> background = AdiabaticBackgroundForcing()
AdiabaticBackgroundForcing(MoistAdiabaticBackground)
```
"""
function AdiabaticBackgroundForcing(FT=Oceananigans.defaults.FloatType; background = :moist,
                                    gravitational_acceleration = nothing)
    gradients = BackgroundGradients(zero(FT), zero(FT), zero(FT), zero(FT))
    g = isnothing(gravitational_acceleration) ? convert(FT, NaN) : convert(FT, gravitational_acceleration)
    return AdiabaticBackgroundForcing(background_kind(background), nothing, gradients, g)
end

Base.summary(f::AdiabaticBackgroundForcing) = string("AdiabaticBackgroundForcing(", nameof(typeof(f.kind)), ")")
Base.show(io::IO, f::AdiabaticBackgroundForcing) = print(io, summary(f))

"""
    BackgroundGradientKernel

The kernel form of an [`AdiabaticBackgroundForcing`](@ref): the specific tendency ``-G w′``
for the uniform gradient `gradient`.
"""
struct BackgroundGradientKernel{FT}
    gradient :: FT
end

@inline (f::BackgroundGradientKernel)(i, j, k, grid, clock, fields) = - f.gradient * ℑzᵃᵃᶜ(i, j, k, grid, fields.w)

current_gradient(f::AdiabaticBackgroundForcing{<:Any, Val{:θ}}) = f.gradients.potential_temperature
current_gradient(f::AdiabaticBackgroundForcing{<:Any, Val{:qᵛ}}) = f.gradients.vapor_mass_fraction

Adapt.adapt_structure(to, f::AdiabaticBackgroundForcing) = BackgroundGradientKernel(current_gradient(f))

@inline (f::AdiabaticBackgroundForcing)(i, j, k, grid, clock, fields) =
    - current_gradient(f) * ℑzᵃᵃᶜ(i, j, k, grid, fields.w)

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::AdiabaticBackgroundForcing,
                                                               field, name, model_field_names,
                                                               context::NamedTuple)
    if name ∉ (:θ, :qᵛ)
        throw(ArgumentError(string("AdiabaticBackgroundForcing must be supplied under `θ` and `qᵛ`, ",
                                   "but was supplied under `", name, "`.")))
    end
    return AdiabaticBackgroundForcing(forcing.kind, Val(name), forcing.gradients,
                                      forcing.gravitational_acceleration)
end

function AtmosphereModels.compute_forcing!(forcing::AdiabaticBackgroundForcing, model)
    reference_state = model.dynamics.reference_state
    p = reference_value(reference_state.pressure)
    T = reference_value(reference_state.temperature)
    qᵛ = reference_value(reference_state.vapor_mass_fraction)
    pˢᵗ = reference_state.standard_pressure
    constants = model.thermodynamic_constants
    g = isnan(forcing.gravitational_acceleration) ? constants.gravitational_acceleration :
                                                     forcing.gravitational_acceleration
    update_background_gradients!(forcing.gradients, forcing.kind, p, T, qᵛ, g, pˢᵗ, constants)
    return nothing
end

# Exner function Π = T / θ of air with vapor mass fraction qᵛ
exner_function(p, qᵛ, pˢᵗ, constants) =
    1 / potential_temperature_from_temperature(1, p, pˢᵗ, constants, MoistureMassFractions(qᵛ))

function frame_supersaturation(θ, qᵛ, p, pˢᵗ, constants)
    T = θ * exner_function(p, qᵛ, pˢᵗ, constants)
    q = MoistureMassFractions(qᵛ)
    ρ = p / (mixture_gas_constant(q, constants) * T)
    return supersaturation(T, ρ, q, constants, PlanarLiquidSurface())
end

"""
$(TYPEDSIGNATURES)

Compute the background gradients for the reference state with pressure `p`, temperature `T`,
and vapor mass fraction `qᵛ`. The partial derivatives of the supersaturation are evaluated
by central differences in `Float64`.
"""
function update_background_gradients!(gradients::BackgroundGradients{FT}, kind, p, T, qᵛ, g, pˢᵗ, constants) where FT
    p, T, qᵛ, g, pˢᵗ = Float64(p), Float64(T), Float64(qᵛ), Float64(g), Float64(pˢᵗ)
    q = MoistureMassFractions(qᵛ)
    Π = exner_function(p, qᵛ, pˢᵗ, constants)
    θ = T / Π
    ρ̄ = p / (mixture_gas_constant(q, constants) * T)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ℒ = liquid_latent_heat(T, constants)

    𝒮(θ, qᵛ, p) = frame_supersaturation(θ, qᵛ, p, pˢᵗ, constants)
    δθ = 1e-4 * θ
    δq = 1e-5 * max(qᵛ, 1e-6)
    δp = 1e-5 * p
    ∂𝒮∂θ = (𝒮(θ + δθ, qᵛ, p) - 𝒮(θ - δθ, qᵛ, p)) / 2δθ
    ∂𝒮∂q = (𝒮(θ, qᵛ + δq, p) - 𝒮(θ, qᵛ - δq, p)) / 2δq
    ∂𝒮∂p = (𝒮(θ, qᵛ, p + δp) - 𝒮(θ, qᵛ, p - δp)) / 2δp

    Gᶿ, Gᵛ = background_gradients(kind, ∂𝒮∂θ, ∂𝒮∂q, ∂𝒮∂p, ρ̄, g, cᵖᵐ, Π, ℒ)

    gradients.potential_temperature = Gᶿ
    gradients.vapor_mass_fraction = Gᵛ
    gradients.supersaturation_source = - (∂𝒮∂θ * Gᶿ + ∂𝒮∂q * Gᵛ)
    gradients.A₁ = - ∂𝒮∂p * ρ̄ * g
    return gradients
end

background_gradients(::IsothermalBackground, ∂𝒮∂θ, ∂𝒮∂q, ∂𝒮∂p, ρ̄, g, cᵖᵐ, Π, ℒ) = (g / (cᵖᵐ * Π), zero(g))

function background_gradients(::MoistAdiabaticBackground, ∂𝒮∂θ, ∂𝒮∂q, ∂𝒮∂p, ρ̄, g, cᵖᵐ, Π, ℒ)
    Gᵛ = ∂𝒮∂p * ρ̄ * g / (∂𝒮∂q - ∂𝒮∂θ * ℒ / (cᵖᵐ * Π))
    Gᶿ = - ℒ * Gᵛ / (cᵖᵐ * Π)
    return Gᶿ, Gᵛ
end
