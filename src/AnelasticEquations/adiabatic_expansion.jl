#####
##### AdiabaticExpansion: the background of a Boussinesq box riding a moving frame
#####

"""
    ExpansionGradients

The uniform background vertical gradients ``Gᶿ = ∂θ̄/∂z`` and ``Gᵛ = ∂q̄ᵛ/∂z`` of an
[`AdiabaticExpansion`](@ref), refreshed from the moving frame before every tendency evaluation.
Also stores the supersaturation source per unit displacement implied by the gradients,
``𝒮_z = -(∂𝒮/∂θ Gᶿ + ∂𝒮/∂qᵛ Gᵛ)``, and the adiabatic-parcel coefficient ``A₁ = -∂𝒮/∂p ρ̄ g``
for diagnostics.
"""
mutable struct ExpansionGradients{FT}
    potential_temperature :: FT
    vapor_mass_fraction :: FT
    supersaturation_source :: FT
    A₁ :: FT
end

"""
    AdiabaticExpansion

The cooling and drying of air displaced vertically by ``w′`` within a [`BoussinesqDynamics`](@ref)
box riding a moving frame. See the constructor for the derivation.
"""
struct AdiabaticExpansion{V, G}
    variable :: V # nothing, or Val(:potential_temperature) or Val(:vapor_mass_fraction) for the forcings
    gradients :: G # nothing until materialized
end

"""
$(TYPEDSIGNATURES)

Return `AdiabaticExpansion`, which represents the expansion of air displaced vertically relative to
the moving frame of a [`BoussinesqDynamics`](@ref) through its `adiabatic_expansion` keyword.

The Boussinesq reference pressure is uniform in space, so without this term air moving with
velocity ``w′`` relative to the frame neither expands nor cools. Following the standard
treatment of stratified homogeneous turbulence, each conserved variable ``φ ∈ \\{θ, qᵛ\\}`` is
split into the frame mean ``φᶠ``, a background with uniform gradient ``Gᵠ``, and a fluctuation,

```math
φ = φᶠ(t) + Gᵠ (z - zᶠ) + φ″ .
```

Since ``D(z - zᶠ)/Dt = w′``, the background enters the equation for the fluctuation only through
the source ``-Gᵠ w′``, which is added to the tendencies of ``ρθ`` and ``ρqᵛ`` as ``-ρ Gᵠ w′``.
Thermodynamics is evaluated from ``φᶠ + φ″`` at the frame pressure, which drops the background and
the hydrostatic pressure deviation ``-ρ̄ g (z - zᶠ)``. That is consistent only if dropping them
leaves the supersaturation ``𝒮`` unchanged,

```math
∂𝒮/∂θ \\, Gᶿ + ∂𝒮/∂qᵛ \\, Gᵛ - ∂𝒮/∂p \\, ρ̄ g = 0 ,
```

so that every parcel's supersaturation responds to its displacement at the adiabatic-parcel rate
``A₁ = -∂𝒮/∂p \\, ρ̄ g``. Requiring also moist neutrality, ``cᵖᵐ Π Gᶿ = -ℒ Gᵛ``, makes the box a
sample of a saturated adiabatic cloud core whose background adds no stratification to buoyancy.
The two conditions determine ``Gᶿ > 0`` and ``Gᵛ < 0``, which are evaluated from the frame pressure,
temperature, and vapor before every tendency evaluation. In dry air both gradients vanish.

The term depends only on ``w′``, not on the frame velocity, so it is also active in a fixed frame
(`frame_vertical_velocity = 0`), which then represents a static saturated adiabatic column.

Examples
========

```jldoctest
julia> using Breeze

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic));

julia> dynamics = BoussinesqDynamics(grid; frame_vertical_velocity=2)
BoussinesqDynamics(pᵣ=101325.0, Tᵣ=288.0, ρᵣ=1.22585, frame_vertical_velocity=2.0, AdiabaticExpansion)
└── pressure_anomaly: not materialized

julia> dynamics.adiabatic_expansion
AdiabaticExpansion

julia> BoussinesqDynamics(grid; frame_vertical_velocity=2, adiabatic_expansion=nothing)
BoussinesqDynamics(pᵣ=101325.0, Tᵣ=288.0, ρᵣ=1.22585, frame_vertical_velocity=2.0)
└── pressure_anomaly: not materialized
```
"""
AdiabaticExpansion() = AdiabaticExpansion(nothing, nothing)

Base.summary(::AdiabaticExpansion) = "AdiabaticExpansion"

function Base.summary(expansion::AdiabaticExpansion{<:Any, <:ExpansionGradients})
    G = expansion.gradients
    return string("AdiabaticExpansion(Gᶿ=", prettysummary(G.potential_temperature),
                  ", Gᵛ=", prettysummary(G.vapor_mass_fraction), ")")
end

Base.show(io::IO, expansion::AdiabaticExpansion) = print(io, summary(expansion))

expansion_summary(::Nothing) = ""
expansion_summary(::AdiabaticExpansion) = ", AdiabaticExpansion"

#####
##### Construction, validation, and materialization
#####

# By default a moving frame carries its adiabatic expansion; a box without a frame does not.
default_adiabatic_expansion(::Nothing) = nothing
default_adiabatic_expansion(frame_vertical_velocity) = AdiabaticExpansion()

validate_adiabatic_expansion(::Nothing, frame) = nothing
validate_adiabatic_expansion(::AdiabaticExpansion, ::MovingFrame) = nothing

function validate_adiabatic_expansion(::AdiabaticExpansion, ::Nothing)
    msg = string("AdiabaticExpansion requires a moving frame, whose mean state defines the background. ",
                 "Set `frame_vertical_velocity` (to zero, for a fixed frame), or set `adiabatic_expansion = nothing`.")
    throw(ArgumentError(msg))
end

materialize_adiabatic_expansion(::Nothing, FT) = nothing
materialize_adiabatic_expansion(::AdiabaticExpansion, FT) =
    AdiabaticExpansion(nothing, ExpansionGradients(zero(FT), zero(FT), zero(FT), zero(FT)))

#####
##### Forcing on the thermodynamic and moisture variables
#####

AtmosphereModels.dynamics_forcing(dynamics::AnelasticDynamics, thermodynamic_name, moisture_name) =
    adiabatic_expansion_forcing(dynamics.adiabatic_expansion, thermodynamic_name, moisture_name)

adiabatic_expansion_forcing(::Nothing, thermodynamic_name, moisture_name) = NamedTuple()

function adiabatic_expansion_forcing(expansion::AdiabaticExpansion, thermodynamic_name, moisture_name)
    G = expansion.gradients
    forcings = (AdiabaticExpansion(Val(:potential_temperature), G), AdiabaticExpansion(Val(:vapor_mass_fraction), G))
    return NamedTuple{(thermodynamic_name, moisture_name)}(forcings)
end

function AtmosphereModels.materialize_atmosphere_model_forcing(::AdiabaticExpansion{Nothing}, field, name,
                                                               model_field_names, context)
    msg = string("AdiabaticExpansion was supplied as a forcing under `", name, "`, ",
                 "but it is a property of the dynamics: use `BoussinesqDynamics(grid; adiabatic_expansion)`.")
    throw(ArgumentError(msg))
end

AtmosphereModels.materialize_atmosphere_model_forcing(expansion::AdiabaticExpansion, field, name,
                                                      model_field_names, context) = expansion

"""
    ExpansionKernel

The kernel form of an [`AdiabaticExpansion`](@ref) forcing: the specific tendency ``-G w′``
for the uniform gradient `gradient`.
"""
struct ExpansionKernel{FT}
    gradient :: FT
end

@inline (f::ExpansionKernel)(i, j, k, grid, clock, fields) = - f.gradient * ℑzᵃᵃᶜ(i, j, k, grid, fields.w)

current_gradient(f::AdiabaticExpansion{Val{N}}) where N = getfield(f.gradients, N)

Adapt.adapt_structure(to, f::AdiabaticExpansion{<:Val}) = ExpansionKernel(current_gradient(f))

@inline (f::AdiabaticExpansion{<:Val})(i, j, k, grid, clock, fields) =
    - current_gradient(f) * ℑzᵃᵃᶜ(i, j, k, grid, fields.w)

function AtmosphereModels.compute_forcing!(forcing::AdiabaticExpansion, model)
    reference_state = model.dynamics.reference_state
    frame = reference_state.frame
    p = frame.pressure
    T = frame.temperature
    qᵛ = frame.vapor_mass_fraction
    g = frame.gravitational_acceleration
    pˢᵗ = reference_state.standard_pressure
    update_expansion_gradients!(forcing.gradients, p, T, qᵛ, g, pˢᵗ, model.thermodynamic_constants)
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

Compute the background gradients of an [`AdiabaticExpansion`](@ref) for a frame with pressure `p`,
temperature `T`, vapor mass fraction `qᵛ`, and gravitational acceleration `g`. The partial derivatives
of the supersaturation are evaluated by central differences in `Float64`.
"""
function update_expansion_gradients!(gradients::ExpansionGradients, p, T, qᵛ, g, pˢᵗ, constants)
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

    # Supersaturation consistency together with moist neutrality, cᵖᵐ Π Gᶿ = -ℒ Gᵛ
    Gᵛ = ∂𝒮∂p * ρ̄ * g / (∂𝒮∂q - ∂𝒮∂θ * ℒ / (cᵖᵐ * Π))
    Gᶿ = - ℒ * Gᵛ / (cᵖᵐ * Π)

    gradients.potential_temperature = Gᶿ
    gradients.vapor_mass_fraction = Gᵛ
    gradients.supersaturation_source = - (∂𝒮∂θ * Gᶿ + ∂𝒮∂q * Gᵛ)
    gradients.A₁ = - ∂𝒮∂p * ρ̄ * g
    return gradients
end
