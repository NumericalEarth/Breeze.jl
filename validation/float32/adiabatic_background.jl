# Adiabatic displacement within a rising Boussinesq frame, as a mean-gradient ("background") forcing.
#
# The rising frame moves at W; the air inside it moves at W + w′. The frame's uniform pressure pᶠ(t) gives
# the mean cooling of the ascent W exactly. Vertical motion w′ *within* the box needs a mean vertical gradient,
# which a triply periodic box cannot hold as a field. The standard remedy (homogeneous stratified turbulence)
# splits every conserved variable φ ∈ {θ, qᵛ} into
#
#     φ(x, t) = φᶠ(t) + Gᵠ (z - zᶠ) + φ″(x, t),
#
# a frame mean, a linear background with a uniform gradient Gᵠ = ∂φ̄/∂z, and a periodic fluctuation φ″.
# The background is not stored; inserting the split into Dφ/Dt = sources gives, because D(z - zᶠ)/Dt = w′,
#
#     Dφ″/Dt = - Gᵠ w′ + (sources - their mean),
#
# so only the periodic part is prognostic and the background acts through the forcing - Gᵠ w′.
# Thermodynamic functions are evaluated from φᶠ + φ″ at the frame pressure; the background terms
# (Gᵠ (z - zᶠ) and the hydrostatic pressure deviation - ρ̄ g (z - zᶠ)) are dropped there. That is consistent
# only if the dropped terms do not change the supersaturation, i.e. if the background has
#
#     ∂𝒮/∂θ Gᶿ + ∂𝒮/∂qᵛ Gᵛ - ∂𝒮/∂p ρ̄ g = 0.                                               (1)
#
# Then every parcel obeys D𝒮/Dt = A₁ (W + w′) + condensation, with A₁ = - ∂𝒮/∂p ρ̄ g (the usual
# A₁ = g / (Rᵈ T) [ℒ Rᵈ / (cᵖ Rᵛ T) - 1] for the pᵛ / pᵛ⁺ supersaturation, both its temperature and its
# pressure part), whatever the box size and without any non-periodic field.
#
# Backgrounds (`background` keyword):
# - `:none`        Gᶿ = Gᵛ = 0: no source; the current Breeze behaviour.
# - `:isothermal`  Gᶿ = g / (cᵖ Π), Gᵛ = 0: the intended NCAR term ∂T′/∂t = - (g / cᵖ) w′ (`gammand`). Uniform T
#                  and p in the box: the 𝒮 source is the temperature part of A₁ only, too strong by
#                  1 / [ℒ Rᵈ / (cᵖ Rᵛ T) - 1] ≈ 23% at 281 K. (In NCAR's code the term is in fact
#                  - (g / cᵖ) (w T′)^ because `wk` is overwritten in `convol`, i.e. effectively absent.)
# - `:moist`       (1) together with moist neutrality Gᶿ = - ℒ Gᵛ / (cᵖᵐ Π): the box is a sample of a saturated
#                  adiabatic cloud core, 𝒮̄ is uniform in the vertical and buoyancy (if on) sees no
#                  background stratification.
#
# The gradients follow the frame state (pᶠ, T̄, q̄ᵛ) and are refreshed by a callback.

using Oceananigans.Architectures: Adapt
using Oceananigans.Operators: ℑzᵃᵃᶜ
using Breeze.Thermodynamics: potential_temperature_from_temperature, supersaturation, mixture_heat_capacity,
                             liquid_latent_heat

mutable struct AdiabaticBackground{FT}
    kind :: Symbol
    potential_temperature_gradient :: FT # Gᶿ
    vapor_gradient :: FT                 # Gᵛ
    supersaturation_source :: FT         # 𝒮 tendency per unit w′ implied by (Gᶿ, Gᵛ) at the frame state
    A₁ :: FT                             # exact A₁ = - ∂𝒮/∂p ρ̄ g at the frame state
end

Adapt.adapt_structure(to, b::AdiabaticBackground) = (Gᶿ = b.potential_temperature_gradient, Gᵛ = b.vapor_gradient)

AdiabaticBackground(kind) = AdiabaticBackground(kind, 0.0, 0.0, 0.0, 0.0)

# Exner function of vapor-only air: θ = T / Π(p, qᵛ)
exner(p, qᵛ, pˢᵗ, constants) = 1 / potential_temperature_from_temperature(1.0, p, pˢᵗ, constants, qᵛ)

function frame_supersaturation(θ, qᵛ, p, pˢᵗ, constants)
    T = θ * exner(p, qᵛ, pˢᵗ, constants)
    q = MoistureMassFractions(qᵛ)
    ρ = p / (mixture_gas_constant(q, constants) * T)
    return supersaturation(T, ρ, q, constants, PlanarLiquidSurface())
end

"""
    update_background!(b, p, T, qᵛ, g, pˢᵗ, constants)

Compute the background gradients for the frame state `(p, T, qᵛ)`; `g` is the gravitational
acceleration of the frame (the model's own may be zero to switch buoyancy off).
"""
function update_background!(b::AdiabaticBackground, p, T, qᵛ, g, pˢᵗ, constants)
    q = MoistureMassFractions(qᵛ)
    Π = exner(p, qᵛ, pˢᵗ, constants)
    θ = T / Π
    ρ̄ = p / (mixture_gas_constant(q, constants) * T)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ℒ = liquid_latent_heat(T, constants)
    𝒮(θ, qᵛ, p) = frame_supersaturation(θ, qᵛ, p, pˢᵗ, constants)

    # Central differences at fixed (θ, qᵛ, p)
    δθ, δq, δp = 1e-4θ, 1e-5qᵛ, 1e-5p
    𝒮θ = (𝒮(θ + δθ, qᵛ, p) - 𝒮(θ - δθ, qᵛ, p)) / 2δθ
    𝒮q = (𝒮(θ, qᵛ + δq, p) - 𝒮(θ, qᵛ - δq, p)) / 2δq
    𝒮p = (𝒮(θ, qᵛ, p + δp) - 𝒮(θ, qᵛ, p - δp)) / 2δp

    A₁ = - 𝒮p * ρ̄ * g

    if b.kind === :none
        Gᶿ, Gᵛ = 0.0, 0.0
    elseif b.kind === :isothermal
        Gᶿ, Gᵛ = g / (cᵖᵐ * Π), 0.0
    elseif b.kind === :moist
        Gᵛ = 𝒮p * ρ̄ * g / (𝒮q - 𝒮θ * ℒ / (cᵖᵐ * Π))
        Gᶿ = - ℒ * Gᵛ / (cᵖᵐ * Π)
    else
        throw(ArgumentError("unknown adiabatic background $(b.kind); use :none, :isothermal or :moist"))
    end

    b.potential_temperature_gradient = Gᶿ
    b.vapor_gradient = Gᵛ
    b.supersaturation_source = - (𝒮θ * Gᶿ + 𝒮q * Gᵛ)
    b.A₁ = A₁
    return b
end

function update_background!(b::AdiabaticBackground, model, g, constants)
    frame = model.dynamics.reference_state.frame
    pˢᵗ = model.dynamics.reference_state.standard_pressure
    return update_background!(b, frame.pressure, frame.temperature, frame.vapor_mass_fraction, g, pˢᵗ, constants)
end

# ∂(ρθ)/∂t = - ρ Gᶿ w′ and ∂(ρqᵛ)/∂t = - ρ Gᵛ w′, with w′ interpolated to cell centers
@inline background_θ_forcing(i, j, k, grid, clock, fields, p) = - p.ρ * p.background.Gᶿ * ℑzᵃᵃᶜ(i, j, k, grid, fields.w)
@inline background_qᵛ_forcing(i, j, k, grid, clock, fields, p) = - p.ρ * p.background.Gᵛ * ℑzᵃᵃᶜ(i, j, k, grid, fields.w)

background_forcings(b::AdiabaticBackground, ρ) =
    (ρθ = Forcing(background_θ_forcing, discrete_form=true, parameters=(; ρ, background=b)),
     ρqᵛ = Forcing(background_qᵛ_forcing, discrete_form=true, parameters=(; ρ, background=b)))

background_kind(s) = s ∈ ("0", "none") ? :none : s ∈ ("1", "isothermal") ? :isothermal :
                     s == "moist" ? :moist : throw(ArgumentError("LAPSE must be 0/none, 1/isothermal or moist"))
