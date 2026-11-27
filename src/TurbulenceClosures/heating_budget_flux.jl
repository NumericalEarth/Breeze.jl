#####
##### The budget flux of diabatic heating: a turbulent flux of θˡⁱ that carries a heating from where
##### it acts to where the turbulence is, and its energy conversion in the TKE equation
#####
##### A component of `TKEBasedTurbulenceClosure` (`heating_budget_flux = HeatingBudgetFlux(...)`).
#####
##### Radiation cools the top few tens of meters of a stratocumulus layer. The air it cools sinks
##### through the whole layer: the cooling generates available potential energy against the mixed
##### state, and the turbulence converts it. A down-gradient flux can carry that cooling only after
##### the mean profile has become unstable, and cannot carry it across a stable cloud base at all.
##### Here the heating 𝒬 (the tendency of θˡⁱ, K s⁻¹) is redistributed by the same construction as
##### the sink of `TKEBudgetThirdMoment`: the heating as the turbulence redistributes it is s = c ω,
##### with the shape ω = e (`turbulent_shape`: e₊² / (e₊ + eᵐⁱⁿ), none in still air) and an amplitude c exchanged between neighbouring levels over the length
##### Λ = ℂᑫᴸ ℓ, and the flux that does the redistributing is J,
#####
#####     (ρJ)ₖ₊½ - (ρJ)ₖ₋½ = ρₖ Δzₖ (𝒬ₖ - sₖ),    (ρJ)½ = 0,
#####     [mₖ + a₊θ₊ + a₋θ₋] sₖ - a₊(1 - θ₊) sₖ₊₁ - a₋(1 - θ₋) sₖ₋₁ = mₖ 𝒬ₖ,
#####     m = ρ Δz,    a± = 2 ρ± Λ±² / Δz±,    θ± = ωₖ±₁ / (ωₖ + ωₖ±₁).
#####
##### The matrix is the M-matrix of `exchange_amplitudes!`; its columns sum to m, so Σ m s = Σ m 𝒬 and J
##### vanishes at the top of the column too. Where the air is still (ω = 0 at a level and its
##### neighbours) the levels do not exchange, s = 𝒬 and J = 0; a heating that is uniform over a
##### turbulent layer gives J = 0; cooling at the top of a turbulent layer, or heating at its bottom,
##### gives J > 0, and the opposite J < 0.
#####
##### The closure applies the share ℂᑫ of the positive part to the mean state,
#####
#####     Jᑫ = ℂᑫ J₊,    J₊ = (J + √(J² + δ²)) / 2,    δ = ℓ ε / (ĉᵖ Π),    ∂t (ρ θˡⁱ) = ... - ∂z (ρ Jᑫ),
#####
##### (a heating that stabilizes stays where it acts, and the gradient closure sees it). The positive part is smooth,
##### with the scale δ of a flux of θˡⁱ at the face: the mixing length times the rate ε / (ĉᵖ Π) at which the
##### dissipation of the turbulence heats the air there, that is 𝕊ᴰ √e times the temperature e / (ĉᵖ Π) that the TKE
##### is worth. It vanishes with the turbulence, so the flux still vanishes in still air, and it is small beside the
##### flux of a radiative cooling (ℓ = 100 m and ε = 10⁻³ m² s⁻³ give 10⁻⁴ K m s⁻¹), so J₊ is the positive part
##### but within that scale of J = 0. A scale built from the gradient of θˡⁱ (√e ℓ |∂z θˡⁱ|, or Kᶜ |∂z θˡⁱ|) would not
##### do: where J = 0 the smooth part is δ / 2, which would add half the magnitude of the flux down the gradient to
##### every stratified turbulent face, heated or not. The stored Jᑫ is both what the mean state receives and what is
##### converted, so the energy closes whatever the scale. The closure gives its
##### energy conversion to the TKE equation, whose production is then P + B + G - ε:
#####
#####     G = b_θ Jᑫ - (b̄ᶜ - b̄ᵛ) aᴺ c_θ Jᑫ - (b̄ⁱ - b̄ᵛ) ηⁱ c_θⁱ Jᑫ    at the faces, moved to the centers with the half-cell weights.
#####
##### The first term is the dynamic enthalpy the flux takes from the mean state, exactly (the dynamic
##### enthalpy is linear in θˡⁱ at the stage's b_θ, as for the fluxes down the gradient). The second is
##### the latent part: the flux of θˡⁱ is a flux -c_θ Jᑫ of the saturation excess, of which the share
##### aᴺ condenses (`_latent_corrected_stability!`); the closure applies no condensate flux with Jᑫ, so
##### the buoyancy conversion of that condensate flux is drawn from the internal energy, as Bᴸ is.
##### Integrated over a layer with one amplitude, ∫ ρ b_θ J dz = -∫ ρ b_θ (z - z̄) 𝒬 dz: the generation
##### of available potential energy against the mixed state.
#####
##### The distribution receives the total flux of θˡⁱ (w′θˡⁱ′ of the moments includes Jᑫ), and the
##### third moment of `TKEBudgetThirdMoment` receives G through P + B. Its damped local member is
##### signed by origin: Gᑫᶜ, the conversion of the flux that the cooling part 𝒬₋ alone would
##### drive (with ℂᑫ = 1), is carried by downdrafts, so
##### the buoyant source of the local member is ℂʷᵇ (B₊ - Gᶜ) - ℂʷᑫ Gᶜ, Gᶜ = min(Gᑫᶜ, B₊) (a smooth minimum, with the
##### scale ε). The cooling part is the smooth negative part 𝒬₋ = (𝒬 - √(𝒬² + (ε / (ĉᵖ Π))²)) / 2, with the scale of a
##### heating rate there: the rate at which the dissipation heats the air. In still air that scale is zero and the
##### part is min(𝒬, 0), which need not vanish there (the levels of still air do not exchange, so it drives no flux).
##### The redistributed cooling and the integral of the flux read the same 𝒬₋, so its flux vanishes at the top too.
##### With ℂᑫ = 0 nothing is applied and J is a diagnostic that only signs the local member.
#####

using Oceananigans.Fields: AbstractField, interior

"""
$(TYPEDEF)

The source of the heating of a [`HeatingBudgetFlux`](@ref) that is the radiative heating of the
model: the flux divergence of `model.radiation`, as a tendency of θˡⁱ. Without radiation it is zero.
"""
struct RadiativeHeating end

"""
$(TYPEDEF)

A source of the heating of a [`HeatingBudgetFlux`](@ref) that is prescribed: `tendency` is a field
at the cell centers (or an array that broadcasts with the interior of one) that holds the tendency of θˡⁱ in
K s⁻¹, for example the field of a `Forcing` of θ that a callback updates with a parameterized
radiative cooling. The closure reads it at every evaluation of its fields and does not change it.

$(TYPEDFIELDS)
"""
struct PrescribedHeating{F}
    "Tendency of θˡⁱ at the cell centers, K s⁻¹"
    tendency :: F
end

Adapt.adapt_structure(to, heating::PrescribedHeating) = PrescribedHeating(adapt(to, heating.tendency))

"""
$(TYPEDEF)

A turbulent flux of θˡⁱ that redistributes diabatic heating over the turbulent layer, and its
energy conversion in the TKE equation of [`TKEBasedTurbulenceClosure`](@ref),

    (ρJ)ₖ₊½ - (ρJ)ₖ₋½ = ρₖ Δzₖ (𝒬ₖ - sₖ),    Jᑫ = ℂᑫ (J + √(J² + δ²)) / 2,    G = (b_θ - (bˡ - bᵛ) aᴺ c_θ) Jᑫ,

where `𝒬` is the heating (the tendency of θˡⁱ of the sources in `heating`) and `s` the heating as
the turbulence redistributes it: `s = c e` with an amplitude `c` exchanged between neighbouring
levels over the length `ℂᑫᴸ ℓ` (one tridiagonal solve per column). `J` vanishes at the bottom and
the top of the column and in still air; its positive part is smooth, with the scale `δ = ℓ ε / (ĉᵖ Π)` of the
dissipation `ε` of the turbulence at the face. The closure applies `-∂z (ρ Jᑫ)` to `ρθ`, and the TKE
equation receives `ρ G`, the energy the flux takes from the mean state: the production is
`P + B + G - ε`.

`heating` is one source or a tuple of sources: [`RadiativeHeating`](@ref) (the default),
[`PrescribedHeating`](@ref), or any object `source` with a method

    Breeze.TurbulenceClosures.add_diabatic_heating!(𝒬, source, model, fields)

that adds its tendency of θˡⁱ (K s⁻¹) to the center field `𝒬`.

$(TYPEDFIELDS)
"""
struct HeatingBudgetFlux{H, FT}
    "Share of the destabilizing flux that is applied to θˡⁱ and converted, in [0, 1]"
    ℂᑫ :: FT
    "Exchange length of the redistributed heating in mixing lengths, `Λ = ℂᑫᴸ ℓ`; positive and finite"
    ℂᑫᴸ :: FT
    "Coefficient of the conversion of the flux driven by cooling in the damped local third moment of `TKEBudgetThirdMoment`; not negative"
    ℂʷᑫ :: FT
    "Sources of the heating"
    heating :: H
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`HeatingBudgetFlux`](@ref). By default half of the destabilizing flux is applied (`ℂᑫ = 0.5`),
the heating is exchanged over eight mixing lengths, the local third moment is not changed (`ℂʷᑫ = 0`),
and the heating is the radiative heating of the model.

The initial values are not calibrated. In a stratocumulus layer the entrainment flux of θˡⁱ nearly cancels
the flux that the radiative cooling drives, and the gradient closure carries the entrainment flux: with
`ℂᑫ = 1` the single column of DYCOMS-II RF01 becomes stably stratified in θˡⁱ by 0.4 K over the layer
(0.12 K without the flux, 0.27 K with `ℂᑫ = 0.5`). `ℂʷᑫ > 0` gives the cloud layer of a stratocumulus
column the negative skewness of its large-eddy simulation, and gives the same sign to cumulus layers
below a radiatively cooled anvil, where the large-eddy simulations have positive skewness.

[`heating_budget_flux_bounds`](@ref) gives an interval for each coefficient for a calibration.
"""
function HeatingBudgetFlux(FT::DataType = Oceananigans.defaults.FloatType;
                           ℂᑫ = 0.5, ℂᑫᴸ = 8, ℂʷᑫ = 0, heating = RadiativeHeating())
    0 <= ℂᑫ <= 1 || throw(ArgumentError("ℂᑫ must lie in [0, 1], got $(ℂᑫ)."))
    0 < ℂᑫᴸ < Inf || throw(ArgumentError("ℂᑫᴸ must be positive and finite (the levels of still air must not exchange), got $(ℂᑫᴸ)."))
    ℂʷᑫ >= 0 || throw(ArgumentError("ℂʷᑫ must not be negative, got $(ℂʷᑫ)."))
    return HeatingBudgetFlux(convert(FT, ℂᑫ), convert(FT, ℂᑫᴸ), convert(FT, ℂʷᑫ), heating)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The interval of each coefficient of a [`HeatingBudgetFlux`](@ref) for a calibration, as a
`NamedTuple` of `(lower, upper)`: the share `ℂᑫ` in `[0, 1]`, the exchange length `ℂᑫᴸ` in
`[1, 32]` mixing lengths and the local third-moment coefficient `ℂʷᑫ` in `[0, 2]`.
"""
heating_budget_flux_bounds() = (ℂᑫ = (0.0, 1.0), ℂᑫᴸ = (1.0, 32.0), ℂʷᑫ = (0.0, 2.0))

convert_eltype(::Type{FT}, hb::HeatingBudgetFlux) where FT =
    HeatingBudgetFlux(convert(FT, hb.ℂᑫ), convert(FT, hb.ℂᑫᴸ), convert(FT, hb.ℂʷᑫ), hb.heating)

Adapt.adapt_structure(to, hb::HeatingBudgetFlux) =
    HeatingBudgetFlux(adapt(to, hb.ℂᑫ), adapt(to, hb.ℂᑫᴸ), adapt(to, hb.ℂʷᑫ), adapt(to, hb.heating))

Base.summary(hb::HeatingBudgetFlux) =
    string("HeatingBudgetFlux(ℂᑫ = ", prettysummary(hb.ℂᑫ), ", ℂᑫᴸ = ", prettysummary(hb.ℂᑫᴸ), ", ℂʷᑫ = ", prettysummary(hb.ℂʷᑫ),
           ", heating = ", heating_summary(hb.heating), ")")

heating_summary(heating) = string(nameof(typeof(heating)))
heating_summary(heating::Tuple) = string("(", join(map(heating_summary, heating), ", "), ")")

Base.show(io::IO, hb::HeatingBudgetFlux) = print(io, summary(hb))

# The heating budget flux of a closure or of an ensemble of closures (which share its sources)
heating_budget_flux(closure::TKEBasedTurbulenceClosure) = closure.heating_budget_flux
heating_budget_flux(closure::TKEClosureArray) = first(closure).heating_budget_flux

# It needs the stored derivatives of the buoyancy and the stage hooks of the applied-flux energetics
validate_heating_budget_flux(::Nothing, static_stability, distribution) = nothing

function validate_heating_budget_flux(hb::HeatingBudgetFlux, static_stability, distribution)
    isnothing(distribution) && !(static_stability isa SaturationWeightedStaticStability) &&
        throw(ArgumentError("HeatingBudgetFlux needs the energetics of the applied fluxes: a subgrid_distribution, or " *
                            "static_stability = SaturationWeightedStaticStability(...)."))
    return nothing
end

# The coefficients are the scalars of the component; the sources are kept
@inline SimplifiedHigherOrderClosures.component_scalars(hb::HeatingBudgetFlux) = (hb.ℂᑫ, hb.ℂᑫᴸ, hb.ℂʷᑫ)
SimplifiedHigherOrderClosures.component_scalar_names(::HeatingBudgetFlux, name) = (:ℂᑫ, :ℂᑫᴸ, :ℂʷᑫ)
@inline SimplifiedHigherOrderClosures.rebuild_component(hb::HeatingBudgetFlux, scalars, n) =
    HeatingBudgetFlux(promote(scalars[n], scalars[n+1], scalars[n+2])..., hb.heating), n + 3
@inline SimplifiedHigherOrderClosures.map_scalars(f, hb::HeatingBudgetFlux) =
    HeatingBudgetFlux(f(hb.ℂᑫ), f(hb.ℂᑫᴸ), f(hb.ℂʷᑫ), hb.heating)

#####
##### Fields: the heating, the heating as redistributed, the applied flux and its conversions, the flux
##### and conversion that the cooling alone would drive, the rate at which the dissipation heats the air (the scale of
##### the smooth parts), and a work array
#####

heating_budget_flux_fields(grid, ::Nothing) = NamedTuple()

heating_budget_flux_fields(grid, ::HeatingBudgetFlux) =
    (𝒬 = CenterField(grid), sᑫ = CenterField(grid), sᑫᶜ = CenterField(grid), heating_work = CenterField(grid),
     dissipative_heating_rate = CenterField(grid),
     Jᑫ = ZFaceField(grid), Jᑫᶜ = ZFaceField(grid),
     Gᑫ = CenterField(grid), Gᑫᴸ = CenterField(grid), Gᑫᶜ = CenterField(grid))

#####
##### The heating: the tendency of θˡⁱ of each source, added to 𝒬
#####

"""
    add_diabatic_heating!(𝒬, source, model, fields)

Add the tendency of θˡⁱ (K s⁻¹) of the diabatic `source` of a [`HeatingBudgetFlux`](@ref) to the
center field `𝒬`. `fields` are the fields of the closure's applied-flux energetics; `fields.cloud.ĉᵖΠ`
is the enthalpy coefficient `d(cᵖᵐ T)/dθˡⁱ` at the centers. Extend it for a source of another kind.
"""
function add_diabatic_heating! end

add_diabatic_heating!(𝒬, sources::Tuple, model, fields) = foreach(source -> add_diabatic_heating!(𝒬, source, model, fields), sources)

# In a kernel rather than a broadcast over the interior views, which a `ReactantState` array outside a traced
# function would index element by element
function add_diabatic_heating!(𝒬, source::PrescribedHeating, model, fields)
    grid = model.grid
    tendency = source.tendency
    launch!(grid.architecture, grid, :xyz, _add_prescribed_heating!, 𝒬, interior_data(tendency), interior_size(tendency))
    return nothing
end

# The values of a prescribed tendency, indexed from 1 over the interior, and the size of that interior: the data of
# a field (a reduced field keeps its data, not the field, so the kernel needs no field method), or an array
interior_data(tendency::AbstractField) = tendency.data
interior_data(tendency::AbstractArray) = tendency
interior_size(tendency::AbstractField) = size(interior(tendency))
interior_size(tendency::AbstractArray) = size(tendency)

# A prescribed tendency broadcasts with the interior: a dimension of size one is read at its single index
@kernel function _add_prescribed_heating!(𝒬, tendency, Nᵗ)
    i, j, k = @index(Global, NTuple)
    iᵗ = ifelse(Nᵗ[1] == 1, 1, i)
    jᵗ = ifelse(Nᵗ[2] == 1, 1, j)
    kᵗ = ifelse(Nᵗ[3] == 1, 1, k)
    @inbounds 𝒬[i, j, k] += tendency[iᵗ, jᵗ, kᵗ]
end

# ∂t θˡⁱ = -∂z ℐ / (ρ ĉᵖ Π), as the tendency of ρθ applies it (∂θˡⁱ/∂h = θˡⁱ / (ĉᵖ Π θˡⁱ))
@kernel function _add_radiative_heating!(𝒬, flux_divergence, ρ, ĉᵖΠ)
    i, j, k = @index(Global, NTuple)
    @inbounds 𝒬[i, j, k] += flux_divergence[i, j, k] / (ρ[i, j, k] * ĉᵖΠ[i, j, k])
end

function add_diabatic_heating!(𝒬, ::RadiativeHeating, model, fields)
    flux_divergence = AtmosphereModels.radiation_flux_divergence(model.radiation)
    isnothing(flux_divergence) && return nothing
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _add_radiative_heating!, 𝒬, flux_divergence, total_density(model.dynamics), fields.cloud.ĉᵖΠ)
    return nothing
end

#####
##### The redistributed heating and the flux, column by column
#####

# The cooling part of a heating, its smooth negative part with the scale `scale` (an array of the centers), as an array
# of the centers
struct CoolingPart{Q, S}
    heating :: Q
    scale :: S
end

@inline Base.getindex(c::CoolingPart, i, j, k) = @inbounds smooth_negative_part(c.heating[i, j, k], c.scale[i, j, k])

# One thread per column: the rate ε / (ĉᵖ Π) at which the dissipation heats the air, the redistributed heatings, then ρJ
# at the faces from the ground, (ρJ)ₖ₊½ = (ρJ)ₖ₋½ + ρ Δz (𝒬 - s), whose smooth positive part over ρ at the face is the
# flux (times the share ℂᑫ for the applied one), with the scale ℓ ε / (ĉᵖ Π) of the face. The boundary faces carry none.
@kernel function _compute_heating_budget_flux!(cl, grid, closure, closure_fields, e, velocities, ρ)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    hb = closure_ij.heating_budget_flux
    ℓᵐ = closure_fields.ℓᵐ
    FT = grid_float_type(grid)
    Nz = size(grid, 3)
    rate = cl.dissipative_heating_rate
    @inbounds for k in 1:Nz
        rate[i, j, k] = dissipationᶜᶜᶜ(i, j, k, grid, closure_ij, e, velocities, closure_fields) / cl.ĉᵖΠ[i, j, k]
    end
    cooling = CoolingPart(cl.𝒬, rate)
    exchange_amplitudes!(cl.sᑫ, cl.sᑫᶜ, cl.heating_work, i, j, grid, hb.ℂᑫᴸ, TurbulentShape(e, closure_ij.minimum_tke),
                         cl.𝒬, cooling, ρ, ℓᵐ)
    below = zero(FT)
    belowᶜ = zero(FT)
    @inbounds begin
        cl.Jᑫ[i, j, 1] = zero(FT)
        cl.Jᑫᶜ[i, j, 1] = zero(FT)
        for k in 1:Nz
            ρΔz = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            below += ρΔz * (cl.𝒬[i, j, k] - cl.sᑫ[i, j, k])
            belowᶜ += ρΔz * (cooling[i, j, k] - cl.sᑫᶜ[i, j, k])
            ρᶠ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ)
            δ = ℓᵐ[i, j, k+1] * (rate[i, j, k] + rate[i, j, min(k + 1, Nz)]) / 2
            cl.Jᑫ[i, j, k+1] = ifelse(k < Nz, hb.ℂᑫ * smooth_positive_part(below / ρᶠ, δ), zero(FT))
            cl.Jᑫᶜ[i, j, k+1] = ifelse(k < Nz, smooth_positive_part(belowᶜ / ρᶠ, δ), zero(FT))
        end
    end
end

compute_heating_budget_flux!(fields, ::Nothing, closure_fields, closure, model, tracers) = nothing

# Within `update_state!`, after the derivatives of the buoyancy (which give ĉᵖ Π) and the mixing length, and
# before the distribution, which receives the flux
function compute_heating_budget_flux!(fields, hb::HeatingBudgetFlux, closure_fields, closure, model, tracers)
    grid = model.grid
    cl = fields.cloud
    fill!(parent(cl.𝒬), 0)
    add_diabatic_heating!(cl.𝒬, hb.heating, model, fields)
    launch!(grid.architecture, grid, :xy, _compute_heating_budget_flux!, cl, grid, closure, closure_fields,
            tracers[TKE_NAME], model.velocities, total_density(model.dynamics))
    return nothing
end

#####
##### The conversions of the flux, at the centers
#####

# The conversions of the flux of θˡⁱ at the interior face k: of its buoyancy b̄_θ J; of the latent part,
# -[(b̄ᶜ - b̄ᵛ) aᴺ c_θ + (b̄ⁱ - b̄ᵛ) ηⁱ c_θⁱ + (b̄ˡ - b̄ᵛ) ηʷⁱ c_θ] J, of the fluxes of condensate, of ice and of liquid on ice
# that the fluxes of the saturation excesses over the distributed condensate, -c_θ J, and over ice, -c_θⁱ J, imply;
# and of both for the flux Jᶜ that the cooling alone drives
@inline function heating_flux_conversionsᶜᶜᶠ(i, j, k, grid, cl)
    @inbounds begin
        b_θ = (cl.b_θ[i, j, k] + cl.b_θ[i, j, k-1]) / 2
        bᶜᵛ, bⁱᵛ, bˡᵛ = latent_buoyancy_derivativesᶜᶜᶠ(i, j, k, cl)
        latent = -bᶜᵛ * cl.aᴺ[i, j, k] * cl.c_θ[i, j, k] - bⁱᵛ * cl.ηⁱ[i, j, k] * cl.c_θⁱ[i, j, k] -
                 bˡᵛ * cl.ηʷⁱ[i, j, k] * cl.c_θ[i, j, k]
        J = cl.Jᑫ[i, j, k]
        Jᶜ = cl.Jᑫᶜ[i, j, k]
    end
    return b_θ * J, latent * J, (b_θ + latent) * Jᶜ
end

# From the two faces of the cell k to its center (`half_cell_conversionᶜᶜᶜ`)
@inline function heating_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    lower = heating_flux_conversionsᶜᶜᶠ(i, j, max(k, 2), grid, cl)
    upper = heating_flux_conversionsᶜᶜᶠ(i, j, min(k + 1, size(grid, 3)), grid, cl)
    return map((l, u) -> half_cell_conversionᶜᶜᶜ(i, j, k, grid, ρ, l, u), lower, upper)
end

# Stores the conversions at the center k and returns what the TKE equation receives; nothing without the flux
@inline store_heating_flux_conversions!(cl, i, j, k, grid, ρ) =
    store_heating_flux_conversions!(has_field(cl, Val(:Jᑫ)), cl, i, j, k, grid, ρ)

@inline store_heating_flux_conversions!(::Val{false}, cl, i, j, k, grid, ρ) = zero(grid_float_type(grid))

@inline function store_heating_flux_conversions!(::Val{true}, cl, i, j, k, grid, ρ)
    dry, latent, cooling = heating_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    @inbounds begin
        cl.Gᑫ[i, j, k] = dry + latent
        cl.Gᑫᴸ[i, j, k] = latent
        cl.Gᑫᶜ[i, j, k] = cooling
    end
    return dry + latent
end

# The stored conversion, the applied flux at a face and the conversion that the cooling alone would drive; zero without the flux
@inline heating_flux_conversion(cl, i, j, k) = stored_or_zero(cl, Val(:Gᑫ), i, j, k, eltype(cl.B))
@inline applied_heating_flux(cl, i, j, k) = stored_or_zero(cl, Val(:Jᑫ), i, j, k, eltype(cl.w′²))
@inline cooling_flux_conversion(cl, i, j, k) = stored_or_zero(cl, Val(:Gᑫᶜ), i, j, k, eltype(cl.B))

# The conversion of the flux that the cooling drives with its coefficient ℂʷᑫ in the damped local third moment;
# `nothing` without a heating budget flux
@inline cooling_conversion(::Nothing, Gᑫᶜ) = nothing
@inline cooling_conversion(hb::HeatingBudgetFlux, Gᑫᶜ) = (Gᑫᶜ, hb.ℂʷᑫ)

#####
##### The flux in the stage tendency of ρθ, with the draw of the latent part of its conversion on the internal energy
#####

@kernel function _add_heating_budget_flux_tendency!(Gρθ, grid, cl, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        divergence = (ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ) * cl.Jᑫ[i, j, k+1] - ℑzᵃᵃᶠ(i, j, k, grid, ρ) * cl.Jᑫ[i, j, k]) / Δzᶜᶜᶜ(i, j, k, grid)
        latent_draw = ρ[i, j, k] * cl.Gᑫᴸ[i, j, k] / cl.ĉᵖΠ[i, j, k]
        Gρθ[i, j, k] -= (divergence + latent_draw) * !inactive_cell(i, j, k, grid)
    end
end

add_heating_budget_flux_tendency!(Gⁿ, fields, ::Nothing, model) = nothing

function add_heating_budget_flux_tendency!(Gⁿ, fields, ::HeatingBudgetFlux, model)
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _add_heating_budget_flux_tendency!, Gⁿ[name], grid, fields.cloud, total_density(model.dynamics))
    return nothing
end
