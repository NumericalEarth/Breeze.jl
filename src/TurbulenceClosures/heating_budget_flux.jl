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
##### with the shape ω = e and an amplitude c exchanged between neighbouring levels over the length
##### Λ = ℂᑫᴸ ℓ, and the flux that does the redistributing is J,
#####
#####     (ρJ)ₖ₊½ - (ρJ)ₖ₋½ = ρₖ Δzₖ (𝒬ₖ - sₖ),    (ρJ)½ = 0,
#####     [mₖ + a₊θ₊ + a₋θ₋] sₖ - a₊(1 - θ₊) sₖ₊₁ - a₋(1 - θ₋) sₖ₋₁ = mₖ 𝒬ₖ,
#####     m = ρ Δz,    a± = 2 ρ± Λ±² / Δz±,    θ± = ωₖ±₁ / (ωₖ + ωₖ±₁).
#####
##### The matrix is the M-matrix of `balance_sinks!`; its columns sum to m, so Σ m s = Σ m 𝒬 and J
##### vanishes at the top of the column too. Where the air is still (ω = 0 at a level and its
##### neighbours) the levels do not exchange, s = 𝒬 and J = 0; a heating that is uniform over a
##### turbulent layer gives J = 0; cooling at the top of a turbulent layer, or heating at its bottom,
##### gives J > 0, and the opposite J < 0.
#####
##### The closure applies the share ℂᑫ of the positive part to the mean state,
#####
#####     Jᑫ = ℂᑫ max(J, 0),    ∂t (ρ θˡⁱ) = ... - ∂z (ρ Jᑫ),
#####
##### (a heating that stabilizes stays where it acts, and the gradient closure sees it), and gives its
##### energy conversion to the TKE equation, whose production is then P + B + G - ε:
#####
#####     G = b_θ Jᑫ - (b̄ˡ - b̄ᵛ) aᴺ c_θ Jᑫ    at the faces, moved to the centers with the half-cell weights.
#####
##### The first term is the dynamic enthalpy the flux takes from the mean state, exactly (the dynamic
##### enthalpy is linear in θˡⁱ at the stage's b_θ, as for the fluxes down the gradient). The second is
##### the latent part: the flux of θˡⁱ is a flux -c_θ Jᑫ of the saturation excess, of which the share
##### aᴺ condenses (`_latent_corrected_stability!`); the closure applies no condensate flux with Jᑫ, so
##### the buoyancy conversion of that condensate flux is drawn from the internal energy, as Bᴸ is.
##### Integrated over a layer with one amplitude, ∫ ρ b_θ J dz = -∫ ρ b_θ (z - z̄) 𝒬 dz: the generation
##### of available potential energy against the mixed state.
#####
##### The distribution receives the total flux of θˡⁱ (w′θˡ′ of the moments includes Jᑫ), and the
##### third moment of `TKEBudgetThirdMoment` receives G through P + B. Its damped local member is
##### signed by origin: Gᑫᶜ, the conversion of the flux that the cooling part min(𝒬, 0) alone would
##### drive (with ℂᑫ = 1), is carried by downdrafts, so
##### the buoyant source of the local member is ℂʷᵇ (B⁺ - Gᶜ) - ℂʷᑫ Gᶜ, Gᶜ = min(Gᑫᶜ, B⁺).
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

    (ρJ)ₖ₊½ - (ρJ)ₖ₋½ = ρₖ Δzₖ (𝒬ₖ - sₖ),    Jᑫ = ℂᑫ max(J, 0),    G = (b_θ - (bˡ - bᵛ) aᴺ c_θ) Jᑫ,

where `𝒬` is the heating (the tendency of θˡⁱ of the sources in `heating`) and `s` the heating as
the turbulence redistributes it: `s = c e` with an amplitude `c` exchanged between neighbouring
levels over the length `ℂᑫᴸ ℓ` (one tridiagonal solve per column). `J` vanishes at the bottom and
the top of the column and in still air. The closure applies `-∂z (ρ Jᑫ)` to `ρθ`, and the TKE
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
below a radiatively cooled anvil, where the large-eddy simulations have positive skewness (harness
`notes/total-energy-design.md` §11).

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
##### and conversion that the cooling alone would drive, and a work array
#####

heating_budget_flux_fields(grid, ::Nothing) = NamedTuple()

heating_budget_flux_fields(grid, ::HeatingBudgetFlux) =
    (𝒬 = CenterField(grid), sᑫ = CenterField(grid), sᑫᶜ = CenterField(grid), heating_work = CenterField(grid),
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

function add_diabatic_heating!(𝒬, source::PrescribedHeating, model, fields)
    interior(𝒬) .+= interior_values(source.tendency)
    return nothing
end

# The values of a prescribed tendency: of a field, or of an array of the interior
interior_values(tendency::AbstractField) = interior(tendency)
interior_values(tendency::AbstractArray) = tendency

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

# The heating as redistributed, `s`, and that of its cooling part min(𝒬, 0), `sᶜ`, of the column at once: the matrix of
# `balance_sinks!` with the shape ω = max(e, 0) and the exchange length ℂᑫᴸ ℓ, eliminated once for the two right-hand
# sides. `work` holds the eliminated upper diagonal.
@inline function balance_redistributed_heating!(s, sᶜ, work, i, j, grid, ℂᑫᴸ, 𝒬, e, ρ, ℓᵐ)
    Nz = size(grid, 3)
    FT = eltype(grid)
    tiny = sqrt(floatmin(FT))
    @inbounds begin
        # Thomas algorithm; the matrix is diagonally dominant by columns
        ω⁻ = zero(FT)
        c′ = zero(FT)
        d′ = zero(FT)
        d′ᶜ = zero(FT)
        a⁻ = zero(FT) # the exchange coefficient at the face below
        ω = max(e[i, j, 1], zero(FT))
        for k in 1:Nz
            m = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            diagonal = m
            lower = zero(FT)
            upper = zero(FT)
            ω⁺ = zero(FT)
            if k > 1
                share = ω + ω⁻ + tiny
                diagonal += a⁻ * ω⁻ / share
                lower = -a⁻ * ω / share
            end
            if k < Nz
                a = sink_exchange_coefficientᶜᶜᶠ(i, j, k + 1, grid, ℂᑫᴸ, ρ, ℓᵐ)
                ω⁺ = max(e[i, j, k+1], zero(FT))
                share = ω + ω⁺ + tiny
                diagonal += a * ω⁺ / share
                upper = -a * ω / share
                a⁻ = a
            end
            heating = 𝒬[i, j, k]
            pivot = diagonal - lower * c′
            d′ = (m * heating - lower * d′) / pivot
            d′ᶜ = (m * min(heating, zero(FT)) - lower * d′ᶜ) / pivot
            c′ = upper / pivot
            work[i, j, k] = c′
            s[i, j, k] = d′
            sᶜ[i, j, k] = d′ᶜ
            ω⁻ = ω
            ω = ω⁺
        end
        for k in Nz-1:-1:1
            s[i, j, k] -= work[i, j, k] * s[i, j, k+1]
            sᶜ[i, j, k] -= work[i, j, k] * sᶜ[i, j, k+1]
        end
    end
    return nothing
end

# One thread per column: the redistributed heatings, then ρJ at the faces from the ground,
# (ρJ)ₖ₊½ = (ρJ)ₖ₋½ + ρ Δz (𝒬 - s), whose positive part over ρ at the face is the flux (times the share ℂᑫ for the
# applied one). The boundary faces carry none.
@kernel function _compute_heating_budget_flux!(cl, grid, closure, ℓᵐ, e, ρ)
    i, j = @index(Global, NTuple)
    hb = getclosure(i, j, closure).heating_budget_flux
    FT = eltype(grid)
    Nz = size(grid, 3)
    balance_redistributed_heating!(cl.sᑫ, cl.sᑫᶜ, cl.heating_work, i, j, grid, hb.ℂᑫᴸ, cl.𝒬, e, ρ, ℓᵐ)
    below = zero(FT)
    belowᶜ = zero(FT)
    @inbounds begin
        cl.Jᑫ[i, j, 1] = zero(FT)
        cl.Jᑫᶜ[i, j, 1] = zero(FT)
        for k in 1:Nz
            ρΔz = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            heating = cl.𝒬[i, j, k]
            below += ρΔz * (heating - cl.sᑫ[i, j, k])
            belowᶜ += ρΔz * (min(heating, zero(FT)) - cl.sᑫᶜ[i, j, k])
            ρᶠ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ)
            cl.Jᑫ[i, j, k+1] = ifelse(k < Nz, hb.ℂᑫ * max(below, zero(FT)) / ρᶠ, zero(FT))
            cl.Jᑫᶜ[i, j, k+1] = ifelse(k < Nz, max(belowᶜ, zero(FT)) / ρᶠ, zero(FT))
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
    launch!(grid.architecture, grid, :xy, _compute_heating_budget_flux!, cl, grid, closure, closure_fields.ℓᵐ,
            tracers[TKE_NAME], total_density(model.dynamics))
    return nothing
end

#####
##### The conversions of the flux, at the centers
#####

# The buoyancy of a flux of θˡⁱ at the interior face k, per unit flux: b̄_θ, and the latent part
# -(b̄ˡ - b̄ᵛ) aᴺ c_θ of the condensate flux that the flux of the saturation excess -c_θ J implies
@inline function heating_flux_buoyancyᶜᶜᶠ(i, j, k, grid, cl)
    @inbounds begin
        b_θ = (cl.b_θ[i, j, k] + cl.b_θ[i, j, k-1]) / 2
        bˡᵛ = (cl.b_qˡ[i, j, k] + cl.b_qˡ[i, j, k-1] - cl.b_qᵛ[i, j, k] - cl.b_qᵛ[i, j, k-1]) / 2
        latent = -bˡᵛ * cl.aᴺ[i, j, k] * cl.c_θ[i, j, k]
    end
    return b_θ, latent
end

# From the two faces of the cell k to its center with the half-cell weights (the boundary faces carry no flux)
@inline function heating_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    Nz = size(grid, 3)
    FT = eltype(grid)
    kˡᵒʷ = max(k, 2)
    kᵘᵖ = min(k + 1, Nz)
    b_θˡᵒʷ, latentˡᵒʷ = heating_flux_buoyancyᶜᶜᶠ(i, j, kˡᵒʷ, grid, cl)
    b_θᵘᵖ, latentᵘᵖ = heating_flux_buoyancyᶜᶜᶠ(i, j, kᵘᵖ, grid, cl)
    @inbounds begin
        Jˡᵒʷ = ifelse(k > 1, cl.Jᑫ[i, j, kˡᵒʷ], zero(FT))
        Jᵘᵖ = ifelse(k < Nz, cl.Jᑫ[i, j, kᵘᵖ], zero(FT))
        Jᶜˡᵒʷ = ifelse(k > 1, cl.Jᑫᶜ[i, j, kˡᵒʷ], zero(FT))
        Jᶜᵘᵖ = ifelse(k < Nz, cl.Jᑫᶜ[i, j, kᵘᵖ], zero(FT))
    end
    zᶜ = znode(i, j, k, grid, Center(), Center(), Center())
    wˡᵒʷ = ℑzᵃᵃᶠ(i, j, k, grid, ρ) * (zᶜ - znode(i, j, k, grid, Center(), Center(), Face()))
    wᵘᵖ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ) * (znode(i, j, k + 1, grid, Center(), Center(), Face()) - zᶜ)
    ρΔz = @inbounds ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
    dry = (wˡᵒʷ * b_θˡᵒʷ * Jˡᵒʷ + wᵘᵖ * b_θᵘᵖ * Jᵘᵖ) / ρΔz
    latent = (wˡᵒʷ * latentˡᵒʷ * Jˡᵒʷ + wᵘᵖ * latentᵘᵖ * Jᵘᵖ) / ρΔz
    cooling = (wˡᵒʷ * (b_θˡᵒʷ + latentˡᵒʷ) * Jᶜˡᵒʷ + wᵘᵖ * (b_θᵘᵖ + latentᵘᵖ) * Jᶜᵘᵖ) / ρΔz
    return dry, latent, cooling
end

# Stores the conversions at the center k and returns what the TKE equation receives; nothing without the flux
@inline function store_heating_flux_conversions!(cl, i, j, k, grid, ρ)
    hasproperty(cl, :Jᑫ) || return zero(eltype(grid))
    dry, latent, cooling = heating_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    @inbounds begin
        cl.Gᑫ[i, j, k] = dry + latent
        cl.Gᑫᴸ[i, j, k] = latent
        cl.Gᑫᶜ[i, j, k] = cooling
    end
    return dry + latent
end

# The stored conversion, the applied flux at a face and the conversion that the cooling alone would drive; zero without the flux
@inline heating_flux_conversion(cl, i, j, k) = hasproperty(cl, :Gᑫ) ? @inbounds(cl.Gᑫ[i, j, k]) : zero(eltype(cl.B))
@inline applied_heating_flux(cl, i, j, k) = hasproperty(cl, :Jᑫ) ? @inbounds(cl.Jᑫ[i, j, k]) : zero(eltype(cl.w′²))
@inline cooling_flux_conversion(cl, i, j, k) = hasproperty(cl, :Gᑫᶜ) ? @inbounds(cl.Gᑫᶜ[i, j, k]) : zero(eltype(cl.B))

# ℂʷᑫ of the closure at a column; zero without the flux
@inline cooling_third_moment_coefficient(::Nothing, FT) = zero(FT)
@inline cooling_third_moment_coefficient(hb::HeatingBudgetFlux, FT) = convert(FT, hb.ℂʷᑫ)

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
