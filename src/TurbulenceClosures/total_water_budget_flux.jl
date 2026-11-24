#####
##### The budget flux of total water: a turbulent flux of qᵗ that carries every non-turbulent tendency of the distributed
##### water from where it acts to where the turbulence is, and its energy conversion in the TKE equation
#####
##### A component of `TKEBasedTurbulenceClosure` (`total_water_budget_flux = TotalWaterBudgetFlux(...)`), alongside the
##### flux down the gradient.
#####
##### In a turbulent layer near steady state the total water changes at the rate the turbulence spreads over the layer,
##### and the flux carries the rest of its non-turbulent tendency 𝒬ᵠ (forcing, microphysics, sedimentation, and the
##### surface flux, which enters the lowest cell): with the redistributed tendency sᵠ = c ω, of the shape ω = e and an
##### amplitude c exchanged between neighbouring levels over the length Λ = ℂᵇᴸ ℓ, the flux Jᵠ that does the
##### redistributing is
#####
#####     (ρJᵠ)ₖ₊½ - (ρJᵠ)ₖ₋½ = ρₖ Δzₖ (𝒬ᵠₖ - sᵠₖ),    (ρJᵠ)½ = 0,
#####     [mₖ + a₊θ₊ + a₋θ₋] sₖ - a₊(1 - θ₊) sₖ₊₁ - a₋(1 - θ₋) sₖ₋₁ = mₖ 𝒬ₖ,
#####     m = ρ Δz,    a± = 2 ρ± Λ±² / Δz±,    θ± = ωₖ±₁ / (ωₖ + ωₖ±₁),
#####
##### the construction of `HeatingBudgetFlux` (and of the sinks of `TKEBudgetThirdMoment`) applied to every tendency the
##### model gives the water. In the amplitude, (ρJ)₊ = -ρ₊ Λ₊² ω̃₊ (c₊ - c) / Δz₊ with ω̃ the harmonic mean of ω of the
##### two levels: the amplitude is exchanged down its gradient. The matrix is the M-matrix of `balance_sink!`, one
##### solve per column; its columns sum to m, so Σ m s = Σ m 𝒬 and Jᵠ vanishes at the top of the column too: the flux
##### conserves the water. Where the air is still the levels do not exchange, s = 𝒬 and Jᵠ = 0: the tendency stays where
##### it acts. The surface flux is carried up over the turbulent layer.
#####
##### The tendencies 𝒬ᵠ are those of the model: the stage tendencies of the distributed water (the moisture density and
##### the cloud liquid) that the model has built when the closure adds its own (`compute_closure_tendencies!`), that is
##### everything but the closure's fluxes. They are stored at each stage and the flux of the next evaluation of the
##### closure fields reads them, so the flux of a stage carries the tendencies of the stage before it (none in the first
##### stage of a run): the microphysics of a stage reads the distribution, which reads the flux, so the flux cannot read
##### the microphysics of its own stage.
#####
##### The closure applies the share ℂᵇᵠ of the flux to the moisture density, as vapor (the microphysics condenses what the
##### vapor brings), ∂t (ρ qᵛ) = ... - ∂z (ρ ℂᵇᵠ Jᵠ), and gives its energy conversion to the TKE equation as an applied
##### flux, at the faces and moved to the centers with the half-cell weights,
#####
#####     G = b_qᵛ Jᵠ + (b̄ᶜ - b̄ᵛ) aᴺ c_q Jᵠ + (b̄ⁱ - b̄ᵛ) ηⁱ c_qⁱ Jᵠ + (b̄ˡ - b̄ᵛ) ηʷⁱ c_q Jᵠ.
#####
##### The first term is the dynamic enthalpy the flux takes from the mean state, exactly; the rest is the latent part,
##### the buoyancy of the condensate fluxes that the flux of the saturation excesses implies, which the closure does not
##### apply and draws from the internal energy as it draws Bᴸ. The distribution receives the total flux of qᵗ; its
##### variances are those of the gradient (see below).
#####

"""
$(TYPEDEF)

A turbulent flux of total water that carries every non-turbulent tendency of the distributed water over the turbulent
layer, alongside the flux down the gradient of [`TKEBasedTurbulenceClosure`](@ref), and its energy conversion in the
TKE equation:

    (ρJᵠ)ₖ₊½ - (ρJᵠ)ₖ₋½ = ρₖ Δzₖ (𝒬ᵠₖ - sᵠₖ),

where `𝒬ᵠ` is the tendency the model gives the distributed water outside the closure (forcing, microphysics,
sedimentation, and the surface flux in the lowest cell) and `sᵠ = c e` the tendency as the turbulence redistributes
it: an amplitude `c` exchanged between neighbouring levels over the length `ℂᵇᴸ ℓ`, one tridiagonal solve per column.
The flux vanishes at the bottom and the top of the column and in still air, and conserves the water. The closure
applies the share `ℂᵇᵠ` of it, `-∂z (ρ ℂᵇᵠ Jᵠ)`, to the moisture density, and the TKE equation receives its
conversion, the energy it takes from the mean state (with the latent part of the condensate fluxes it implies drawn
from the internal energy). The subgrid distribution receives the total flux of qᵗ; its variances stay those of the
gradient, which vanish with the turbulence.

The tendencies are those the model built in the stage before (stored when the closure adds its own tendencies), so
the first stage of a run has no budget flux. The lag is structural: the microphysics of a stage reads the subgrid
distribution (P3's `SubgridCondensation`), which reads the flux, so the flux cannot read the microphysical tendency of
its own stage. On eight CfSite windows the effect of the flux on the 4 h state and on the first-hour TKE has the same
sign and size at time steps of 300 s and 30 s. The closure needs `VerticallyImplicitTimeDiscretization`, under which
the stage tendencies hold none of its own vertical fluxes.

$(TYPEDFIELDS)
"""
struct TotalWaterBudgetFlux{FT}
    "Share of the budget flux of qᵗ that is applied and converted, in [0, 1]"
    ℂᵇᵠ :: FT
    "Exchange length of the redistributed tendency in mixing lengths, `Λ = ℂᵇᴸ ℓ`; positive and finite"
    ℂᵇᴸ :: FT
end

# Coefficients of mixed types (an integer exchange length rebound by `with_parameters`) are promoted
TotalWaterBudgetFlux(ℂᵇᵠ::AbstractFloat, ℂᵇᴸ::Integer) = TotalWaterBudgetFlux(promote(ℂᵇᵠ, ℂᵇᴸ)...)
TotalWaterBudgetFlux(ℂᵇᵠ::Integer, ℂᵇᴸ::AbstractFloat) = TotalWaterBudgetFlux(promote(ℂᵇᵠ, ℂᵇᴸ)...)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`TotalWaterBudgetFlux`](@ref). By default the flux is applied whole (`ℂᵇᵠ = 1`) and the tendencies are
exchanged over a hundred mixing lengths, the exchange length with which the flux of this form, built from the
tendencies of large-eddy simulations of cumulus and stratocumulus (CfSites of the CNRM-CM6-1 amip pool), is closest
to their flux (normalized RMS error of w′qᵗ′ 0.20 below the cloud base and 0.42 in the cloud layer, against 0.75 and
0.77 of the down-gradient flux of the closure at its defaults).

[`total_water_budget_flux_bounds`](@ref) gives an interval for each coefficient for a calibration.
"""
function TotalWaterBudgetFlux(FT::DataType = Oceananigans.defaults.FloatType; ℂᵇᵠ = 1, ℂᵇᴸ = 100)
    0 <= ℂᵇᵠ <= 1 || throw(ArgumentError("ℂᵇᵠ must lie in [0, 1], got $(ℂᵇᵠ)."))
    0 < ℂᵇᴸ < Inf || throw(ArgumentError("ℂᵇᴸ must be positive and finite (the levels of still air must not exchange), got $(ℂᵇᴸ)."))
    return TotalWaterBudgetFlux(convert(FT, ℂᵇᵠ), convert(FT, ℂᵇᴸ))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The interval of each coefficient of a [`TotalWaterBudgetFlux`](@ref) for a calibration, as a `NamedTuple` of
`(lower, upper)`: the share `ℂᵇᵠ` in `[0, 1]` and the exchange length `ℂᵇᴸ` in `[4, 512]` mixing lengths.
"""
total_water_budget_flux_bounds() = (ℂᵇᵠ = (0.0, 1.0), ℂᵇᴸ = (4.0, 512.0))

convert_eltype(::Type{FT}, sb::TotalWaterBudgetFlux) where FT = TotalWaterBudgetFlux(convert(FT, sb.ℂᵇᵠ), convert(FT, sb.ℂᵇᴸ))

Adapt.adapt_structure(to, sb::TotalWaterBudgetFlux) = TotalWaterBudgetFlux(adapt(to, sb.ℂᵇᵠ), adapt(to, sb.ℂᵇᴸ))

Base.summary(sb::TotalWaterBudgetFlux) =
    string("TotalWaterBudgetFlux(ℂᵇᵠ = ", prettysummary(sb.ℂᵇᵠ), ", ℂᵇᴸ = ", prettysummary(sb.ℂᵇᴸ), ")")

Base.show(io::IO, sb::TotalWaterBudgetFlux) = print(io, summary(sb))

# The total water budget flux of a closure or of an ensemble of closures
total_water_budget_flux(closure::TKEBasedTurbulenceClosure) = closure.total_water_budget_flux
total_water_budget_flux(closure::TKEClosureArray) = first(closure).total_water_budget_flux

validate_total_water_budget_flux(::Nothing, static_stability, distribution, time_discretization) = nothing

function validate_total_water_budget_flux(::TotalWaterBudgetFlux, static_stability, distribution, time_discretization)
    time_discretization isa VerticallyImplicitTimeDiscretization ||
        throw(ArgumentError("TotalWaterBudgetFlux reads the stage tendencies without the closure's fluxes, which the vertically " *
                            "implicit time discretization leaves out of them; got $(summary(time_discretization))."))
    isnothing(distribution) && !(static_stability isa SaturationWeightedStaticStability) &&
        throw(ArgumentError("TotalWaterBudgetFlux needs the energetics of the applied fluxes: a subgrid_distribution, or " *
                            "static_stability = SaturationWeightedStaticStability(...)."))
    return nothing
end

#####
##### Fields: the stored tendency, the tendency as redistributed, a work array, the applied flux and its conversions
#####

total_water_budget_flux_fields(grid, ::Nothing) = NamedTuple()

total_water_budget_flux_fields(grid, ::TotalWaterBudgetFlux) =
    (𝒬ᵠ = CenterField(grid), sᵠ = CenterField(grid), budget_work = CenterField(grid), Jᵠ = ZFaceField(grid),
     Gᵇ = CenterField(grid), Gᵇᴸ = CenterField(grid))

#####
##### The tendency of the stage, without the closure's: stored when the closure adds its own
#####

# The tendency of the distributed water, (Gρqᵛᵉ + Gρqᶜˡ) / ρ
@kernel function _store_budget_sources!(cl, q_tendency)
    i, j, k = @index(Global, NTuple)
    @inbounds cl.𝒬ᵠ[i, j, k] = q_tendency[i, j, k]
end

store_budget_sources!(fields, ::Nothing, Gⁿ, model) = nothing

cloud_liquid_tendency(Gⁿ, ::Nothing) = nothing
cloud_liquid_tendency(Gⁿ, ρqᶜˡ) = Gⁿ.ρqᶜˡ

function store_budget_sources!(fields, ::TotalWaterBudgetFlux, Gⁿ, model)
    grid = model.grid
    ρ = total_density(model.dynamics)
    q_name = AtmosphereModels.moisture_prognostic_name(model.microphysics)
    q_tendency = SpecificTotalWater(getproperty(Gⁿ, q_name), cloud_liquid_tendency(Gⁿ, cloud_liquid_density(model.microphysical_fields)), ρ)
    launch!(grid.architecture, grid, :xyz, _store_budget_sources!, fields.cloud, q_tendency)
    return nothing
end

#####
##### The redistributed tendency and the flux, column by column
#####

# One thread per column: the redistributed tendency, then ρJ at the faces from the ground, (ρJ)ₖ₊½ = (ρJ)ₖ₋½ + ρ Δz (𝒬 - s),
# over ρ at the face and times the share ℂᵇᵠ for the applied flux. The boundary faces carry none.
@kernel function _compute_total_water_budget_flux!(cl, grid, closure, ℓᵐ, e, ρ)
    i, j = @index(Global, NTuple)
    sb = getclosure(i, j, closure).total_water_budget_flux
    FT = grid_float_type(grid)
    Nz = size(grid, 3)
    balance_redistributed_sources!(cl.sᵠ, nothing, cl.budget_work, i, j, grid, sb.ℂᵇᴸ, cl.𝒬ᵠ, nothing, e, ρ, ℓᵐ)
    below = zero(FT)
    @inbounds begin
        cl.Jᵠ[i, j, 1] = zero(FT)
        for k in 1:Nz
            below += ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid) * (cl.𝒬ᵠ[i, j, k] - cl.sᵠ[i, j, k])
            ρᶠ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ)
            cl.Jᵠ[i, j, k+1] = ifelse(k < Nz, sb.ℂᵇᵠ * below / ρᶠ, zero(FT))
        end
    end
end

compute_total_water_budget_flux!(fields, ::Nothing, closure_fields, closure, model, tracers) = nothing

# Within `update_state!`, after the mixing length and before the distribution, which receives the flux
function compute_total_water_budget_flux!(fields, ::TotalWaterBudgetFlux, closure_fields, closure, model, tracers)
    grid = model.grid
    launch!(grid.architecture, grid, :xy, _compute_total_water_budget_flux!, fields.cloud, grid, closure, closure_fields.ℓᵐ,
            tracers[TKE_NAME], total_density(model.dynamics))
    return nothing
end

#####
##### The variances are those of the gradient. A variance of the total flux, -Kᶜ ∂z qᵗ + Jᵠ, in the local production
##### balance, V (∂z qᵗ - Jᵠ / Kᶜ)², or in the flux–variance relation, (w′qᵗ′)² / (r² w′²), is singular where the turbulence
##### vanishes and the budget flux does not: the redistribution of the budget depends on the shape of e, not on its
##### magnitude, so a uniformly weak turbulence carries Jᵠ as a strong one does. The flux enters the distribution through
##### w′qᵗ′ (the means of its components, and the bound ℂʳ of the correlation where it is set).
#####

#####
##### The conversions of the flux, at the centers
#####

# The buoyancy of a flux of qᵗ at the interior face k, per unit flux: the vapor's, b̄_qᵛ, and the latent part of the
# condensate and ice fluxes that the fluxes of the saturation excesses over the distributed condensate, c_q Jᵠ, and over
# ice, c_qⁱ Jᵠ, imply
@inline function moisture_flux_buoyancyᶜᶜᶠ(i, j, k, grid, cl)
    @inbounds begin
        b_qᵛ = (cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, k-1]) / 2
        bᶜᵛ, bⁱᵛ, bˡᵛ = latent_buoyancy_derivativesᶜᶜᶠ(i, j, k, cl)
        latent = bᶜᵛ * cl.aᴺ[i, j, k] * cl.c_q[i, j, k] + bⁱᵛ * cl.ηⁱ[i, j, k] * cl.c_qⁱ[i, j, k] + bˡᵛ * cl.ηʷⁱ[i, j, k] * cl.c_q[i, j, k]
        Jᵠ = cl.Jᵠ[i, j, k]
    end
    return b_qᵛ * Jᵠ, latent * Jᵠ
end

# From the two faces of the cell k to its center with the half-cell weights (the boundary faces carry no flux)
@inline function budget_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    dryˡᵒʷ, latentˡᵒʷ = moisture_flux_buoyancyᶜᶜᶠ(i, j, max(k, 2), grid, cl)
    dryᵘᵖ, latentᵘᵖ = moisture_flux_buoyancyᶜᶜᶠ(i, j, min(k + 1, Nz), grid, cl)
    wˡᵒʷ, wᵘᵖ, ρΔz = half_cell_weights(i, j, k, grid, ρ)
    wˡᵒʷ = ifelse(k > 1, wˡᵒʷ, zero(FT))
    wᵘᵖ = ifelse(k < Nz, wᵘᵖ, zero(FT))
    return (wˡᵒʷ * dryˡᵒʷ + wᵘᵖ * dryᵘᵖ) / ρΔz, (wˡᵒʷ * latentˡᵒʷ + wᵘᵖ * latentᵘᵖ) / ρΔz
end

# Stores the conversions at the center k and returns what the TKE equation receives; nothing without the flux
@inline function store_budget_flux_conversions!(cl, i, j, k, grid, ρ)
    hasproperty(cl, :Jᵠ) || return zero(grid_float_type(grid))
    dry, latent = budget_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    @inbounds begin
        cl.Gᵇ[i, j, k] = dry + latent
        cl.Gᵇᴸ[i, j, k] = latent
    end
    return dry + latent
end

# The applied flux at a face; zero without it
@inline applied_budget_flux_q(cl, i, j, k) = hasproperty(cl, :Jᵠ) ? @inbounds(cl.Jᵠ[i, j, k]) : zero(eltype(cl.w′²))

# The stored conversion of the total water budget flux; zero without it
@inline total_water_budget_flux_conversion(cl, i, j, k) = hasproperty(cl, :Gᵇ) ? @inbounds(cl.Gᵇ[i, j, k]) : zero(eltype(cl.B))

# The conversions of both budget fluxes, which the stage tendency gives the TKE
@inline budget_flux_conversions(cl, i, j, k) = heating_flux_conversion(cl, i, j, k) + total_water_budget_flux_conversion(cl, i, j, k)

#####
##### The flux in the stage tendency of the moisture density, with the draw of the latent part of its conversion on the
##### internal energy
#####

@kernel function _add_total_water_budget_flux_tendency!(Gρθ, Gρq, grid, cl, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        divergence = (ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ) * cl.Jᵠ[i, j, k+1] - ℑzᵃᵃᶠ(i, j, k, grid, ρ) * cl.Jᵠ[i, j, k]) / Δzᶜᶜᶜ(i, j, k, grid)
        latent_draw = ρ[i, j, k] * cl.Gᵇᴸ[i, j, k] / cl.ĉᵖΠ[i, j, k]
        active = !inactive_cell(i, j, k, grid)
        Gρθ[i, j, k] -= latent_draw * active
        Gρq[i, j, k] -= divergence * active
    end
end

add_total_water_budget_flux_tendency!(Gⁿ, fields, ::Nothing, model) = nothing

function add_total_water_budget_flux_tendency!(Gⁿ, fields, ::TotalWaterBudgetFlux, model)
    grid = model.grid
    θ_name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    q_name = AtmosphereModels.moisture_prognostic_name(model.microphysics)
    launch!(grid.architecture, grid, :xyz, _add_total_water_budget_flux_tendency!, getproperty(Gⁿ, θ_name), getproperty(Gⁿ, q_name),
            grid, fields.cloud, total_density(model.dynamics))
    return nothing
end
