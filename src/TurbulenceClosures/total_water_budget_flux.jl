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
##### two levels: the amplitude is exchanged down its gradient. The matrix is the M-matrix of `exchange_amplitudes!`, one
##### solve per column; its columns sum to m, so Σ m s = Σ m 𝒬 and Jᵠ vanishes at the top of the column too: the flux
##### conserves the water. Where the air is still the levels do not exchange, s = 𝒬 and Jᵠ = 0: the tendency stays where
##### it acts. The surface flux is carried up over the turbulent layer.
#####
##### The tendencies 𝒬ᵠ are those of the model, in two parts, 𝒬ᵠ = 𝒬ᵠᵈ + 𝒬ᵠᵐ:
#####
#####   * 𝒬ᵠᵈ, the part that does not read the subgrid distribution, is diagnosed within the stage, before the
#####     distribution: the transport of the distributed water (the moisture density and the cloud liquid) by the
#####     resolved flow and by the sedimentation velocity of the cloud liquid, their forcing, and their boundary fluxes
#####     (the surface flux enters the lowest cell), evaluated with the model's own functions (`scalar_tendency` without
#####     a closure, and the boundary conditions) from the state of the stage.
#####   * 𝒬ᵠᵐ, the microphysical part (phase changes and process rates), reads the distribution, which reads the flux:
#####     the flux of a stage cannot read the microphysics of its own stage, and reads that of the stage before. The
#####     model records it where it adds the microphysical sources to the stage tendencies
#####     (`record_microphysical_tendencies!`), as the difference of the tendencies of the distributed water across
#####     them. It is state carried between stages, and it is checkpointed (`Oceananigans.prognostic_state` of the
#####     closure fields), so that a restart continues exactly; it is zero until a stage has built its tendencies, so the
#####     first stage of a run carries no microphysical tendency. A microphysics that updates the state outside the
#####     stage tendencies (an operator-split update) gives none.
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

The tendency is `𝒬ᵠ = 𝒬ᵠᵈ + 𝒬ᵠᵐ`. The part `𝒬ᵠᵈ` that does not read the subgrid distribution (the transport by the
resolved flow and by the sedimentation of the cloud liquid, the forcing and the boundary fluxes) is diagnosed within the
stage from its state. The microphysical part `𝒬ᵠᵐ` is that of the stage before, recorded when the model adds the
microphysical sources to the stage tendencies and checkpointed with the model: the microphysics of a stage reads the
subgrid distribution (P3's `SubgridCondensation`), which reads the flux, so the flux cannot read the microphysical
tendency of its own stage, and the first stage of a run carries none. The closure needs
`VerticallyImplicitTimeDiscretization`, under which its own vertical fluxes are applied by the implicit solves.

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
##### Fields: the tendency and its carried microphysical part, the tendency as redistributed, a work array, the applied
##### flux and its conversions
#####

total_water_budget_flux_fields(grid, ::Nothing) = NamedTuple()

# `𝒬ᵠᵐ` is the microphysical tendency that the flux reads, of the stage before; `stage_microphysical_water_tendency`
# is that of the stage being built, which becomes `𝒬ᵠᵐ` once the stage has its tendencies
total_water_budget_flux_fields(grid, ::TotalWaterBudgetFlux) =
    (𝒬ᵠ = CenterField(grid), 𝒬ᵠᵐ = CenterField(grid), stage_microphysical_water_tendency = CenterField(grid),
     sᵠ = CenterField(grid), budget_work = CenterField(grid), Jᵠ = ZFaceField(grid),
     Gᵇ = CenterField(grid), Gᵇᴸ = CenterField(grid))

#####
##### The part of the tendency that does not read the distribution, diagnosed from the state of the stage
#####

# The arguments of `AtmosphereModels.scalar_tendency` that belong to one scalar: its specific field, the position among
# the closure scalars (not read without a closure), its name, forcing and advection scheme
distributed_water_scalar(model, c, name) = (c, Val(1), Val(name), model.forcing[name], model.advection[name])

cloud_liquid_scalar(model, ::Nothing) = nothing
cloud_liquid_scalar(model, ρqᶜˡ) = distributed_water_scalar(model, AtmosphereModels.specific_scalar_fields(model).ρqᶜˡ, :ρqᶜˡ)

@inline transport_and_forcing(i, j, k, grid, ::Nothing, common) = zero(grid)
@inline transport_and_forcing(i, j, k, grid, scalar, common) = AtmosphereModels.scalar_tendency(i, j, k, grid, scalar..., common...)

# The tendency of the density of the distributed water by transport (resolved and sedimenting) and forcing
@kernel function _compute_water_transport_and_forcing!(tendency, grid, moisture, cloud_liquid, common)
    i, j, k = @index(Global, NTuple)
    @inbounds tendency[i, j, k] = transport_and_forcing(i, j, k, grid, moisture, common) +
                                  transport_and_forcing(i, j, k, grid, cloud_liquid, common)
end

@kernel function _sum_water_tendency!(cl, density_tendency, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds cl.𝒬ᵠ[i, j, k] = density_tendency[i, j, k] / ρ[i, j, k] + cl.𝒬ᵠᵐ[i, j, k]
end

# The boundary fluxes of the prognostic field `name`, as the stage tendencies receive them
function add_boundary_fluxes!(tendency, model, name)
    c = Oceananigans.prognostic_fields(model)[name]
    arguments = (model.grid.architecture, boundary_condition_args(model)...)
    compute_x_bcs!(tendency, c, arguments...)
    compute_y_bcs!(tendency, c, arguments...)
    AtmosphereModels.compute_stage_z_bcs!(tendency, c, AtmosphereModels.closure_applies_bottom_flux(model.closure, model, name), arguments...)
    return nothing
end

add_boundary_fluxes!(tendency, model, name, ::Nothing) = nothing
add_boundary_fluxes!(tendency, model, name, field) = add_boundary_fluxes!(tendency, model, name)

# 𝒬ᵠ = 𝒬ᵠᵈ + 𝒬ᵠᵐ. The diagnosed part is what `compute_tendencies!` and `compute_flux_bc_tendencies!` give the moisture
# density and the cloud liquid outside the microphysical sources and the closure: the model's `scalar_tendency`, without
# a closure, and its boundary fluxes. The closure fields are computed before the model refreshes its forcings and
# boundary conditions for the stage, so those of the distributed water are refreshed here (both refreshes are
# idempotent within a stage).
function diagnose_water_tendency!(cl, model)
    grid = model.grid
    arch = grid.architecture
    moisture_name = AtmosphereModels.moisture_prognostic_name(model.microphysics)
    ρqᶜˡ = cloud_liquid_density(model.microphysical_fields)
    prognostic = Oceananigans.prognostic_fields(model)

    AtmosphereModels.compute_forcing!(model.forcing[moisture_name])
    isnothing(ρqᶜˡ) || AtmosphereModels.compute_forcing!(model.forcing.ρqᶜˡ)
    update_boundary_conditions!(isnothing(ρqᶜˡ) ? (prognostic[moisture_name],) : (prognostic[moisture_name], prognostic.ρqᶜˡ), model)

    moisture = distributed_water_scalar(model, AtmosphereModels.specific_prognostic_moisture(model), moisture_name)
    cloud_liquid = cloud_liquid_scalar(model, ρqᶜˡ)
    common = (model.dynamics, model.formulation, model.thermodynamic_constants, AtmosphereModels.specific_prognostic_moisture(model),
              AtmosphereModels.transport_velocities(model), model.microphysics, model.microphysical_fields, nothing, nothing,
              model.clock, Oceananigans.fields(model))

    density_tendency = cl.budget_work
    launch!(arch, grid, :xyz, _compute_water_transport_and_forcing!, density_tendency, grid, moisture, cloud_liquid, common)
    add_boundary_fluxes!(density_tendency, model, moisture_name)
    add_boundary_fluxes!(density_tendency, model, :ρqᶜˡ, ρqᶜˡ)
    launch!(arch, grid, :xyz, _sum_water_tendency!, cl, density_tendency, total_density(model.dynamics))
    return nothing
end

#####
##### The microphysical part of the tendency, recorded where the model adds the microphysical sources to the stage
##### tendencies and carried to the next stage
#####

cloud_liquid_tendency(Gⁿ, ::Nothing) = nothing
cloud_liquid_tendency(Gⁿ, ρqᶜˡ) = Gⁿ.ρqᶜˡ

# The tendency of the distributed water, (Gρqᵛᵉ + Gρqᶜˡ) / ρ
function distributed_water_tendency(Gⁿ, model)
    moisture_name = AtmosphereModels.moisture_prognostic_name(model.microphysics)
    return SpecificTotalWater(getproperty(Gⁿ, moisture_name), cloud_liquid_tendency(Gⁿ, cloud_liquid_density(model.microphysical_fields)),
                              total_density(model.dynamics))
end

# Before the microphysical sources, minus the tendency of the distributed water (`kept = 0`, `sign = -1`); after them,
# plus it (`kept = 1`, `sign = 1`): the difference is the microphysical tendency
@kernel function _record_microphysical_water_tendency!(recorded, water_tendency, kept, sign)
    i, j, k = @index(Global, NTuple)
    @inbounds recorded[i, j, k] = kept * recorded[i, j, k] + sign * water_tendency[i, j, k]
end

@kernel function _carry_microphysical_water_tendency!(cl)
    i, j, k = @index(Global, NTuple)
    @inbounds cl.𝒬ᵠᵐ[i, j, k] = cl.stage_microphysical_water_tendency[i, j, k]
end

record_microphysical_water_tendency!(side, fields, ::Nothing, model) = nothing

function record_microphysical_water_tendency!(::AtmosphereModels.BeforeMicrophysicalSources, fields, ::TotalWaterBudgetFlux, model)
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _record_microphysical_water_tendency!, fields.cloud.stage_microphysical_water_tendency,
            distributed_water_tendency(model.timestepper.Gⁿ, model), 0, -1)
    return nothing
end

# A time stepper that builds the scalar tendencies outside `update_state!` builds them once in a stage, after the
# closure has added its tendencies: the recorded tendency is carried at once
function record_microphysical_water_tendency!(::AtmosphereModels.AfterMicrophysicalSources, fields, ::TotalWaterBudgetFlux, model)
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _record_microphysical_water_tendency!, fields.cloud.stage_microphysical_water_tendency,
            distributed_water_tendency(model.timestepper.Gⁿ, model), 1, 1)
    AtmosphereModels.scalar_tendencies_in_update_state(model.timestepper) ||
        launch!(grid.architecture, grid, :xyz, _carry_microphysical_water_tendency!, fields.cloud)
    return nothing
end

AtmosphereModels.record_microphysical_tendencies!(side, closure_fields, closure::FlavorOfTKEClosure, model) =
    record_microphysical_water_tendency!(side, closure_fields.subgrid_cloud, total_water_budget_flux(closure), model)

carry_microphysical_water_tendency!(fields, ::Nothing, model) = nothing

# Once in a stage, when the closure adds its tendencies: `update_state!` has recorded the microphysical tendency of the
# stage (and records the same again if it is called again), and the next evaluation of the closure fields reads it
function carry_microphysical_water_tendency!(fields, ::TotalWaterBudgetFlux, model)
    grid = model.grid
    AtmosphereModels.scalar_tendencies_in_update_state(model.timestepper) &&
        launch!(grid.architecture, grid, :xyz, _carry_microphysical_water_tendency!, fields.cloud)
    return nothing
end

# What a restart needs of the closure fields: the carried microphysical tendency
carried_fields(::Nothing) = nothing
carried_fields(fields) = carried_fields(has_field(fields.cloud, Val(:𝒬ᵠᵐ)), fields.cloud)
carried_fields(::Val{false}, cl) = nothing
carried_fields(::Val{true}, cl) = (; cl.𝒬ᵠᵐ)

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
    exchange_amplitudes!(cl.sᵠ, nothing, cl.budget_work, i, j, grid, sb.ℂᵇᴸ, NonNegativePart(e), cl.𝒬ᵠ, nothing, ρ, ℓᵐ)
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
    diagnose_water_tendency!(fields.cloud, model)
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

# The conversions of the flux of qᵗ at the interior face k: of the buoyancy of the vapor, b̄_qᵛ Jᵠ, and of the latent
# part, [(b̄ᶜ - b̄ᵛ) aᴺ c_q + (b̄ⁱ - b̄ᵛ) ηⁱ c_qⁱ + (b̄ˡ - b̄ᵛ) ηʷⁱ c_q] Jᵠ, of the fluxes of condensate, of ice and of liquid on
# ice that the fluxes of the saturation excesses over the distributed condensate, c_q Jᵠ, and over ice, c_qⁱ Jᵠ, imply
@inline function total_water_flux_conversionsᶜᶜᶠ(i, j, k, grid, cl)
    @inbounds begin
        b_qᵛ = (cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, k-1]) / 2
        bᶜᵛ, bⁱᵛ, bˡᵛ = latent_buoyancy_derivativesᶜᶜᶠ(i, j, k, cl)
        latent = bᶜᵛ * cl.aᴺ[i, j, k] * cl.c_q[i, j, k] + bⁱᵛ * cl.ηⁱ[i, j, k] * cl.c_qⁱ[i, j, k] + bˡᵛ * cl.ηʷⁱ[i, j, k] * cl.c_q[i, j, k]
        Jᵠ = cl.Jᵠ[i, j, k]
    end
    return b_qᵛ * Jᵠ, latent * Jᵠ
end

# From the two faces of the cell k to its center (`half_cell_conversionᶜᶜᶜ`)
@inline function total_water_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    lower = total_water_flux_conversionsᶜᶜᶠ(i, j, max(k, 2), grid, cl)
    upper = total_water_flux_conversionsᶜᶜᶠ(i, j, min(k + 1, size(grid, 3)), grid, cl)
    return map((l, u) -> half_cell_conversionᶜᶜᶜ(i, j, k, grid, ρ, l, u), lower, upper)
end

# Stores the conversions at the center k and returns what the TKE equation receives; nothing without the flux
@inline store_budget_flux_conversions!(cl, i, j, k, grid, ρ) =
    store_budget_flux_conversions!(has_field(cl, Val(:Jᵠ)), cl, i, j, k, grid, ρ)

@inline store_budget_flux_conversions!(::Val{false}, cl, i, j, k, grid, ρ) = zero(grid_float_type(grid))

@inline function store_budget_flux_conversions!(::Val{true}, cl, i, j, k, grid, ρ)
    dry, latent = total_water_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    @inbounds begin
        cl.Gᵇ[i, j, k] = dry + latent
        cl.Gᵇᴸ[i, j, k] = latent
    end
    return dry + latent
end

# The applied flux at a face; zero without it
@inline applied_budget_flux_q(cl, i, j, k) = stored_or_zero(cl, Val(:Jᵠ), i, j, k, eltype(cl.w′²))

# The stored conversion of the total water budget flux; zero without it
@inline total_water_budget_flux_conversion(cl, i, j, k) = stored_or_zero(cl, Val(:Gᵇ), i, j, k, eltype(cl.B))

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
