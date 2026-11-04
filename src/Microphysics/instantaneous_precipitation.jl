using ..Thermodynamics:
    MoistureMassFractions,
    LiquidIceDensityState,
    LiquidIcePotentialTemperatureState,
    temperature,
    with_moisture,
    with_temperature,
    WarmPhaseEquilibrium

using ..AtmosphereModels:
    AtmosphereModels,
    continuity_density,
    dynamics_pressure,
    kernel_time_step,
    liquid_ice_exner_exponent,
    total_density,
    standard_pressure

using Breeze.CompressibleEquations: CompressibleDynamics
using Breeze.Solvers: NewtonSolver

using Oceananigans: Oceananigans, CenterField, Field, Integral
using Oceananigans.Architectures: architecture
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: launch!
using KernelAbstractions: @index, @kernel
using DocStringExtensions: TYPEDSIGNATURES

"""
    struct InstantaneousPrecipitation{S}

Instantaneous-precipitation microphysics: an instantaneous,
*irreversible* condensation with immediate rain-out and no re-evaporation
(no cloud or rain stage). Excess water vapor above saturation condenses, releases
its latent heat to the air, and is removed as precipitation in the same step. This
is the "large-scale condensation" of the DCMIP2016 Reed–Jablonowski simple-physics suite.

The saturation/latent-heat solve is delegated to a `SaturationAdjustment`
instance (`saturation_adjustment`), so the equilibrium thermodynamics are shared
and validated. The distinction is irreversibility: the condensate is purged from
the prognostic vapor every step and cannot re-evaporate.

The prognostic moisture is the vapor density `ρqᵛ` (no condensate is retained).
On the compressible core, rain-out leaves dry-air density unchanged and reduces
total density by the removed condensate. The remaining vapor mass fraction and
potential temperature are recomputed at that density, preserving the temperature
reached during condensation.
"""
struct InstantaneousPrecipitation{S}
    saturation_adjustment :: S
end

"""
$(TYPEDSIGNATURES)

Construct an `InstantaneousPrecipitation` scheme. `equilibrium` selects the phase
equilibrium used by the underlying saturation solve (default warm-phase), and `solver`
controls its iteration (see [`SaturationAdjustment`](@ref)).
"""
function InstantaneousPrecipitation(FT::DataType=Oceananigans.defaults.FloatType;
                                         equilibrium = WarmPhaseEquilibrium(),
                                         solver = SecantSolver(FT))
    sa = SaturationAdjustment(FT; solver, equilibrium)
    return InstantaneousPrecipitation(sa)
end

const IP = InstantaneousPrecipitation

@inline Diagnostics.microphysics_phase_equilibrium(μ::IP) = microphysics_phase_equilibrium(μ.saturation_adjustment)

# Prognostic vapor density; no retained condensate fields.
AtmosphereModels.moisture_prognostic_name(::IP) = :ρqᵛ
AtmosphereModels.prognostic_field_names(::IP) = tuple()

AtmosphereModels.liquid_mass_fraction(::IP, model) = nothing
AtmosphereModels.ice_mass_fraction(::IP, model) = nothing
# Condensate is removed instantly rather than transported, so nothing sediments and the
# generic `microphysical_velocities`/`sedimentation_velocity` fallbacks stand.
@inline AtmosphereModels.microphysical_tendency(::IP, name, ρ, ℳ, 𝒰, constants) = zero(ρ)

# Diagnostic vapor + a precipitation-rate field (kg m⁻³ s⁻¹ of condensed/removed water).
AtmosphereModels.materialize_microphysical_fields(::IP, grid, bcs) =
    (; qᵛ = CenterField(grid), precipitation_rate = CenterField(grid))

# Prognostic vapor only; no stored condensate (all moisture is vapor between steps).
@inline function AtmosphereModels.grid_moisture_fractions(i, j, k, grid, ::IP, ρ, qᵛ, μ)
    return MoistureMassFractions(qᵛ)
end

@inline function AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, ::IP, ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    return nothing
end

# Like DCMIP2016KM, the diagnostic state is returned UNCHANGED: all condensation
# (and its latent heating) happens once per step in microphysics_model_update!,
# not in the per-stage equilibrium adjustment. Delegating to SA here would make
# update_state! re-condense after the kernel and double-count the latent heat.
@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰, ::IP, qᵛ, constants) = 𝒰

AtmosphereModels.precipitation_rate(model, ::IP, ::Val{:liquid}) = model.microphysical_fields.precipitation_rate
AtmosphereModels.precipitation_rate(model, ::IP, ::Val{:ice}) = nothing

"""
$(TYPEDSIGNATURES)

Return the bottom precipitation flux for the instantaneous-precipitation scheme.

The scheme removes condensed water immediately, so the flux is the column integral
of the volumetric precipitation rate. Units are kg/m²/s.
"""
function AtmosphereModels.bottom_precipitation_flux(model, ::IP)
    P = model.microphysical_fields.precipitation_rate
    return Field(Integral(P, dims=3))
end

#####
##### Once-per-step irreversible condensation + rain-out
#####

"""
$(TYPEDSIGNATURES)

Condense supersaturation, retain the released latent heat, and remove the
condensate as precipitation — applied directly to the prognostic vapor `ρqᵛ`
and liquid-ice potential temperature density `ρθˡⁱ`.
"""
function AtmosphereModels.microphysics_model_update!(microphysics::IP, model)
    grid = model.grid
    arch = architecture(grid)
    Δt = model.clock.last_Δt

    # Skip before the first step (Δt invalid during construction).
    invalid_time_step(Δt) && return nothing

    pˢᵗ = standard_pressure(model.dynamics)
    constants = model.thermodynamic_constants
    ρθˡⁱ = model.formulation.potential_temperature_density
    ρqᵛ = model.moisture_density
    μ = model.microphysical_fields

    saturation_adjustment = microphysics.saturation_adjustment

    kernel_Δt = kernel_time_step(arch, grid, Δt)

    κ̂ = liquid_ice_exner_exponent(model.formulation)

    launch!(arch, grid, :xyz, _instantaneous_precipitation_update!,
            model.dynamics, saturation_adjustment, constants, pˢᵗ, κ̂, kernel_Δt, ρθˡⁱ, ρqᵛ, μ)

    # The kernel mutated prognostic fields in the interior only; refill halos and recompute
    # diagnostics and tendencies so the post-update state is consistent for the next step.
    update_state!(model)

    return nothing
end

# The condensation holds fixed what the dynamics holds fixed, and the state type carries that
# constraint into `adjust_thermodynamic_state` and the θˡⁱ inversion. The compressible core evolves
# its density, so the adjustment is isochoric at the cell's total density ρ. The anelastic core, like
# the other cores closed on a pressure (kinematic, parcel), holds the reference or prescribed
# pressure fixed, so the adjustment is isobaric there: temperature and saturation are evaluated at
# that pressure, at the thermodynamic state density p / (Rᵐ T), not at the total density ρ = ρᵣ that
# weights the anelastic prognostics. This is the split `diagnose_thermodynamic_state` makes.
@inline function instantaneous_precipitation_state(i, j, k, dynamics, θ, q, ρ, pˢᵗ, κ̂)
    p = @inbounds dynamics_pressure(dynamics)[i, j, k]
    return LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, p, κ̂)
end

@inline instantaneous_precipitation_state(i, j, k, ::CompressibleDynamics, θ, q, ρ, pˢᵗ, κ̂) =
    LiquidIceDensityState(θ, q, pˢᵗ, ρ, NewtonSolver(typeof(ρ)), κ̂)

# Pressure-closed cores keep their density weighting fixed when condensate leaves.
@inline rainout_state(𝒰, qᵛ, ρqᵛ, ρᵈ) = with_moisture(𝒰, MoistureMassFractions(qᵛ))

# The compressible cell loses condensate mass at fixed volume, leaving only dry air
# and vapor: ρ = ρᵈ + ρqᵛ. Renormalize the vapor before inverting θ at the retained T.
@inline function rainout_state(𝒰::LiquidIceDensityState, qᵛ, ρqᵛ, ρᵈ)
    ρ = ρᵈ + ρqᵛ
    q = MoistureMassFractions(ρqᵛ / ρ)
    return LiquidIceDensityState(𝒰.potential_temperature, q, 𝒰.standard_pressure, ρ,
                                𝒰.temperature_solver, 𝒰.exner_exponent)
end

@kernel function _instantaneous_precipitation_update!(dynamics, saturation_adjustment,
                                                   constants, pˢᵗ, κ̂, Δt, ρθˡⁱ, ρqᵛ, μ)
    i, j, k = @index(Global, NTuple)

    ρ_field  = total_density(dynamics)       # total ρ: mass fractions and the density weighting them
    ρᵈ_field = continuity_density(dynamics)  # ρᵈ: ρθ = ρᵈθ is dry-coupled

    @inbounds begin
        ρ  = ρ_field[i, j, k]
        ρᵈ = ρᵈ_field[i, j, k]
        ρqᵛ₀ = ρqᵛ[i, j, k]
        qᵗ = ρqᵛ₀ / ρ              # total water mass fraction (÷ total ρ); prognostic vapor, no retained condensate
        θ₀ = ρθˡⁱ[i, j, k] / ρᵈ     # θˡⁱ = ρθˡⁱ/ρᵈ (dry-coupled), conserved by the reversible condensation

        # Condensation: saturation adjustment with θˡⁱ held fixed, at the cell's density on the
        # compressible core and at its pressure otherwise (see `instantaneous_precipitation_state`).
        # Delegated to the shared #765 secant. Subsaturated cells short-circuit untouched, matching
        # the original large-scale-condensation behavior.
        𝒰₀  = instantaneous_precipitation_state(i, j, k, dynamics, θ₀, MoistureMassFractions(qᵗ), ρ, pˢᵗ, κ̂)
        𝒰₁  = adjust_thermodynamic_state(𝒰₀, saturation_adjustment, constants)
        q   = 𝒰₁.moisture_mass_fractions
        T   = temperature(𝒰₁, constants)
        qᵛ⁺ = q.vapor

        # Rain-out keeps T while removing condensate. On the compressible core, recover θ
        # using the reduced total density and renormalized vapor mass fraction that
        # update_state! will diagnose; pressure-closed cores retain their prescribed pressure.
        ρqᵛ⁺ = ρ * qᵛ⁺
        𝒰ᵥ = rainout_state(𝒰₁, qᵛ⁺, ρqᵛ⁺, ρᵈ)
        θᶠ = with_temperature(𝒰ᵥ, T, constants).potential_temperature
        # Mass-fraction form (≡ ρqᵛ₀ - ρqᵛ⁺, since qᵗ = ρqᵛ₀/ρ) but exactly 0 when subsaturated
        # (qᵛ⁺ = qᵗ), avoiding a round-trip ρ·(ρqᵛ₀/ρ) that would leave a spurious tiny precip rate.
        condensed_water_density = ρ * max(0, qᵗ - qᵛ⁺)
        precipitation_rate = condensed_water_density / Δt

        ρqᵛ[i, j, k]  = ρqᵛ⁺
        ρθˡⁱ[i, j, k] = ρᵈ * θᶠ    # ρθ = ρᵈθ (dry-coupled)
        μ.qᵛ[i, j, k] = 𝒰ᵥ.moisture_mass_fractions.vapor
        μ.precipitation_rate[i, j, k] = ifelse(condensed_water_density > 0,
                                               precipitation_rate,
                                               μ.precipitation_rate[i, j, k])
    end
end
