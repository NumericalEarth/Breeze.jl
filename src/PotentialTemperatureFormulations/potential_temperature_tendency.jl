using Breeze.AtmosphereModels.Diagnostics: Diagnostics
using Breeze.AtmosphereModels: AtmosphereModel, specific_prognostic_moisture

using Oceananigans.Fields: Field, set!
using Breeze.Thermodynamics: temperature
using Breeze.BoundaryConditions: theta_to_energy_bcs, materialize_atmosphere_field_bcs

const PotentialTemperatureModel = AtmosphereModel{<:Any, <:LiquidIcePotentialTemperatureFormulation}

AtmosphereModels.specific_thermodynamic_field(formulation::LiquidIcePotentialTemperatureFormulation) = formulation.potential_temperature

#####
##### Helper accessors
#####

AtmosphereModels.liquid_ice_potential_temperature_density(model::PotentialTemperatureModel) = model.formulation.potential_temperature_density
AtmosphereModels.liquid_ice_potential_temperature(model::PotentialTemperatureModel) = model.formulation.potential_temperature
AtmosphereModels.static_energy(model::PotentialTemperatureModel) = Diagnostics.StaticEnergy(model, :specific)

"""
    static_energy_density(model::PotentialTemperatureModel)

Return the static energy density as a `Field` with boundary conditions that return
energy fluxes when used with `BoundaryConditionOperation`.

For `LiquidIcePotentialTemperatureFormulation`, the prognostic variable is potential
temperature density `ρθ`. This function converts the `ρθ` boundary conditions to
energy flux boundary conditions by multiplying by ``ĉᵖ Π``, the inverse of ``∂θˡⁱ/∂h``.
"""
function AtmosphereModels.static_energy_density(model::PotentialTemperatureModel)
    ρθ = model.formulation.potential_temperature_density
    ρθ_bcs = ρθ.boundary_conditions

    # Convert θ BCs to energy BCs
    ρs_bcs = theta_to_energy_bcs(ρθ_bcs, model.formulation.exner_exponent)

    # Regularize the converted BCs (populate microphysics, constants, side)
    loc = (Center(), Center(), Center())
    ρs_bcs = materialize_atmosphere_field_bcs(ρs_bcs, loc, model.grid, model.dynamics, model.microphysics,
                                              model.thermodynamic_constants)

    # Create the energy density operation and wrap in a Field with proper BCs
    ρs_op = Diagnostics.StaticEnergy(model, :density)
    return Field(ρs_op; boundary_conditions=ρs_bcs)
end

#####
##### Tendency computation
#####

function AtmosphereModels.compute_thermodynamic_tendency!(model::PotentialTemperatureModel, common_args, tracer_transport_velocity)
    grid = model.grid
    arch = grid.architecture

    ρθ_args = (
        Val(1),
        model.forcing.ρθ,
        model.forcing.ρE,
        model.advection.ρθ,
        radiation_flux_divergence(model.radiation),
        common_args...,
        model.temperature)

    Gρθ = model.timestepper.Gⁿ.ρθ
    launch!(arch, grid, :xyz, compute_potential_temperature_tendency!, Gρθ, grid, ρθ_args)
    add_sedimentation_tendency!(Gρθ, model, tracer_transport_velocity)
    return nothing
end

@inline function potential_temperature_tendency(i, j, k, grid,
                                                id,
                                                ρθ_forcing,
                                                ρE_forcing,
                                                advection,
                                                radiation_flux_divergence_field,
                                                dynamics,
                                                formulation::LiquidIcePotentialTemperatureFormulation,
                                                constants,
                                                specific_prognostic_moisture,
                                                velocities,
                                                microphysics,
                                                microphysical_fields,
                                                closure,
                                                closure_fields,
                                                clock,
                                                model_fields,
                                                temperature_field)

    potential_temperature = formulation.potential_temperature
    ρ_field = dynamics_density(dynamics)                # coupling density ρᵈ (advection/diffusion carrier)
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants, microphysics, microphysical_fields)

    @inbounds begin
        ρᵈ = ρ_field[i, j, k]
        ρ = total_density(dynamics)[i, j, k]           # total ρ (mass fractions)
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        θ = potential_temperature[i, j, k]
        T = temperature_field[i, j, k]
    end

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    H = liquid_ice_enthalpy(T, q, constants)
    ∂θ∂h = potential_temperature_heating_response(θ, H)

    FρE = ρE_forcing(i, j, k, grid, clock, model_fields)
    div_ℐ = radiation_flux_divergence(i, j, k, grid, radiation_flux_divergence_field)
    Dt_ln_p = pressure_logarithm_tendency(i, j, k, grid, dynamics, constants, model_fields.w)

    return ( - div_ρUc(i, j, k, grid, advection, ρ_field, velocities, potential_temperature)
             + c_div_ρU(i, j, k, grid, dynamics, velocities, potential_temperature)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, potential_temperature, clock, model_fields, closure_buoyancy)
             + ρθ_forcing(i, j, k, grid, clock, model_fields)
             + ρᵈ / ρ * (FρE + div_ℐ) * ∂θ∂h # a volumetric heating Q changes h at Q / ρ, so ρᵈ θˡⁱ at (ρᵈ / ρ) Q ∂θˡⁱ/∂h
             + ρᵈ * expansion_work_source(θ, T, q, formulation.exner_exponent, constants) * Dt_ln_p) # expansion work ρᵈ (Rᵐ T − κ ĉᵖ Π θˡⁱ) / (ĉᵖ Π) D ln p / Dt
end

# Thermodynamic state at cell (i, j, k) from the prognostic fields: the total density (mass
# fractions), the prognostic moisture, and the microphysical state.
@inline function grid_thermodynamic_state(i, j, k, grid, formulation, dynamics,
                                          microphysics, microphysical_fields, specific_prognostic_moisture)
    @inbounds ρ = total_density(dynamics)[i, j, k]  # total ρ (mass fractions)
    @inbounds qᵛᵉ = specific_prognostic_moisture[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    return diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
end

# The enthalpy H = cᵖᵐ T − ℒˡ₀ qˡ − ℒⁱ₀ qⁱ = ĉᵖ Π θˡⁱ that defines θˡⁱ
@inline liquid_ice_enthalpy(T, q, constants) = mixture_heat_capacity(q, constants) * T - condensate_latent_energy(q, constants)

# The heating response ∂θˡⁱ/∂h = 1 / (ĉᵖ Π) at fixed pressure and composition, written θˡⁱ / H so that
# it is read off the stored temperature without evaluating the Exner function (a power, and a
# Newton inversion on the compressible core; NumericalEarth/Breeze.jl#1107). A state at absolute
# zero (θˡⁱ = H = 0, e.g. before `set!`) has no response.
@inline potential_temperature_heating_response(θ, H) = θ / ifelse(θ == 0, one(H), H)

# Expansion-work source. At fixed θˡⁱ and composition the definition ĉᵖ Π θˡⁱ = H gives
# cᵖᵐ dT = κ ĉᵖ Π θˡⁱ d ln p, whereas the first law requires cᵖᵐ dT = Rᵐ T d ln p for an adiabatic
# parcel. Adding (Rᵐ T − κ ĉᵖ Π θˡⁱ) / (ĉᵖ Π) D ln p / Dt to Dθˡⁱ/Dt closes the difference. With the
# vapor-equivalent exponent κ ĉᵖ = R̂, and the source is zero in unsaturated air and about
# 6 × 10⁵ qˡ J/kg per unit ln p in cloud; with the dry exponent it has a small clear-air part,
# (Rᵐ − κᵈ cᵖᵐ) T ≈ −0.23 qᵛ Rᵈ T per unit ln p.
#
# D ln p / Dt is estimated from hydrostatic balance, −g ρ w / p, with the density and pressure the
# dynamics close the thermodynamics on: the reference state on the anelastic core (where it is
# exact, since the thermodynamic pressure is pᵣ(z)) and the actual density and pressure on the
# compressible core. On the compressible core the exact rate is
# D ln p / Dt = (cᵖᵐ / cᵛᵐ) [−∇·u + D ln Rᵐ / Dt + Q / (cᵖᵐ T)], a possible follow-up; the adiabatic
# divergence form −(cᵖᵐ / cᵛᵐ) ∇·u alone double-counts heating-driven expansion.
@inline function expansion_work_source(θ, T, q, κ̂, constants)
    H = liquid_ice_enthalpy(T, q, constants)
    κ = exner_exponent(κ̂, q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    return potential_temperature_heating_response(θ, H) * (Rᵐ * T - κ * H)
end

@inline function pressure_logarithm_tendency(i, j, k, grid, dynamics, constants, w)
    g = constants.gravitational_acceleration
    @inbounds begin
        ρ = total_density(dynamics)[i, j, k]
        p = dynamics_pressure(dynamics)[i, j, k]
    end
    wᶜ = ℑzᵃᵃᶜ(i, j, k, grid, w)
    return - g * ρ * wᶜ / p
end

#####
##### Condensate content of ρθ for its sedimentation tendency
#####
#
# The content per unit falling mass of phase x is the derivative χˣ = ∇_q θˡⁱ · Δqˣ at fixed T
# and p along the composition increment Δqˣ of `sedimentation_composition_increment`: q̂ˣ − q̂ᵈ on
# the anelastic core, whose total density is fixed, q̂ˣ − q on the compressible core, whose total
# density falls with the condensate while the pressure p = (ρᵈ Rᵈ + ρᵛ Rᵛ) T does not. Losing
# condensate at this content leaves the temperature unchanged on either core. With
# θˡⁱ = H / (ĉᵖ Π), H = cᵖᵐ T − ℒˡ₀ qˡ − ℒⁱ₀ qⁱ and Π = (p / pˢᵗ)^κ,
#
#   χˣ = [ΔH − θˡⁱ Δ(ĉᵖ Π)] / (ĉᵖ Π) ,   Δ(ĉᵖ Π) = Π Δĉᵖ + ĉᵖ Π ln Π Δln κ ,
#
# with ΔH = Δcᵖ T − ΔHᶜ the enthalpy increment (`enthalpy_increment`), Δĉᵖ = (cᵖᵛ − cᵖᵈ) Δqᵗ, and
# Δln κ the change of the exponent's logarithm: zero for `DryExponent`, ΔR̂ / R̂ − Δĉᵖ / ĉᵖ for
# `VaporEquivalentExponent`. ΔH is the enthalpy the falling condensate carries relative to what its
# mass gives way to, −ℒˣ₀ / (ĉᵖ Π) to leading order; the other terms account for ĉᵖ and κ changing
# with the total water. ln Π is recovered from the stored temperature through Π = H / (ĉᵖ θˡⁱ),
# so every state type serves.
#
# The transported enthalpy and thermal response depend on the dynamics
# (`sedimentation_thermal_response`). χ remains a local composition derivative, not a
# transported quantity. These are instantaneous responses; multiplying them by a finite mass
# increment does not exactly reconstruct thermal energy.
#
# The explicit tendency is built from the state `update_state!` just diagnosed, in
# `update_state!` itself or at the start of an acoustic stage before anything moves θ or q, so
# the temperature field matches θ and q: the content reads that temperature and recovers the
# Exner function from ĉᵖ Π θˡⁱ = H, rather than inverting θ for T again (a Newton iteration on the
# compressible core) in this cell and both of its neighbors. `implicit_sedimentation_step!` runs
# after the tracers' solves have moved q, where the temperature field no longer matches the
# state, so it passes `nothing` (see `content_temperature`) and T is rediagnosed from (θ, q).
@inline function AtmosphereModels.condensate_content(i, j, k, grid, formulation::LiquidIcePotentialTemperatureFormulation,
                                                     dynamics, constants, microphysics, microphysical_fields,
                                                     specific_prognostic_moisture, temperature_field)
    @inbounds ρ = total_density(dynamics)[i, j, k]  # total ρ (mass fractions)
    @inbounds qᵛᵉ = specific_prognostic_moisture[i, j, k]
    @inbounds θ = formulation.potential_temperature[i, j, k]
    @inbounds T = temperature_field[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    return potential_temperature_condensate_content(dynamics, formulation.exner_exponent, constants, q, θ, T)
end

@inline function AtmosphereModels.condensate_content(i, j, k, grid, formulation::LiquidIcePotentialTemperatureFormulation,
                                                     dynamics, constants, microphysics, microphysical_fields,
                                                     specific_prognostic_moisture, ::Nothing)
    𝒰 = grid_thermodynamic_state(i, j, k, grid, formulation, dynamics,
                                 microphysics, microphysical_fields, specific_prognostic_moisture)
    T = temperature(𝒰, constants)
    return potential_temperature_condensate_content(dynamics, formulation.exner_exponent, constants,
                                                    𝒰.moisture_mass_fractions, 𝒰.potential_temperature, T)
end

# The content, transported enthalpies and thermal response of a cell with mass fractions `q`,
# potential temperature `θ` and temperature `T`, with ĉᵖ Π = H / θ
@inline function potential_temperature_condensate_content(dynamics, κ̂, constants, q, θ, T)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    H = liquid_ice_enthalpy(T, q, constants)
    Π = H / (ĉᵖ * θ)

    Δqˡ = sedimentation_composition_increment(dynamics, q, Val(:liquid))
    Δqⁱ = sedimentation_composition_increment(dynamics, q, Val(:ice))
    χˡ = potential_temperature_content(Δqˡ, κ̂, q, constants, T, θ, ĉᵖ, Π)
    χⁱ = potential_temperature_content(Δqⁱ, κ̂, q, constants, T, θ, ĉᵖ, Π)

    h, ∂θ∂h = sedimentation_thermal_response(dynamics, κ̂, q, constants, T, Π)
    return (; χ = (χˡ, χⁱ), h, ∂φ∂h = ∂θ∂h)
end

# `implicit_sedimentation_step!` rediagnoses T from the post-solve state (see above).
AtmosphereModels.content_temperature(::LiquidIcePotentialTemperatureFormulation, temperature_field) = nothing

# The derivative of θˡⁱ = H / (ĉᵖ Π) along one composition increment at fixed T and p
@inline function potential_temperature_content(Δq, κ̂, q, constants, T, θ, ĉᵖ, Π)
    ΔH = enthalpy_increment(Δq, constants, T)
    Δqᵗ = - dry_air_increment(Δq)
    Δĉᵖ = (constants.vapor.heat_capacity - constants.dry_air.heat_capacity) * Δqᵗ
    Δ_ĉᵖΠ = Π * (Δĉᵖ + exponent_increment(κ̂, q, constants, Δqᵗ, ĉᵖ, Π))
    return (ΔH - θ * Δ_ĉᵖΠ) / (ĉᵖ * Π)
end

# ĉᵖ ln Π Δln κ: the part of Δ(ĉᵖ Π) / Π due to the exponent changing along Δqᵗ
@inline exponent_increment(::DryExponent, q, constants, Δqᵗ, ĉᵖ, Π) = 0

@inline function exponent_increment(::VaporEquivalentExponent, q, constants, Δqᵗ, ĉᵖ, Π)
    R̂ = vapor_equivalent_gas_constant(q, constants)
    ΔR̂ = (vapor_gas_constant(constants) - dry_air_gas_constant(constants)) * Δqᵗ
    Δĉᵖ = (constants.vapor.heat_capacity - constants.dry_air.heat_capacity) * Δqᵗ
    return ĉᵖ * log(Π) * (ΔR̂ / R̂ - Δĉᵖ / ĉᵖ)
end

"""
$(TYPEDSIGNATURES)

Return the transported liquid/ice enthalpies and local potential-temperature heating response.
With a fixed total density (the default) the departed condensate mass is made up by dry air at
prescribed pressure: transport `hˣ − hᵈ`, the enthalpy change along the composition increment,
and respond with `1 / (ĉᵖ Π)`. `CompressibleDynamics` transports phase enthalpy `hˣ` and uses
`β_cv` for isolated sedimentation at fixed volume and gas partial densities, without phase
change or resolved motion. These instantaneous responses do not reconstruct finite-step energy.

This callback serves the liquid-ice potential-temperature formulation. The compressible
`temperature_and_pressure` diagnosis supports that formulation, not `StaticEnergyFormulation`;
this sedimentation correction does not add a compressible static-energy diagnosis.
"""
@inline function sedimentation_thermal_response(dynamics, κ̂, q, constants, T, Π)
    hˡ = enthalpy_increment(sedimentation_composition_increment(dynamics, q, Val(:liquid)), constants, T)
    hⁱ = enthalpy_increment(sedimentation_composition_increment(dynamics, q, Val(:ice)), constants, T)
    return (hˡ, hⁱ), 1 / (vapor_equivalent_heat_capacity(q, constants) * Π)
end

#####
##### Set thermodynamic variables
#####

AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Union{Val{:ρθ}, Val{:ρθˡⁱ}}, value) =
    set!(model.formulation.potential_temperature_density, value)

function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Union{Val{:θ}, Val{:θˡⁱ}}, value)
    set!(model.formulation.potential_temperature, value)
    ρ = dynamics_density(model.dynamics)
    θˡⁱ = model.formulation.potential_temperature
    set!(model.formulation.potential_temperature_density, ρ * θˡⁱ)
    return nothing
end

# Setting from static energy
function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Val{:s}, value)
    formulation = model.formulation
    s = model.temperature # scratch space
    set!(s, value)

    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, :xyz,
            _potential_temperature_from_energy!,
            formulation.potential_temperature_density,
            formulation.potential_temperature,
            grid,
            s,
            specific_prognostic_moisture(model),
            model.dynamics,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants,
            formulation.exner_exponent)

    return nothing
end

function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Val{:ρs}, value)
    ρs = model.temperature # scratch space
    set!(ρs, value)
    ρ = dynamics_density(model.dynamics)
    return set_thermodynamic_variable!(model, Val(:s), ρs / ρ)
end

@kernel function _potential_temperature_from_energy!(potential_temperature_density,
                                                     potential_temperature,
                                                     grid,
                                                     specific_energy,
                                                     specific_prognostic_moisture,
                                                     dynamics,
                                                     microphysics,
                                                     microphysical_fields,
                                                     constants,
                                                     κ̂)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        pᵣ = dynamics_pressure(dynamics)[i, j, k]
        ρ = total_density(dynamics)[i, j, k]      # total ρ (mass fractions)
        ρᵈ = dynamics_density(dynamics)[i, j, k]  # coupling density ρᵈ (ρθ = ρᵈθ)
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        s = specific_energy[i, j, k]
    end

    z = znode(i, j, k, grid, c, c, c)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    𝒰s₀ = StaticEnergyState(s, q, z, pᵣ)
    μ = AtmosphereModels.extract_microphysical_prognostics(i, j, k, microphysics, microphysical_fields)
    𝒰s₁ = maybe_adjust_thermodynamic_state(𝒰s₀, microphysics, qᵛᵉ, constants, μ, ρ)
    T = temperature(𝒰s₁, constants)

    pˢᵗ = standard_pressure(dynamics)
    q₁ = 𝒰s₁.moisture_mass_fractions
    𝒰θ = LiquidIcePotentialTemperatureState(zero(T), q₁, pˢᵗ, pᵣ, κ̂)
    𝒰θ = with_temperature(𝒰θ, T, constants)
    θ = 𝒰θ.potential_temperature
    @inbounds potential_temperature[i, j, k] = θ
    @inbounds potential_temperature_density[i, j, k] = ρᵈ * θ
end

#####
##### Setting temperature directly
#####

"""
    $(TYPEDSIGNATURES)

Set the thermodynamic state from in-situ temperature ``T``.

The temperature is converted to liquid-ice potential temperature `θˡⁱ` using
the relation between ``T`` and `θˡⁱ`` that accounts for the moisture distribution.

For unsaturated air (no condensate), this simplifies to ``θ = T / Π`` where
``Π`` is the Exner function.
"""
function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Val{:T}, value)
    T_field = model.temperature # use temperature field as scratch/storage
    set!(T_field, value)

    grid = model.grid
    arch = grid.architecture
    formulation = model.formulation

    launch!(arch, grid, :xyz,
            _potential_temperature_from_temperature!,
            formulation.potential_temperature_density,
            formulation.potential_temperature,
            grid,
            T_field,
            specific_prognostic_moisture(model),
            model.dynamics,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants,
            formulation.exner_exponent)

    return nothing
end

@kernel function _potential_temperature_from_temperature!(potential_temperature_density,
                                                          potential_temperature,
                                                          grid,
                                                          temperature_field,
                                                          specific_prognostic_moisture,
                                                          dynamics,
                                                          microphysics,
                                                          microphysical_fields,
                                                          constants,
                                                          κ̂)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = total_density(dynamics)[i, j, k]      # total ρ (mass fractions)
        ρᵈ = dynamics_density(dynamics)[i, j, k]  # coupling density ρᵈ (ρθ = ρᵈθ)
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        T = temperature_field[i, j, k]
    end

    # Get moisture fractions (vapor only for unsaturated air)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    pᵣ = pressure_from_density_temperature(i, j, k, grid, dynamics, ρ, T, q, constants)

    # Convert temperature to potential temperature using the inverse of the T(θ) relation
    pˢᵗ = standard_pressure(dynamics)
    𝒰₀ = LiquidIcePotentialTemperatureState(zero(T), q, pˢᵗ, pᵣ, κ̂)
    𝒰₁ = with_temperature(𝒰₀, T, constants)
    θ = 𝒰₁.potential_temperature

    @inbounds potential_temperature[i, j, k] = θ
    @inbounds potential_temperature_density[i, j, k] = ρᵈ * θ
end
