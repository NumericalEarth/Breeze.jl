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
energy flux boundary conditions by multiplying by the mixture heat capacity `cᵖᵐ`.
"""
function AtmosphereModels.static_energy_density(model::PotentialTemperatureModel)
    ρθ = model.formulation.potential_temperature_density
    ρθ_bcs = ρθ.boundary_conditions

    # Convert θ BCs to energy BCs
    ρs_bcs = theta_to_energy_bcs(ρθ_bcs)

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
        values(model.sedimentation),
        tracer_transport_velocity,
        common_args...,
        model.temperature)

    Gρθ = model.timestepper.Gⁿ.ρθ
    launch!(arch, grid, :xyz, compute_potential_temperature_tendency!, Gρθ, grid, ρθ_args)
    return nothing
end

@inline function potential_temperature_tendency(i, j, k, grid,
                                                id,
                                                ρθ_forcing,
                                                ρE_forcing,
                                                advection,
                                                radiation_flux_divergence_field,
                                                sedimenting_condensates,
                                                tracer_transport_velocity,
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
    𝒰 = grid_thermodynamic_state(i, j, k, grid, formulation, dynamics,
                                 microphysics, microphysical_fields, specific_prognostic_moisture)
    Π = exner_function(𝒰, constants)
    cᵖᵐ = mixture_heat_capacity(𝒰.moisture_mass_fractions, constants)
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants)

    FρE = ρE_forcing(i, j, k, grid, clock, model_fields)
    div_ℐ = radiation_flux_divergence(i, j, k, grid, radiation_flux_divergence_field)

    return ( - div_ρUc(i, j, k, grid, advection, ρ_field, velocities, potential_temperature)
             + c_div_ρU(i, j, k, grid, dynamics, velocities, potential_temperature)
             + sedimentation_tendency(i, j, k, grid, sedimenting_condensates, tracer_transport_velocity,
                                      formulation, dynamics, constants, microphysics, microphysical_fields,
                                      specific_prognostic_moisture, temperature_field)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, potential_temperature, clock, model_fields, closure_buoyancy)
             + ρθ_forcing(i, j, k, grid, clock, model_fields)
             + (FρE + div_ℐ) / (cᵖᵐ * Π)
    )
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

#####
##### Condensate content of ρθ for its sedimentation tendency
#####
#
# The content per unit falling mass of phase x is the derivative χˣ = ∇_q θˡⁱ · Δqˣ at fixed T
# and p along the composition increment Δqˣ of `sedimentation_composition_increment`: q̂ˣ − q̂ᵈ on
# the anelastic core, whose total density is fixed, q̂ˣ − q on the compressible core, whose total
# density falls with the condensate while the pressure p = (ρᵈ Rᵈ + ρᵛ Rᵛ) T does not. Losing
# condensate at this content leaves the temperature unchanged on either core. With
# T = Π θ + Λ / cᵖᵐ, Λ = ℒˡᵣ qˡ + ℒⁱᵣ qⁱ, Π = (p / pˢᵗ)^(Rᵐ / cᵖᵐ), and Δcᵖ, ΔR, ΔΛ the changes of
# cᵖᵐ, Rᵐ and Λ along Δqˣ,
#
#   χˣ = −(ΔΛ − D Δcᵖ) / (cᵖᵐ Π) + θ lnΠ (Δcᵖ / cᵖᵐ − ΔR / Rᵐ) ,
#
# with D = Λ / cᵖᵐ = T − Π θ the latent deficit. The first term is the deficit the falling
# condensate carries, −ℒˣᵣ / (cᵖᵐ Π) to leading order; the second accounts for the heat capacity
# and gas constant of the mixture changing with the composition (lnΠ = (Rᵐ / cᵖᵐ) ln(p / pˢᵗ) is
# written through Π so that every state type that defines an Exner function serves).
#
# The transported enthalpy and thermal response depend on the dynamics
# (`sedimentation_thermal_response`). χ remains a local composition derivative, not a
# transported quantity. These are instantaneous responses; multiplying them by a finite mass
# increment does not exactly reconstruct thermal energy. The temperature is rediagnosed from the
# state, T = Π θ + D, so the temperature field is unused.
@inline function AtmosphereModels.condensate_content(i, j, k, grid, formulation::LiquidIcePotentialTemperatureFormulation,
                                                     dynamics, constants, microphysics, microphysical_fields,
                                                     specific_prognostic_moisture, temperature_field)
    𝒰 = grid_thermodynamic_state(i, j, k, grid, formulation, dynamics,
                                 microphysics, microphysical_fields, specific_prognostic_moisture)
    q = 𝒰.moisture_mass_fractions
    θ = 𝒰.potential_temperature
    Π = exner_function(𝒰, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    D = (constants.liquid.reference_latent_heat * q.liquid + constants.ice.reference_latent_heat * q.ice) / cᵖᵐ
    θlnΠ = θ * log(Π)

    Δqˡ = sedimentation_composition_increment(dynamics, q, Val(:liquid))
    Δqⁱ = sedimentation_composition_increment(dynamics, q, Val(:ice))
    χˡ = potential_temperature_content(Δqˡ, constants, cᵖᵐ, Rᵐ, D, Π, θlnΠ)
    χⁱ = potential_temperature_content(Δqⁱ, constants, cᵖᵐ, Rᵐ, D, Π, θlnΠ)

    T = Π * θ + D
    h, ∂θ∂h = sedimentation_thermal_response(dynamics, q, constants, T, Π)
    return (; χ = (χˡ, χⁱ), h, ∂φ∂h = ∂θ∂h)
end

# The derivative of θˡⁱ above along one composition increment
@inline function potential_temperature_content(Δq, constants, cᵖᵐ, Rᵐ, D, Π, θlnΠ)
    Δcᵖ = heat_capacity_increment(Δq, constants)
    ΔR = gas_constant_increment(Δq, constants)
    ΔΛ = latent_heat_increment(Δq, constants)
    return -(ΔΛ - D * Δcᵖ) / (cᵖᵐ * Π) + θlnΠ * (Δcᵖ / cᵖᵐ - ΔR / Rᵐ)
end

"""
$(TYPEDSIGNATURES)

Return the transported liquid/ice enthalpies and local potential-temperature heating response.
With a fixed total density (the default) the departed condensate mass is made up by dry air at
prescribed pressure: transport `hˣ − hᵈ`, the enthalpy change along the composition increment,
and respond with `1 / (cᵖᵐ Π)`. `CompressibleDynamics` transports phase enthalpy `hˣ` and uses
`β_cv` for isolated sedimentation at fixed volume and gas partial densities, without phase
change or resolved motion. These instantaneous responses do not reconstruct finite-step energy.

This callback serves the liquid-ice potential-temperature formulation. The compressible
`temperature_and_pressure` diagnosis supports that formulation, not `StaticEnergyFormulation`;
this sedimentation correction does not add a compressible static-energy diagnosis.
"""
@inline function sedimentation_thermal_response(dynamics, q, constants, T, Π)
    hˡ = enthalpy_increment(sedimentation_composition_increment(dynamics, q, Val(:liquid)), constants, T)
    hⁱ = enthalpy_increment(sedimentation_composition_increment(dynamics, q, Val(:ice)), constants, T)
    return (hˡ, hⁱ), 1 / (mixture_heat_capacity(q, constants) * Π)
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
            model.thermodynamic_constants)

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
                                                     constants)
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
    𝒰s₁ = maybe_adjust_thermodynamic_state(𝒰s₀, microphysics, qᵛᵉ, constants)
    T = temperature(𝒰s₁, constants)

    pˢᵗ = standard_pressure(dynamics)
    q₁ = 𝒰s₁.moisture_mass_fractions
    𝒰θ = LiquidIcePotentialTemperatureState(zero(T), q₁, pˢᵗ, pᵣ)
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
            model.thermodynamic_constants)

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
                                                          constants)
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
    𝒰₀ = LiquidIcePotentialTemperatureState(zero(T), q, pˢᵗ, pᵣ)
    𝒰₁ = with_temperature(𝒰₀, T, constants)
    θ = 𝒰₁.potential_temperature

    @inbounds potential_temperature[i, j, k] = θ
    @inbounds potential_temperature_density[i, j, k] = ρᵈ * θ
end
