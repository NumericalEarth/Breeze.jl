using Breeze.AtmosphereModels.Diagnostics: Diagnostics
using Breeze.AtmosphereModels: AtmosphereModel

using Oceananigans.Fields: Field, set!
using Breeze.Thermodynamics: temperature
using Breeze.BoundaryConditions: theta_to_energy_bcs, materialize_atmosphere_field_bcs

const PotentialTemperatureModel = AtmosphereModel{<:Any, <:LiquidIcePotentialTemperatureFormulation}

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
    ρe_bcs = theta_to_energy_bcs(ρθ_bcs)

    # Regularize the converted BCs (populate microphysics, constants, side)
    loc = (Center(), Center(), Center())
    ρe_bcs = materialize_atmosphere_field_bcs(ρe_bcs, loc, model.grid, model.dynamics, model.microphysics,
                                              nothing, model.thermodynamic_constants, nothing, nothing, nothing)

    # Create the energy density operation and wrap in a Field with proper BCs
    ρe_op = Diagnostics.StaticEnergy(model, :density)
    return Field(ρe_op; boundary_conditions=ρe_bcs)
end

#####
##### Tendency computation
#####

function AtmosphereModels.compute_thermodynamic_tendency!(model::PotentialTemperatureModel, common_args)
    grid = model.grid
    arch = grid.architecture

    ρθ_args = (
        Val(1),
        model.forcing.ρθ,
        model.forcing.ρe,
        model.advection.ρθ,
        radiation_flux_divergence(model.radiation),
        common_args...)

    Gρθ = model.timestepper.Gⁿ.ρθ
    launch!(arch, grid, :xyz, compute_potential_temperature_tendency!, Gρθ, grid, ρθ_args)
    return nothing
end

@inline function potential_temperature_tendency(i, j, k, grid,
                                                id,
                                                ρθ_forcing,
                                                ρe_forcing,
                                                advection,
                                                radiation_flux_divergence_field,
                                                dynamics,
                                                formulation::LiquidIcePotentialTemperatureFormulation,
                                                constants,
                                                specific_moisture,
                                                velocities,
                                                microphysics,
                                                microphysical_fields,
                                                closure,
                                                closure_fields,
                                                clock,
                                                model_fields)

    potential_temperature = formulation.potential_temperature
    ρ_field = dynamics_density(dynamics)
    @inbounds ρ = ρ_field[i, j, k]
    @inbounds qᵗ = specific_moisture[i, j, k]

    # Compute moisture fractions first
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, microphysical_fields)
    𝒰 = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

    Π = exner_function(𝒰, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants)

    Fρe = ρe_forcing(i, j, k, grid, clock, model_fields)
    div_ℐ = radiation_flux_divergence(i, j, k, grid, radiation_flux_divergence_field)

    return ( - div_ρUc(i, j, k, grid, advection, ρ_field, velocities, potential_temperature)
             + c_div_ρU(i, j, k, grid, dynamics, velocities, potential_temperature)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, potential_temperature, clock, model_fields, closure_buoyancy)
             + grid_microphysical_tendency(i, j, k, grid, microphysics, Val(:ρθ), ρ, microphysical_fields, 𝒰, constants, velocities)
             + ρθ_forcing(i, j, k, grid, clock, model_fields)
             + (Fρe + div_ℐ) / (cᵖᵐ * Π)
    )
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
function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Val{:e}, value)
    formulation = model.formulation
    e = model.temperature # scratch space
    set!(e, value)

    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, :xyz,
            _potential_temperature_from_energy!,
            formulation.potential_temperature_density,
            formulation.potential_temperature,
            grid,
            e,
            model.specific_moisture,
            model.dynamics,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    return nothing
end

function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Val{:ρe}, value)
    ρe = model.temperature # scratch space
    set!(ρe, value)
    ρ = dynamics_density(model.dynamics)
    return set_thermodynamic_variable!(model, Val(:e), ρe / ρ)
end

@kernel function _potential_temperature_from_energy!(potential_temperature_density,
                                                     potential_temperature,
                                                     grid,
                                                     specific_energy,
                                                     specific_moisture,
                                                     dynamics,
                                                     microphysics,
                                                     microphysical_fields,
                                                     constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        pᵣ = dynamics_pressure(dynamics)[i, j, k]
        ρᵣ = dynamics_density(dynamics)[i, j, k]
        qᵗ = specific_moisture[i, j, k]
        e = specific_energy[i, j, k]
    end

    z = znode(i, j, k, grid, c, c, c)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    𝒰e₀ = StaticEnergyState(e, q, z, pᵣ)
    𝒰e₁ = maybe_adjust_thermodynamic_state(𝒰e₀, microphysics, qᵗ, constants)
    T = temperature(𝒰e₁, constants)

    pˢᵗ = standard_pressure(dynamics)
    q₁ = 𝒰e₁.moisture_mass_fractions
    𝒰θ = LiquidIcePotentialTemperatureState(zero(T), q₁, pˢᵗ, pᵣ)
    𝒰θ = with_temperature(𝒰θ, T, constants)
    θ = 𝒰θ.potential_temperature
    @inbounds potential_temperature[i, j, k] = θ
    @inbounds potential_temperature_density[i, j, k] = ρᵣ * θ
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
            model.specific_moisture,
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
                                                          specific_moisture,
                                                          dynamics,
                                                          microphysics,
                                                          microphysical_fields,
                                                          constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        pᵣ = dynamics_pressure(dynamics)[i, j, k]
        ρᵣ = dynamics_density(dynamics)[i, j, k]
        qᵗ = specific_moisture[i, j, k]
        T = temperature_field[i, j, k]
    end

    # Get moisture fractions (vapor only for unsaturated air)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)

    # Convert temperature to potential temperature using the inverse of the T(θ) relation
    pˢᵗ = standard_pressure(dynamics)
    𝒰₀ = LiquidIcePotentialTemperatureState(zero(T), q, pˢᵗ, pᵣ)
    𝒰₁ = with_temperature(𝒰₀, T, constants)
    θ = 𝒰₁.potential_temperature

    @inbounds potential_temperature[i, j, k] = θ
    @inbounds potential_temperature_density[i, j, k] = ρᵣ * θ
end
