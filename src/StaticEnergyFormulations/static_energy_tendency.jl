using Breeze.AtmosphereModels.Diagnostics: Diagnostics
using Breeze.AtmosphereModels: AtmosphereModel

using Oceananigans.Fields: set!
using Breeze.Thermodynamics: temperature

const StaticEnergyModel = AtmosphereModel{<:Any, <:StaticEnergyFormulation}

#####
##### Helper accessors
#####

AtmosphereModels.liquid_ice_potential_temperature(model::StaticEnergyModel) = Diagnostics.LiquidIcePotentialTemperature(model, :specific)
AtmosphereModels.liquid_ice_potential_temperature_density(model::StaticEnergyModel) = Diagnostics.LiquidIcePotentialTemperature(model, :density)
AtmosphereModels.static_energy(model::StaticEnergyModel) = model.formulation.specific_energy
AtmosphereModels.static_energy_density(model::StaticEnergyModel) = model.formulation.energy_density

#####
##### Tendency computation
#####

function AtmosphereModels.compute_thermodynamic_tendency!(model::StaticEnergyModel, common_args)
    grid = model.grid
    arch = grid.architecture

    ρe_args = (
        Val(1),
        model.forcing.ρe,
        model.advection.ρe,
        common_args...,
        model.temperature)

    Gρe = model.timestepper.Gⁿ.ρe
    launch!(arch, grid, :xyz, compute_static_energy_tendency!, Gρe, grid, ρe_args)
    return nothing
end

@inline function static_energy_tendency(i, j, k, grid,
                                        id,
                                        ρe_forcing,
                                        advection,
                                        dynamics,
                                        formulation,
                                        constants,
                                        specific_moisture,
                                        velocities,
                                        microphysics,
                                        microphysical_fields,
                                        closure,
                                        closure_fields,
                                        clock,
                                        model_fields,
                                        temperature_field)

    specific_energy = formulation.specific_energy
    ρ_field = dynamics_density(dynamics)
    @inbounds ρ = ρ_field[i, j, k]
    @inbounds qᵗ = specific_moisture[i, j, k]

    # Compute moisture fractions first
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, microphysical_fields)
    𝒰 = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

    # Compute the buoyancy flux term, ρᵣ w b
    buoyancy_flux = ℑzᵃᵃᶜ(i, j, k, grid, w_buoyancy_forceᶜᶜᶠ,
                          velocities.w, dynamics, temperature_field, specific_moisture,
                          microphysics, microphysical_fields, constants)

    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants)
    return ( - div_ρUc(i, j, k, grid, advection, ρ_field, velocities, specific_energy)
             + c_div_ρU(i, j, k, grid, dynamics, velocities, specific_energy)
             + buoyancy_flux
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, specific_energy, clock, model_fields, closure_buoyancy)
             + grid_microphysical_tendency(i, j, k, grid, microphysics, Val(:ρe), ρ, microphysical_fields, 𝒰, constants, velocities)
             + ρe_forcing(i, j, k, grid, clock, model_fields))
end

#####
##### Set thermodynamic variables
#####

AtmosphereModels.set_thermodynamic_variable!(model::StaticEnergyModel, ::Val{:ρe}, value) =
    set!(model.formulation.energy_density, value)

function AtmosphereModels.set_thermodynamic_variable!(model::StaticEnergyModel, ::Val{:e}, value)
    set!(model.formulation.specific_energy, value)
    ρ = dynamics_density(model.dynamics)
    e = model.formulation.specific_energy
    set!(model.formulation.energy_density, ρ * e)
    return nothing
end

# Setting :θ (potential temperature)
const PotentialTemperatureNames = Union{Val{:θ}, Val{:θˡⁱ}}

function AtmosphereModels.set_thermodynamic_variable!(model::StaticEnergyModel, ::PotentialTemperatureNames, value)
    formulation = model.formulation
    θ = model.temperature # scratch space
    set!(θ, value)

    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, :xyz,
            _energy_density_from_potential_temperature!,
            formulation.energy_density,
            formulation.specific_energy,
            grid,
            θ,
            model.specific_moisture,
            model.dynamics,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    return nothing
end

@kernel function _energy_density_from_potential_temperature!(energy_density,
                                                             specific_energy,
                                                             grid,
                                                             potential_temperature,
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
        θ = potential_temperature[i, j, k]
    end

    pˢᵗ = standard_pressure(dynamics)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    𝒰θ₀ = LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, pᵣ)
    𝒰θ₁ = maybe_adjust_thermodynamic_state(𝒰θ₀, microphysics, qᵗ, constants)
    T = temperature(𝒰θ₁, constants)

    z = znode(i, j, k, grid, c, c, c)
    q₁ = 𝒰θ₁.moisture_mass_fractions
    𝒰e₀ = StaticEnergyState(zero(T), q₁, z, pᵣ)
    𝒰e₁ = with_temperature(𝒰e₀, T, constants)
    e = 𝒰e₁.static_energy

    @inbounds specific_energy[i, j, k] = e
    @inbounds energy_density[i, j, k] = ρᵣ * e
end

#####
##### Setting temperature directly
#####

"""
    $(TYPEDSIGNATURES)

Set the thermodynamic state from temperature ``T``.

The temperature is converted to static energy ``e`` using the relation:

```math
e = cᵖᵐ T + g z - ℒˡ qˡ - ℒⁱ qⁱ .
```
"""
function AtmosphereModels.set_thermodynamic_variable!(model::StaticEnergyModel, ::Val{:T}, value)
    T_field = model.temperature # use temperature field as scratch/storage
    set!(T_field, value)

    grid = model.grid
    arch = grid.architecture
    formulation = model.formulation

    launch!(arch, grid, :xyz,
            _energy_density_from_temperature!,
            formulation.energy_density,
            formulation.specific_energy,
            grid,
            T_field,
            model.specific_moisture,
            model.dynamics,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    return nothing
end

@kernel function _energy_density_from_temperature!(energy_density,
                                                   specific_energy,
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

    # Convert temperature to static energy
    z = znode(i, j, k, grid, c, c, c)
    𝒰₀ = StaticEnergyState(zero(T), q, z, pᵣ)
    𝒰₁ = with_temperature(𝒰₀, T, constants)

    e = 𝒰₁.static_energy
    @inbounds specific_energy[i, j, k] = e
    @inbounds energy_density[i, j, k] = ρᵣ * e
end
