using Breeze.AtmosphereModels.Diagnostics: Diagnostics
using Breeze.AtmosphereModels: AtmosphereModel, specific_prognostic_moisture

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

    ρs_args = (
        Val(1),
        model.forcing.ρs,
        model.advection.ρs,
        radiation_flux_divergence(model.radiation),
        model.sedimentation_constituents,
        common_args...,
        model.temperature)

    Gρs = model.timestepper.Gⁿ.ρs
    launch!(arch, grid, :xyz, compute_static_energy_tendency!, Gρs, grid, ρs_args)
    return nothing
end

@inline function static_energy_tendency(i, j, k, grid,
                                        id,
                                        ρs_forcing,
                                        advection,
                                        radiation_flux_divergence_field,
                                        sedimenting_constituents,
                                        dynamics,
                                        formulation,
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

    specific_energy = formulation.specific_energy
    ρ_field = dynamics_density(dynamics)

    # Compute the buoyancy flux term, ρᵣ w b
    buoyancy_flux = ℑzᵃᵃᶜ(i, j, k, grid, w_buoyancy_forceᶜᶜᶠ,
                          velocities.w, dynamics, temperature_field, specific_prognostic_moisture,
                          microphysics, microphysical_fields, constants)

    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants)
    return ( - div_ρUc(i, j, k, grid, advection, ρ_field, velocities, specific_energy)
             + c_div_ρU(i, j, k, grid, dynamics, velocities, specific_energy)
             - buoyancy_flux
             - condensate_sedimentation_divergence(i, j, k, grid, sedimenting_constituents, velocities.w, dynamics,
                                                   energy_condensate_content, constants, temperature_field)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, specific_energy, clock, model_fields, closure_buoyancy)
             + ρs_forcing(i, j, k, grid, clock, model_fields)
             + radiation_flux_divergence(i, j, k, grid, radiation_flux_divergence_field))
end

#####
##### Sedimentation transport of the condensate part of ρs
#####
#
# The content per unit falling mass of phase x is χˣ = ∂s/∂qˣ at fixed T, with the dry mass
# fraction qᵈ = 1 − qᵛ − qˡ − qⁱ taking up the departed mass: sedimentation alone must not
# change the temperature. From s = cᵖᵐ T + g z − ℒˡᵣ qˡ − ℒⁱᵣ qⁱ,
#
#   χˣ = (cˣ − cᵖᵈ) T − ℒˣᵣ ,
#
# the latent deficit plus the sensible heat of the condensate relative to the dry air that
# replaces it; the geopotential is independent of the composition and drops out. This is exact
# for the anelastic core (fixed total density) and accurate to O(q) for the compressible core,
# where the whole mixture rather than dry air takes up the departed mass. The frictional heating
# from the fall (g wˣ qˣ) is neglected. The shared `condensate_sedimentation_divergence`
# evaluates the content in each flux's upwind cell and owns the discretization.
@inline function energy_condensate_content(i, j, k, grid, constants, temperature_field)
    @inbounds T = temperature_field[i, j, k]
    cᵖᵈ = constants.dry_air.heat_capacity
    χˡ = (constants.liquid.heat_capacity - cᵖᵈ) * T - constants.liquid.reference_latent_heat
    χⁱ = (constants.ice.heat_capacity - cᵖᵈ) * T - constants.ice.reference_latent_heat
    return χˡ, χⁱ
end

#####
##### Set thermodynamic variables
#####

AtmosphereModels.set_thermodynamic_variable!(model::StaticEnergyModel, ::Val{:ρs}, value) =
    set!(model.formulation.energy_density, value)

function AtmosphereModels.set_thermodynamic_variable!(model::StaticEnergyModel, ::Val{:s}, value)
    set!(model.formulation.specific_energy, value)
    ρ = dynamics_density(model.dynamics)
    s = model.formulation.specific_energy
    set!(model.formulation.energy_density, ρ * s)
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
            specific_prognostic_moisture(model),
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
                                                             specific_prognostic_moisture,
                                                             dynamics,
                                                             microphysics,
                                                             microphysical_fields,
                                                             constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        pᵣ = dynamics_pressure(dynamics)[i, j, k]
        ρ = total_density(dynamics)[i, j, k]      # total ρ (mass fractions)
        ρᵈ = dynamics_density(dynamics)[i, j, k]  # coupling density ρᵈ (ρs = ρᵈs)
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        θ = potential_temperature[i, j, k]
    end

    pˢᵗ = standard_pressure(dynamics)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    𝒰θ₀ = LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, pᵣ)
    𝒰θ₁ = maybe_adjust_thermodynamic_state(𝒰θ₀, microphysics, qᵛᵉ, constants)
    T = temperature(𝒰θ₁, constants)

    z = znode(i, j, k, grid, c, c, c)
    q₁ = 𝒰θ₁.moisture_mass_fractions
    𝒰s₀ = StaticEnergyState(zero(T), q₁, z, pᵣ)
    𝒰s₁ = with_temperature(𝒰s₀, T, constants)
    s = 𝒰s₁.static_energy

    @inbounds specific_energy[i, j, k] = s
    @inbounds energy_density[i, j, k] = ρᵈ * s
end

#####
##### Setting temperature directly
#####

"""
    $(TYPEDSIGNATURES)

Set the thermodynamic state from temperature ``T``.

The temperature is converted to static energy ``s`` using the relation:

```math
s = cᵖᵐ T + g z - ℒˡ qˡ - ℒⁱ qⁱ .
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
            specific_prognostic_moisture(model),
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
                                                   specific_prognostic_moisture,
                                                   dynamics,
                                                   microphysics,
                                                   microphysical_fields,
                                                   constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = total_density(dynamics)[i, j, k]      # total ρ (mass fractions)
        ρᵈ = dynamics_density(dynamics)[i, j, k]  # coupling density ρᵈ (ρs = ρᵈs)
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        T = temperature_field[i, j, k]
    end

    # Get moisture fractions (vapor only for unsaturated air)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    pᵣ = pressure_from_density_temperature(i, j, k, grid, dynamics, ρ, T, q, constants)

    # Convert temperature to static energy
    z = znode(i, j, k, grid, c, c, c)
    𝒰₀ = StaticEnergyState(zero(T), q, z, pᵣ)
    𝒰₁ = with_temperature(𝒰₀, T, constants)

    s = 𝒰₁.static_energy
    @inbounds specific_energy[i, j, k] = s
    @inbounds energy_density[i, j, k] = ρᵈ * s
end
