using ..Advection: div_ρUc
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, with_temperature, exner_function, mixture_heat_capacity
using Oceananigans: Oceananigans
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!

struct LiquidIcePotentialTemperatureThermodynamics{F, T}
    potential_temperature_density :: F  # ρθ (prognostic)
    potential_temperature :: T          # θ = ρθ / ρᵣ (diagnostic)
end

Adapt.adapt_structure(to, thermo::LiquidIcePotentialTemperatureThermodynamics) =
    LiquidIcePotentialTemperatureThermodynamics(adapt(to, thermo.potential_temperature_density),
                                       adapt(to, thermo.potential_temperature))

function BoundaryConditions.fill_halo_regions!(thermo::LiquidIcePotentialTemperatureThermodynamics)
    fill_halo_regions!(thermo.potential_temperature_density)
    fill_halo_regions!(thermo.potential_temperature)
    return nothing
end

const APTF = AnelasticFormulation{<:LiquidIcePotentialTemperatureThermodynamics}

prognostic_field_names(formulation::APTF) = tuple(:ρθ)
additional_field_names(formulation::APTF) = tuple(:θ)
thermodynamic_density_name(::APTF) = :ρθ
thermodynamic_density(formulation::APTF) = formulation.thermodynamics.potential_temperature_density
Oceananigans.fields(formulation::APTF) = (; θ=formulation.thermodynamics.potential_temperature)
Oceananigans.prognostic_fields(formulation::APTF) = (; ρθ=formulation.thermodynamics.potential_temperature_density)

function materialize_thermodynamics(::Val{:LiquidIcePotentialTemperature}, grid, boundary_conditions)
    potential_temperature_density = CenterField(grid, boundary_conditions=boundary_conditions.ρθ)
    potential_temperature = CenterField(grid) # θ = ρθ / ρᵣ (diagnostic)
    return LiquidIcePotentialTemperatureThermodynamics(potential_temperature_density, potential_temperature)
end

function compute_auxiliary_thermodynamic_variables!(formulation::APTF, i, j, k, grid)
    @inbounds begin
        ρᵣ = formulation.reference_state.density[i, j, k]
        ρθ = formulation.thermodynamics.potential_temperature_density[i, j, k]
        formulation.thermodynamics.potential_temperature[i, j, k] = ρθ / ρᵣ
    end
    return nothing
end

function diagnose_thermodynamic_state(i, j, k, grid, formulation::APTF,
                                      microphysics,
                                      microphysical_fields,
                                      constants,
                                      specific_moisture)
  
    θ = @inbounds formulation.thermodynamics.potential_temperature[i, j, k]
    pᵣ = @inbounds formulation.reference_state.pressure[i, j, k]
    ρᵣ = @inbounds formulation.reference_state.density[i, j, k]
    p₀ = formulation.reference_state.surface_pressure
    qᵗ = @inbounds specific_moisture[i, j, k]

    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)

    return LiquidIcePotentialTemperatureState(θ, q, p₀, pᵣ)
end

function collect_prognostic_fields(formulation::APTF,
                                   momentum,
                                   moisture_density,
                                   microphysical_fields,
                                   tracers)

    ρθ = formulation.thermodynamics.potential_temperature_density
    thermodynamic_variables = (ρθ=ρθ, ρqᵗ=moisture_density)
    return merge(momentum, thermodynamic_variables, microphysical_fields, tracers)
end

const LiquidIcePotentialTemperatureAnelasticModel = AtmosphereModel{<:APTF}
const LIPTAM = LiquidIcePotentialTemperatureAnelasticModel 

liquid_ice_potential_temperature_density(model::LIPTAM) = model.formulation.thermodynamics.potential_temperature_density
liquid_ice_potential_temperature(model::LIPTAM) = model.formulation.thermodynamics.potential_temperature
static_energy(model::LIPTAM) = Diagnostics.StaticEnergy(model, :specific)
static_energy_density(model::LIPTAM) = Diagnostics.StaticEnergy(model, :density)

function compute_thermodynamic_tendency!(model::LiquidIcePotentialTemperatureAnelasticModel, common_args)
    grid = model.grid
    arch = grid.architecture

    ρθ_args = (
        Val(1),
        model.forcing.ρθ,
        model.forcing.ρe,
        model.advection.ρθ,
        common_args...,
        model.temperature)

    Gρθ = model.timestepper.Gⁿ.ρθ
    launch!(arch, grid, :xyz, compute_potential_temperature_tendency!, Gρθ, grid, ρθ_args)
    return nothing
end

@inline function potential_temperature_tendency(i, j, k, grid,
                                                id,
                                                ρθ_forcing,
                                                ρe_forcing,
                                                advection,
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
                                                temperature)

    potential_temperature = formulation.thermodynamics.potential_temperature
    ρᵣ = formulation.reference_state.density

    𝒰 = diagnose_thermodynamic_state(i, j, k, grid,
                                     formulation,
                                     microphysics,
                                     microphysical_fields,
                                     constants,
                                     specific_moisture)

    Π = exner_function(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    closure_buoyancy = AtmosphereModelBuoyancy(formulation, constants)

    return ( - div_ρUc(i, j, k, grid, advection, ρᵣ, velocities, potential_temperature)
             - ∇_dot_Jᶜ(i, j, k, grid, ρᵣ, closure, closure_fields, id, potential_temperature, clock, model_fields, closure_buoyancy)
             + microphysical_tendency(i, j, k, grid, microphysics, Val(:ρθ), microphysical_fields, 𝒰, constants)
             + ρθ_forcing(i, j, k, grid, clock, model_fields)
             + ρe_forcing(i, j, k, grid, clock, model_fields) / (cᵖᵐ * Π))
end

#####
##### Set
#####

set_thermodynamic_variable!(model::LiquidIcePotentialTemperatureAnelasticModel, ::Union{Val{:ρθ}, Val{:ρθˡⁱ}}, value) =
    set!(model.formulation.thermodynamics.potential_temperature_density, value)

function set_thermodynamic_variable!(model::LiquidIcePotentialTemperatureAnelasticModel, ::Union{Val{:θ}, Val{:θˡⁱ}}, value)
    set!(model.formulation.thermodynamics.potential_temperature, value)
    ρᵣ = model.formulation.reference_state.density
    θˡⁱ = model.formulation.thermodynamics.potential_temperature
    set!(model.formulation.thermodynamics.potential_temperature_density, ρᵣ * θˡⁱ)
    return nothing
end

# Setting :θ (potential temperature)
function set_thermodynamic_variable!(model::LiquidIcePotentialTemperatureAnelasticModel, ::Val{:e}, value)
    thermo = model.formulation.thermodynamics
    e = model.temperature # scratch space
    set!(e, value)

    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, :xyz,
            _potential_temperature_from_energy!,
            thermo.potential_temperature_density,
            thermo.potential_temperature,
            grid,
            e,
            model.specific_moisture,
            model.formulation,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    return nothing
end

function set_thermodynamic_variable!(model::LiquidIcePotentialTemperatureAnelasticModel, ::Val{:ρe}, value)
    ρe = model.temperature # scratch space
    set!(ρe, value)
    ρᵣ = model.formulation.reference_state.density
    return set_thermodynamic_variable!(model, Val(:e), ρe / ρᵣ)
end

@kernel function _potential_temperature_from_energy!(potential_temperature_density,
                                                     potential_temperature,
                                                     grid,
                                                     specific_energy,
                                                     specific_moisture,
                                                     formulation,
                                                     microphysics,
                                                     microphysical_fields,
                                                     constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        pᵣ = formulation.reference_state.pressure[i, j, k]
        ρᵣ = formulation.reference_state.density[i, j, k]
        qᵗ = specific_moisture[i, j, k]
        e = specific_energy[i, j, k]
    end

    z = znode(i, j, k, grid, c, c, c)
    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    𝒰e₀ = StaticEnergyState(e, q, z, pᵣ)
    𝒰e₁ = maybe_adjust_thermodynamic_state(𝒰e₀, microphysics, microphysical_fields, qᵗ, constants)
    T = temperature(𝒰e₁, constants)

    p₀ = formulation.reference_state.surface_pressure
    q₁ = 𝒰e₁.moisture_mass_fractions
    𝒰θ = LiquidIcePotentialTemperatureState(zero(T), q₁, p₀, pᵣ)
    @inbounds potential_temperature[i, j, k] = with_temperature(𝒰θ, T, constants).potential_temperature
    @inbounds potential_temperature_density[i, j, k] = ρᵣ * with_temperature(𝒰θ, T, constants).potential_temperature
end
