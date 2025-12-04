struct LiquidIcePotentialTemperatureThermodynamics{F, T}
    potential_temperature_density :: F  # ρθ (prognostic)
    potential_temperature :: T          # θ = ρθ / ρᵣ (diagnostic)
end

Adapt.adapt_structure(to, thermo::LiquidIcePotentialTemperatureThermodynamics) =
    LiquidIcePotentialTemperatureThermodynamics(adapt(to, thermo.potential_temperature_density),
                                       adapt(to, thermo.potential_temperature))

function fill_halo_regions!(thermo::LiquidIcePotentialTemperatureThermodynamics)
    fill_halo_regions!(thermo.potential_temperature_density)
    fill_halo_regions!(thermo.potential_temperature)
    return nothing
end

const APTF = AnelasticFormulation{<:LiquidIcePotentialTemperatureThermodynamics}
prognostic_field_names(formulation::APTF) = tuple(:ρθ)
additional_field_names(formulation::APTF) = tuple(:θ)
thermodynamic_density_name(::APTF) = :ρθ
fields(formulation::APTF) = (; θ=formulation.thermodynamics.potential_temperature)
prognostic_fields(formulation::APTF) = (; ρθ=formulation.thermodynamics.potential_temperature_density)

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
    p₀ = formulation.reference_state.base_pressure
    qᵗ = @inbounds specific_moisture[i, j, k]

    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)

    return PotentialTemperatureState(θ, q, p₀, pᵣ)
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

potential_temperature_density(thermo::LiquidIcePotentialTemperatureThermodynamics) = thermo.potential_temperature_density
potential_temperature(thermo::LiquidIcePotentialTemperatureThermodynamics) = thermo.potential_temperature
energy_density(::LiquidIcePotentialTemperatureThermodynamics) = nothing
specific_energy(::LiquidIcePotentialTemperatureThermodynamics) = nothing

const PotentialTemperatureAnelasticModel = AtmosphereModel{<:APTF}

function compute_thermodynamic_tendency!(model::LiquidIcePotentialTemperatureAnelasticModel, common_args)
    grid = model.grid
    arch = grid.architecture

    ρθ_args = (
        Val(1),
        model.forcing.ρθ,
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
    ρ = formulation.reference_state.density

    # Note: Unlike static energy, potential temperature does not have a buoyancy flux term
    # since potential temperature is conserved under adiabatic processes.

    𝒰 = diagnose_thermodynamic_state(i, j, k, grid,
                                     formulation,
                                     microphysics,
                                     microphysical_fields,
                                     constants,
                                     specific_moisture)

    closure_buoyancy = AtmosphereModelBuoyancy(formulation, constants)

    return ( - div_ρUc(i, j, k, grid, advection, ρ, velocities, potential_temperature)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ, closure, closure_fields, id, potential_temperature, clock, model_fields, closure_buoyancy)
             + microphysical_tendency(i, j, k, grid, microphysics, Val(:ρθ), microphysical_fields, 𝒰, constants)
             + ρθ_forcing(i, j, k, grid, clock, model_fields))
end
