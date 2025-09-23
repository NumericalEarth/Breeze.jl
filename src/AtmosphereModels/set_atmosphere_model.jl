using Oceananigans.Grids: znode, Center
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models.NonhydrostaticModels: compute_pressure_correction!, make_pressure_correction!

using ..Thermodynamics: exner_function, SpecificHumidities, mixture_heat_capacity, PotentialTemperatureState, temperature

import Oceananigans.Fields: set!

const c = Center()

move_to_front(names, name) = tuple(name, filter(n -> n != name, names)...)

function prioritize_names(names)
    for n in (:w, :ρw, :v, :ρv, :u, :ρu, :qᵗ, :ρqᵗ)
        if n ∈ names
            names = move_to_front(names, n)
        end
    end

    return names
end

function set!(model::AtmosphereModel; enforce_mass_conservation=true, kw...)
    names = collect(keys(kw))
    prioritized = prioritize_names(names)

    for name in prioritized
        value = kw[name]

        # Prognostic variables
        if name ∈ propertynames(model.momentum)
            ρu = getproperty(model.momentum, name)
            set!(ρu, value)
        elseif name ∈ propertynames(model.tracers)
            c = getproperty(model.tracers, name)
            set!(c, value)
        elseif name == :ρe
            set!(model.energy, value)
        elseif name == :ρqᵗ
            set!(model.absolute_humidity, value)
        end

        # Setting diagnostic variables
        if name == :θ
            θ = model.temperature # use scratch
            set!(θ, value)

            grid = model.grid
            arch = grid.architecture
            thermo = model.thermodynamics
            formulation = model.formulation
            energy = model.energy
            specific_humidity = model.specific_humidity
            launch!(arch, grid, :xyz, _energy_from_potential_temperature!, energy, grid,
                    θ, specific_humidity, formulation, thermo)

        elseif name == :qᵗ
            qᵗ = model.specific_humidity
            set!(qᵗ, value)
            ρʳ = model.formulation.reference_density
            ρqᵗ = model.absolute_humidity
            set!(ρqᵗ, ρʳ * qᵗ)                

        elseif name ∈ (:u, :v, :w)
            u = model.velocities[name]
            set!(u, value)

            ρʳ = model.formulation.reference_density
            ϕ = model.momentum[Symbol(:ρ, name)]
            value = ρʳ * u
            set!(ϕ, value)                
        end
    end

    # Apply a mask
    foreach(mask_immersed_field!, prognostic_fields(model))
    update_state!(model, compute_tendencies=false)
    
    if enforce_mass_conservation
        FT = eltype(model.grid)
        Δt = one(FT)
        compute_pressure_correction!(model, Δt)
        make_pressure_correction!(model, Δt)
        update_state!(model, compute_tendencies=false)
    end

    fill_halo_regions!(model.energy)

    return nothing
end

@kernel function _energy_from_potential_temperature!(moist_static_energy, grid,
                                                     potential_temperature,
                                                     specific_humidity,
                                                     formulation,
                                                     thermo)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρʳ = formulation.reference_density[i, j, k]
        qᵗ = specific_humidity[i, j, k]
        θ = potential_temperature[i, j, k]
        z = znode(i, j, k, grid, c, c, c)
    end

    # Assume non-condensed state
    # TODO: relax this assumption
    q = SpecificHumidities(qᵗ, zero(qᵗ), zero(qᵗ))
    ref = formulation.reference_state_constants
    𝒰 = PotentialTemperatureState(θ, q, z, ref)
    T = temperature(𝒰, thermo)
    ℒ₀ = thermo.liquid.latent_heat
    g = thermo.gravitational_acceleration
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    @inbounds moist_static_energy[i, j, k] = ρʳ * (cᵖᵐ * T + g * z + qᵗ * ℒ₀)
end
