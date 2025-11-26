using Oceananigans.Grids: znode, Center
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!, update_state!

using ..Thermodynamics:
    PotentialTemperatureState,
    MoistureMassFractions,
    mixture_heat_capacity,
    temperature

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
            set!(model.energy_density, value)
        elseif name == :ρqᵗ
            set!(model.moisture_density, value)
            set!(model.specific_moisture, model.moisture_density / model.formulation.reference_state.density)
        elseif name ∈ prognostic_field_names(model.microphysics)
            μ = getproperty(model.microphysical_fields, name)
            set!(μ, value)

        elseif name == :qᵗ
            qᵗ = model.specific_moisture
            set!(qᵗ, value)
            ρᵣ = model.formulation.reference_state.density
            ρqᵗ = model.moisture_density
            set!(ρqᵗ, ρᵣ * qᵗ)                

        elseif name ∈ (:u, :v, :w)
            u = model.velocities[name]
            set!(u, value)

            ρᵣ = model.formulation.reference_state.density
            ϕ = model.momentum[Symbol(:ρ, name)]
            value = ρᵣ * u
            set!(ϕ, value)    

        elseif name == :θ
            θ = model.temperature # use scratch
            set!(θ, value)

            grid = model.grid
            arch = grid.architecture

            launch!(arch, grid, :xyz,
                    _energy_density_from_potential_temperature!,
                    model.energy_density,
                    model.specific_energy,
                    grid,
                    θ,
                    model.specific_moisture,
                    model.formulation,
                    model.microphysics,
                    model.microphysical_fields,
                    model.thermodynamics)

        elseif name == :e
            # Set specific energy directly
            set!(model.specific_energy, value)
            ρᵣ = model.formulation.reference_state.density
            set!(model.energy_density, ρᵣ * model.specific_energy)

        else
            prognostic_names = keys(prognostic_fields(model))
            supported_diagnostic_variables = (:qᵗ, :u, :v, :w, :θ, :e)

            msg = "Cannot set! $name in AtmosphereModel because $name is neither a
                   prognostic variable nor a supported diagnostic variable!
                   The prognostic variables are: $prognostic_names
                   The supported diagnostic variables are: $supported_diagnostic_variables"

            throw(ArgumentError(msg))
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

    fill_halo_regions!(model.energy_density)

    return nothing
end

@kernel function _energy_density_from_potential_temperature!(energy_density,
                                                             specific_energy,
                                                             grid,
                                                             potential_temperature,
                                                             specific_moisture,
                                                             formulation::AnelasticFormulation,
                                                             microphysics,
                                                             microphysical_fields,
                                                             thermo)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        pᵣ = formulation.reference_state.pressure[i, j, k]
        ρᵣ = formulation.reference_state.density[i, j, k]
        qᵗ = specific_moisture[i, j, k]
        θ = potential_temperature[i, j, k]
    end

    g = thermo.gravitational_acceleration
    z = znode(i, j, k, grid, c, c, c)
    p₀ = formulation.reference_state.base_pressure

    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    𝒰₀ = PotentialTemperatureState(θ, q, p₀, pᵣ)
    𝒰 = maybe_adjust_thermodynamic_state(𝒰₀, microphysics, microphysical_fields, qᵗ, thermo)

    T = temperature(𝒰, thermo)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, thermo)

    ℒˡᵣ = thermo.liquid.reference_latent_heat
    ℒⁱᵣ = thermo.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ
    @inbounds specific_energy[i, j, k] = e
    @inbounds energy_density[i, j, k] = ρᵣ * e
end
