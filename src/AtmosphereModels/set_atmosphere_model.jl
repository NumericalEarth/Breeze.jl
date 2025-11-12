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
        end

        # Setting diagnostic variables
        if name == :qᵗ
            qᵗ = model.moisture_mass_fraction
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
                    grid,
                    θ,
                    model.moisture_density,
                    model.formulation,
                    model.microphysics,
                    model.thermodynamics)
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

@kernel function _energy_density_from_potential_temperature!(energy_density, grid,
                                                             potential_temperature,
                                                             moisture_density,
                                                             formulation::AnelasticFormulation,
                                                             microphysics,
                                                             thermo)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        pᵣ = formulation.reference_state.pressure[i, j, k]
        ρᵣ = formulation.reference_state.density[i, j, k]
        θ = potential_temperature[i, j, k]
        qᵗ = moisture_density[i, j, k] / ρᵣ
    end

    g = thermo.gravitational_acceleration
    z = znode(i, j, k, grid, c, c, c)
    p₀ = formulation.reference_state.base_pressure

    # Assuming a state with no condensate?
    # TODO use microphysics model in the course of determining q
    q = MoistureMassFractions(qᵗ)
    𝒰₀ = PotentialTemperatureState(θ, q, z, p₀, pᵣ, ρᵣ)
    𝒰 = compute_thermodynamic_state(𝒰₀, microphysics, thermo)

    T = temperature(𝒰, thermo)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, thermo)

    ℒˡᵣ = thermo.liquid.reference_latent_heat
    ℒⁱᵣ = thermo.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    @inbounds energy_density[i, j, k] = ρᵣ * (cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ)
end
