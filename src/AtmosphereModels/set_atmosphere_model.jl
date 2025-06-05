import Oceananigans.Fields: set!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models.NonhydrostaticModels: calculate_pressure_correction!, make_pressure_correction!
using Oceananigans.Models.NonhydrostaticModels: compute_pressure_correction!, pressure_correct_velocities!

function set!(model::AtmosphereModel; enforce_mass_conservation=true, kw...)
    for (name, value) in kw

        # Prognostic variables
        if name ∈ propertynames(model.momentum)
            ϕ = getproperty(model.momentum, name)
            set!(ϕ, value)
        elseif name ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, name)
        elseif name == :e
            ϕ = model.energy
        elseif name == :ρq
            ϕ = model.absolute_humidity
        end

        # Setting diagnostic variables
        if name == :θ
            θ = model.temperature # use scratch
            set!(θ, value)

            ρʳ = model.formulation.reference_density
            cᵖᵈ = model.thermodynamics.dry_air.heat_capacity
            ϕ = model.energy # use scratch
            value = ρʳ * cᵖᵈ * θ
        elseif name == :q
            q = model.specific_humidity
            set!(q, value)

            ρʳ = model.formulation.reference_density
            ϕ = model.absolute_humidity
            value = ρʳ * q
        elseif name ∈ (:u, :v, :w)
            u = model.velocities[name]
            set!(u, value)

            ρʳ = model.formulation.reference_density
            ϕ = model.momentum[Symbol(:ρ, name)]
            value = ρʳ * u
        end

        set!(ϕ, value)                
        fill_halo_regions!(ϕ, model.clock, fields(model))
    end

    # Apply a mask
    # foreach(mask_immersed_field!, prognostic_fields(model))
    update_state!(model, compute_tendencies=false)
    
    if enforce_mass_conservation
        FT = eltype(model.grid)
        calculate_pressure_correction!(model, one(FT))
        make_pressure_correction!(model, one(FT))
        update_state!(model, compute_tendencies=false)
        compute_pressure_correction!(model, one(FT))
        make_pressure_correction!(model, one(FT))
        update_state!(model)
    end

    return nothing
end
