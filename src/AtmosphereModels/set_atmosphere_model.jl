using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!, update_state!

import Oceananigans.Fields: set!

function set!(model::AtmosphereModel; enforce_mass_conservation=true, kw...)
    for (name, value) in kw

        # Prognostic variables
        if name ∈ propertynames(model.momentum)
            ϕ = getproperty(model.momentum, name)
            set!(ϕ, value)
        elseif name ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, name)
        elseif name == :ρe
            ϕ = model.energy
        elseif name == :ρq
            ϕ = model.absolute_humidity
        end

        # Setting diagnostic variables
        if name == :θ
            θ = model.temperature # use scratch
            set!(θ, value)

            ρᵣ = model.formulation.reference_density
            cᵖᵈ = model.thermodynamics.dry_air.heat_capacity
            ϕ = model.energy
            value = ρᵣ * cᵖᵈ * θ
        elseif name == :q
            q = model.specific_humidity
            set!(q, value)

            ρᵣ = model.formulation.reference_density
            ϕ = model.absolute_humidity
            value = ρᵣ * q
        elseif name ∈ (:u, :v, :w)
            u = model.velocities[name]
            set!(u, value)

            ρᵣ = model.formulation.reference_density
            ϕ = model.momentum[Symbol(:ρ, name)]
            value = ρᵣ * u
        end

        set!(ϕ, value)
        fill_halo_regions!(ϕ, model.clock, fields(model))
    end

    # Apply a mask
    # foreach(mask_immersed_field!, prognostic_fields(model))
    update_state!(model, compute_tendencies=false)

    if enforce_mass_conservation
        FT = eltype(model.grid)
        Δt = one(FT)
        compute_pressure_correction!(model, Δt)
        make_pressure_correction!(model, Δt)
        update_state!(model, compute_tendencies=false)
    end

    return nothing
end
