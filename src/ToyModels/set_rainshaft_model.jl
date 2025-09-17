import Oceananigans.Fields: set!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models.NonhydrostaticModels: compute_pressure_correction!, make_pressure_correction!

function set!(model::RainshaftModel; kw...)
    for (name, value) in kw

        # Prognostic variables
        if name ∈ propertynames(model.density)
            ϕ = getproperty(model.momentum, name)
            set!(ϕ, value)
        elseif name ∈ propertynames(model.temperature)
            ϕ = getproperty(model.temperature, name)
            set!(ϕ, value)
        elseif name ∈ propertynames(model.water_vapor)
            ϕ = getproperty(model.water_vapor, name)
            set!(ϕ, value)
        elseif name ∈ propertynames(model.water_condensates)
            ϕ = getproperty(model.water_condensates, name)
            set!(ϕ, value)
        end

        # Setting diagnostic variables
        if name == :T || name == :ρ
            T = model.temperature
            rho = model.density
            ϕ = getproperty(model.pressure, name)
            value = air_pressure(model.thermodynamics, T, rho)
        end

        set!(ϕ, value)                
        fill_halo_regions!(ϕ, model.clock, fields(model))
    end

    return nothing
end
