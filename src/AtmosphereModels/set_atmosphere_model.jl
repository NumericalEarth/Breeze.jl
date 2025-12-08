using Oceananigans.Grids: znode, Center
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!, update_state!

using ..Thermodynamics:
    LiquidIcePotentialTemperatureState,
    MoistureMassFractions,
    mixture_heat_capacity,
    mixture_gas_constant,
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

function set_thermodynamic_variable! end

"""
    set!(model::AtmosphereModel; enforce_mass_conservation=true, kw...)

Set variables in an `AtmosphereModel`.

# Keyword Arguments

Variables are set via keyword arguments. Supported variables include:

**Prognostic variables** (density-weighted):
- `ρu`, `ρv`, `ρw`: momentum components
- `ρqᵗ`: total moisture density
- `ρθ`: potential temperature density (for `LiquidIcePotentialTemperatureThermodynamics`)
- `ρe`: static energy density (for `StaticEnergyThermodynamics`)

**Diagnostic variables** (specific, ie per unit mass):
- `u`, `v`, `w`: velocity components (sets both velocity and momentum)
- `qᵗ`: total specific moisture (sets both specific and density-weighted moisture)
- `θ`: potential temperature (interpreted as liquid-ice potential temperature ``θˡⁱ``)
- `e`: static energy

!!! note "The meaning of `θ`"
    When using `set!(model, θ=...)`, the value is interpreted as the **liquid-ice
    potential temperature** ``θˡⁱ``, not the dry potential temperature ``θᵈ``.
    This is consistent with the prognostic variable in `LiquidIcePotentialTemperatureThermodynamics`.

# Options

- `enforce_mass_conservation`: If `true` (default), applies a pressure correction
  to ensure the velocity field satisfies the anelastic continuity equation.
"""
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
            set_thermodynamic_variable!(model, Val(:ρe), value)

        elseif name == :ρθ
            set_thermodynamic_variable!(model, Val(:ρθ), value)

        elseif name == :ρqᵗ
            set!(model.moisture_density, value)
            ρqᵗ = model.moisture_density
            ρᵣ = model.formulation.reference_state.density
            set!(model.specific_moisture, ρqᵗ / ρᵣ)

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

        elseif name == :e
            set_thermodynamic_variable!(model, Val(:e), value)

        elseif name == :θ
            set_thermodynamic_variable!(model, Val(:θ), value)

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

    return nothing
end
