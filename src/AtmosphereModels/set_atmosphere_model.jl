using Oceananigans.Fields: Fields, set!
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

const settable_thermodynamic_variables = (:ρθ, :θ, :ρθˡⁱ, :θˡⁱ, :ρe, :e, :T)
function set_thermodynamic_variable! end

"""
    set!(model::AtmosphereModel; enforce_mass_conservation=true, kw...)

Set variables in an `AtmosphereModel`.

# Keyword Arguments

Variables are set via keyword arguments. Supported variables include:

**Prognostic variables** (density-weighted):
- `ρu`, `ρv`, `ρw`: momentum components
- `ρqᵗ`: total moisture density
- Prognostic microphysical variables
- Prognostic user-specified tracer fields

**Settable thermodynamic variables**:
- `T`: in-situ temperature
- `θ`: potential temperature
- `θˡⁱ`: liquid-ice potential temperature
- `e`: static energy
- `ρθ`: potential temperature density
- `ρθˡⁱ`: liquid-ice potential temperature density
- `ρe`: static energy density (for `StaticEnergyThermodynamics`)

**Diagnostic variables** (specific, i.e., per unit mass):
- `u`, `v`, `w`: velocity components (sets both velocity and momentum)
- `qᵗ`: total specific moisture (sets both specific and density-weighted moisture)

!!! note "The meaning of `θ`"
    When using `set!(model, θ=...)`, the value is interpreted as the **liquid-ice
    potential temperature** ``θˡⁱ``.

# Options

- `enforce_mass_conservation`: If `true` (default), applies a pressure correction
  to ensure the velocity field satisfies the anelastic continuity equation.
"""
function Fields.set!(model::AtmosphereModel; time=nothing, enforce_mass_conservation=true, kw...)
    if !isnothing(time)
        model.clock.time = time
    end

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

        elseif name ∈ settable_thermodynamic_variables
            set_thermodynamic_variable!(model, Val(name), value)

        else
            prognostic_names = keys(prognostic_fields(model))
            settable_diagnostic_variables = (:qᵗ, :u, :v, :w)

            msg = "Cannot set! $name in AtmosphereModel because $name is neither a
                   prognostic variable, a settable thermodynamic variable, nor a settable
                   diagnostic variable! The settable variables are
                       - prognostic variables: $prognostic_names
                       - settable thermodynamic variables: $settable_thermodynamic_variables
                       - settable diagnostic variables: $settable_diagnostic_variables"

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
