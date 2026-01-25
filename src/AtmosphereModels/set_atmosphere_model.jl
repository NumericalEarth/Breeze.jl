using Oceananigans.Fields: Fields, set!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!, compute_pressure_correction!, make_pressure_correction!

using .Diagnostics: SaturationSpecificHumidity

using ..Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity,
    mixture_gas_constant

move_to_front(names, name) = tuple(name, filter(n -> n != name, names)...)

function prioritize_names(names)
    # Priority order (first items applied last, so reverse order of priority):
    # 1. ρ must be set first for compressible dynamics (density needed for momentum)
    # 2. Then velocities/momentum and moisture
    for n in (:w, :ρw, :v, :ρv, :u, :ρu, :qᵗ, :ρqᵗ, :ρ)
        if n ∈ names
            names = move_to_front(names, n)
        end
    end

    return names
end

const settable_thermodynamic_variables = (:ρθ, :θ, :ρθˡⁱ, :θˡⁱ, :ρe, :e, :T)
function set_thermodynamic_variable! end

#####
##### Velocity and momentum setting (extensible for kinematic models)
#####

"""
    set_velocity!(model, name, value)

Set the velocity component `name` (`:u`, `:v`, or `:w`) to `value`.
Also updates the corresponding momentum field.
"""
function set_velocity!(model::AtmosphereModel, name::Symbol, value)
    u = model.velocities[name]
    set!(u, value)
    ρ = dynamics_density(model.dynamics)
    ϕ = model.momentum[Symbol(:ρ, name)]
    set!(ϕ, ρ * u)
    return nothing
end

"""
    set_momentum!(model, name, value)

Set the momentum component `name` (`:ρu`, `:ρv`, or `:ρw`) to `value`.
"""
function set_momentum!(model::AtmosphereModel, name::Symbol, value)
    ρu = getproperty(model.momentum, name)
    set!(ρu, value)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Convert a specific microphysical variable name to its density-weighted counterpart.
For example, `:qᶜˡ` → `:ρqᶜˡ`, `:qʳ` → `:ρqʳ`, `:nᶜˡ` → `:ρnᶜˡ`.

Returns `nothing` if the name doesn't start with 'q' or 'n'.
"""
function specific_to_density_weighted(name::Symbol)
    str = string(name)
    if startswith(str, "q") || startswith(str, "n")
        return Symbol("ρ" * str)
    else
        return nothing
    end
end

"""
$(TYPEDSIGNATURES)

Return a tuple of specific (non-density-weighted) names that can be set
for the given microphysics scheme. These are derived from the prognostic
field names by removing the 'ρ' prefix.

For mass fields (e.g., `ρqᶜˡ` → `qᶜˡ`) and number fields (e.g., `ρnᶜˡ` → `nᶜˡ`).
"""
function settable_specific_microphysical_names(microphysics)
    prog_names = prognostic_field_names(microphysics)
    specific_names = Symbol[]
    for name in prog_names
        str = string(name)
        # Handle both mass fields (ρq*) and number fields (ρn*)
        if startswith(str, "ρq") || startswith(str, "ρn")
            push!(specific_names, Symbol(str[nextind(str, 1):end]))  # Remove 'ρ' prefix
        end
    end
    return Tuple(specific_names)
end

settable_specific_microphysical_names(::Nothing) = ()

"""
    set!(model::AtmosphereModel; enforce_mass_conservation=true, kw...)

Set variables in an [`AtmosphereModel`](@ref).

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
- `ℋ`: relative humidity (sets total moisture via `qᵗ = ℋ * qᵛ⁺`, where `qᵛ⁺` is the
  saturation specific humidity at the current temperature). Relative humidity is in
  the range [0, 1]. For models with saturation adjustment microphysics, `ℋ > 1` throws
  an error since the saturation adjustment would immediately reduce it to 1.

**Specific microphysical variables** (automatically converted to density-weighted):
- `qᶜˡ`: specific cloud liquid (sets `ρqᶜˡ = ρᵣ * qᶜˡ`)
- `qʳ`: specific rain (sets `ρqʳ = ρᵣ * qʳ`)
- `nᶜˡ`: specific cloud liquid number [1/kg] (sets `ρnᶜˡ = ρᵣ * nᶜˡ`)
- `nʳ`: specific rain number [1/kg] (sets `ρnʳ = ρᵣ * nʳ`)
- Other prognostic microphysical variables with the `ρ` prefix removed

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
            set_momentum!(model, name, value)

        elseif name ∈ propertynames(model.tracers)
            c = getproperty(model.tracers, name)
            set!(c, value)

        elseif name == :ρqᵗ
            set!(model.moisture_density, value)
            ρqᵗ = model.moisture_density
            ρ = dynamics_density(model.dynamics)
            set!(model.specific_moisture, ρqᵗ / ρ)

        elseif name ∈ prognostic_field_names(model.microphysics)
            μ = getproperty(model.microphysical_fields, name)
            set!(μ, value)

        elseif name ∈ settable_specific_microphysical_names(model.microphysics)
            # Convert specific value to density-weighted: ρq = ρ * q
            density_name = specific_to_density_weighted(name)
            ρμ = model.microphysical_fields[density_name]
            set!(ρμ, value)
            ρ = dynamics_density(model.dynamics)
            set!(ρμ, ρ * ρμ)

        elseif name == :qᵗ
            qᵗ = model.specific_moisture
            set!(qᵗ, value)
            ρ = dynamics_density(model.dynamics)
            ρqᵗ = model.moisture_density
            set!(ρqᵗ, ρ * qᵗ)

        elseif name ∈ (:u, :v, :w)
            set_velocity!(model, name, value)

        elseif name ∈ settable_thermodynamic_variables
            set_thermodynamic_variable!(model, Val(name), value)

        elseif name == :ρ
            # Set density for compressible dynamics
            ρ = dynamics_density(model.dynamics)
            set!(ρ, value)
            # Fill halos immediately - needed for velocity→momentum conversion
            fill_halo_regions!(ρ)

        elseif name == :ℋ
            # Call update_state! to ensure temperature is computed from thermodynamic variables
            update_state!(model, compute_tendencies=false)

            # Compute saturation specific humidity using GPU-compatible kernel
            # Use :equilibrium flavor which handles both saturated and unsaturated conditions
            qᵛ⁺ = SaturationSpecificHumidity(model, :equilibrium)

            # Set qᵗ = ℋ * qᵛ⁺
            qᵗ = model.specific_moisture
            set!(qᵗ, value * qᵛ⁺)

            ρ = dynamics_density(model.dynamics)
            ρqᵗ = model.moisture_density
            set!(ρqᵗ, ρ * qᵗ)

        else
            prognostic_names = keys(prognostic_fields(model))
            settable_diagnostic_variables = (:qᵗ, :ℋ, :u, :v, :w)
            specific_microphysical = settable_specific_microphysical_names(model.microphysics)

            msg = "Cannot set! $name in AtmosphereModel because $name is neither a
                   prognostic variable, a settable thermodynamic variable, nor a settable
                   diagnostic variable! The settable variables are
                       - prognostic variables: $prognostic_names
                       - settable thermodynamic variables: $settable_thermodynamic_variables
                       - settable diagnostic variables: $settable_diagnostic_variables
                       - specific microphysical variables: $specific_microphysical"

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
