using Oceananigans.Fields: Fields, set!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: initialize_closure_fields!

using .Diagnostics: SaturationSpecificHumidity

using ..Thermodynamics:
    MoistureMassFractions,
    mixture_gas_constant

move_to_front(names, name) = tuple(name, filter(n -> n != name, names)...)

function prioritize_names(names)
    # Priority order (first items applied last, so reverse order of priority):
    # 1. ρ must be set first for compressible dynamics (density needed for momentum)
    # 2. Then velocities/momentum and moisture
    for n in (:w, :ρw, :v, :ρv, :u, :ρu, :qᵗ, :ρqᵗ, :qᵛ, :ρqᵛ, :qᵉ, :ρqᵉ, :ρ)
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

function initialize_density_weighted_aerosol_number!(model::AtmosphereModel)
    μ = model.microphysical_fields
    (:ρnᵃ ∈ keys(μ)) || return nothing

    Nᵃ₀ = initial_aerosol_number(model.microphysics)
    iszero(Nᵃ₀) && return nothing

    ρ = dynamics_density(model.dynamics)
    set!(μ.ρnᵃ, ρ * Nᵃ₀)
    return nothing
end

"""
    set!(model::AtmosphereModel; enforce_mass_conservation=true, kw...)

Set variables in an [`AtmosphereModel`](@ref).

# Keyword Arguments

Variables are set via keyword arguments. Supported variables include:

**Prognostic variables** (density-weighted):
- `ρu`, `ρv`, `ρw`: momentum components
- `ρqᵉ`/`ρqᵛ`/`ρqᵗ`: moisture density (scheme-dependent)
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
- `qᶜˡ`: specific cloud liquid, sets `ρqᶜˡ = ρᵣ * qᶜˡ`
- `qʳ`: specific rain, sets `ρqʳ = ρᵣ * qʳ`
- `nᶜˡ`: specific cloud liquid number [1/kg], sets `ρnᶜˡ = ρᵣ * nᶜˡ`
- `nʳ`: specific rain number [1/kg], sets `ρnʳ = ρᵣ * nʳ`
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
    total_moisture_was_set = false

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
            ρ = dynamics_density(model.dynamics)
            qᵛᵉ = specific_prognostic_moisture(model)
            set!(qᵛᵉ, model.moisture_density / ρ)
            total_moisture_was_set = true

        elseif name ∈ (:ρqᵛ, :ρqᵉ)
            set!(model.moisture_density, value)
            ρ = dynamics_density(model.dynamics)
            qᵛᵉ = specific_prognostic_moisture(model)
            set!(qᵛᵉ, model.moisture_density / ρ)

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
            qᵛᵉ = specific_prognostic_moisture(model)
            set!(qᵛᵉ, value)
            ρ = dynamics_density(model.dynamics)
            set!(model.moisture_density, ρ * qᵛᵉ)
            total_moisture_was_set = true

        elseif name ∈ (:qᵛ, :qᵉ)
            qᵛᵉ = specific_prognostic_moisture(model)
            set!(qᵛᵉ, value)
            ρ = dynamics_density(model.dynamics)
            set!(model.moisture_density, ρ * qᵛᵉ)

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
            initialize_density_weighted_aerosol_number!(model)

        elseif name == :ℋ
            # Call update_state! to ensure temperature is computed from thermodynamic variables
            update_state!(model, compute_tendencies=false)

            # Compute saturation specific humidity into a concrete field.
            # This must be materialized before overwriting qᵗ, because
            # SaturationSpecificHumidity reads qᵗ by reference.
            qᵛ⁺ = Field(SaturationSpecificHumidity(model, :equilibrium))

            # Set specific prognostic moisture = ℋ * qᵛ⁺
            qᵛᵉ = specific_prognostic_moisture(model)

            # Set qᵛᵉ = ℋ * qᵛ⁺
            # First set ℋ onto qᵛᵉ (evaluates functions on CPU for GPU compatibility),
            # then multiply by the materialized saturation specific humidity.
            set!(qᵛᵉ, value)
            set!(qᵛᵉ, qᵛᵉ * qᵛ⁺)

            ρ = dynamics_density(model.dynamics)
            set!(model.moisture_density, ρ * qᵛᵉ)

        else
            prognostic_names = keys(prognostic_fields(model))
            settable_diagnostic_variables = (:qᵗ, :qᵛ, :qᵉ, :ℋ, :u, :v, :w)
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

    if total_moisture_was_set
        ρ = dynamics_density(model.dynamics)
        qᵗ = model.moisture_density / ρ

        if !isnothing(model.microphysics) &&
           hasmethod(specific_prognostic_moisture_from_total,
                     Tuple{typeof(model.microphysics), typeof(qᵗ), typeof(model.microphysical_fields), typeof(ρ)})
            qᵛᵉ = specific_prognostic_moisture(model)
            set!(qᵛᵉ, specific_prognostic_moisture_from_total(model.microphysics, qᵗ, model.microphysical_fields, ρ))
            set!(model.moisture_density, ρ * qᵛᵉ)
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

    initialize_closure_fields!(model.closure_fields, model.closure, model)

    return nothing
end
