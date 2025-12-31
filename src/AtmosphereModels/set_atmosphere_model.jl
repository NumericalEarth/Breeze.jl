using Oceananigans.Fields: Fields, set!, interior
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!, update_state!

using ..Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity,
    mixture_gas_constant,
    saturation_vapor_pressure,
    dry_air_gas_constant,
    vapor_gas_constant,
    PlanarLiquidSurface

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

"""
    specific_to_density_weighted(name::Symbol)

Convert a specific microphysical variable name to its density-weighted counterpart.
For example, `:qᶜˡ` → `:ρqᶜˡ`, `:qʳ` → `:ρqʳ`.

Returns `nothing` if the name doesn't start with 'q'.
"""
function specific_to_density_weighted(name::Symbol)
    str = string(name)
    if startswith(str, "q")
        return Symbol("ρ" * str)
    else
        return nothing
    end
end

"""
    settable_specific_microphysical_names(microphysics)

Return a tuple of specific (non-density-weighted) names that can be set
for the given microphysics scheme. These are derived from the prognostic
field names by removing the 'ρ' prefix.
"""
function settable_specific_microphysical_names(microphysics)
    prog_names = prognostic_field_names(microphysics)
    specific_names = Symbol[]
    for name in prog_names
        str = string(name)
        if startswith(str, "ρq")
            push!(specific_names, Symbol(str[nextind(str, 1):end]))  # Remove 'ρ' prefix
        end
    end
    return Tuple(specific_names)
end

settable_specific_microphysical_names(::Nothing) = ()

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
- `ℋ`: relative humidity (sets total moisture via `qᵗ = ℋ * qᵛ⁺`, where `qᵛ⁺` is the
  saturation specific humidity at the current temperature). Relative humidity is in
  the range [0, 1]. For models with saturation adjustment microphysics, `ℋ > 1` throws
  an error since the saturation adjustment would immediately reduce it to 1.

**Specific microphysical variables** (automatically converted to density-weighted):
- `qᶜˡ`: specific cloud liquid (sets `ρqᶜˡ = ρᵣ * qᶜˡ`)
- `qʳ`: specific rain (sets `ρqʳ = ρᵣ * qʳ`)
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
            ρu = getproperty(model.momentum, name)
            set!(ρu, value)

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

        elseif name == :ℋ
            # Set total moisture from relative humidity
            # Compute qᵛ⁺ (saturation specific humidity at the current temperature)
            # then set qᵗ = ℋ * qᵛ⁺
            set_total_moisture_from_relative_humidity!(model, value)

        elseif name ∈ (:u, :v, :w)
            u = model.velocities[name]
            set!(u, value)

            ρ = dynamics_density(model.dynamics)
            ϕ = model.momentum[Symbol(:ρ, name)]
            value = ρ * u
            set!(ϕ, value)    

        elseif name ∈ settable_thermodynamic_variables
            set_thermodynamic_variable!(model, Val(name), value)

        elseif name == :ρ
            # Set density for compressible dynamics
            ρ = dynamics_density(model.dynamics)
            set!(ρ, value)
            # Fill halos immediately - needed for velocity→momentum conversion
            fill_halo_regions!(ρ)

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

"""
    set_total_moisture_from_relative_humidity!(model, ℋ)

Set the total moisture content from relative humidity `ℋ`.

Computes `qᵗ = ℋ * qᵛ⁺` where `qᵛ⁺` is the saturation specific humidity
at the current temperature. Relative humidity `ℋ` should be in the range [0, 1].

For models with saturation adjustment microphysics (as determined by
`is_saturation_adjustment`), throws an error if any value of `ℋ` exceeds 1,
since the saturation adjustment would immediately reduce the relative humidity to 1.
"""
function set_total_moisture_from_relative_humidity!(model, ℋ_value)
    # First, check if we have saturation adjustment microphysics
    # If so, we need to validate that ℋ ≤ 1 everywhere
    if is_saturation_adjustment(model.microphysics)
        validate_relative_humidity_for_saturation_adjustment(ℋ_value)
    end

    # Call update_state! to ensure temperature is computed from thermodynamic variables
    update_state!(model, compute_tendencies=false)

    # Compute saturation specific humidity and then qᵗ = ℋ * qᵛ⁺
    compute_moisture_from_relative_humidity!(model, ℋ_value)

    return nothing
end

# Handle scalar/number relative humidity
function compute_moisture_from_relative_humidity!(model, ℋ::Number)
    T = model.temperature
    ρ = dynamics_density(model.dynamics)
    constants = model.thermodynamic_constants

    qᵛ⁺ = saturation_specific_humidity_field(T, ρ, constants)

    qᵗ = model.specific_moisture
    set!(qᵗ, ℋ * qᵛ⁺)

    ρqᵗ = model.moisture_density
    set!(ρqᵗ, ρ * qᵗ)

    return nothing
end

# Handle function relative humidity: need to evaluate the function first
function compute_moisture_from_relative_humidity!(model, ℋ_func::Function)
    T = model.temperature
    ρ = dynamics_density(model.dynamics)
    constants = model.thermodynamic_constants
    grid = model.grid

    qᵛ⁺ = saturation_specific_humidity_field(T, ρ, constants)

    # Create a temporary field to hold ℋ values
    ℋ_field = CenterField(grid)
    set!(ℋ_field, ℋ_func)

    # Now we can check for supersaturation with saturation adjustment
    if is_saturation_adjustment(model.microphysics)
        validate_relative_humidity_for_saturation_adjustment(interior(ℋ_field))
    end

    qᵗ = model.specific_moisture
    set!(qᵗ, ℋ_field * qᵛ⁺)

    ρqᵗ = model.moisture_density
    set!(ρqᵗ, ρ * qᵗ)

    return nothing
end

# Handle array/field relative humidity
function compute_moisture_from_relative_humidity!(model, ℋ_array::AbstractArray)
    T = model.temperature
    ρ = dynamics_density(model.dynamics)
    constants = model.thermodynamic_constants

    qᵛ⁺ = saturation_specific_humidity_field(T, ρ, constants)

    qᵗ = model.specific_moisture
    set!(qᵗ, ℋ_array .* qᵛ⁺)

    ρqᵗ = model.moisture_density
    set!(ρqᵗ, ρ * qᵗ)

    return nothing
end

"""
    saturation_specific_humidity_field(T, ρ, constants)

Compute an array containing the saturation specific humidity using the formula:
```math
qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T)
```
where `pᵛ⁺` is the saturation vapor pressure.

This is consistent with the relative humidity diagnostic which computes
`ℋ = pᵛ / pᵛ⁺ = (ρ qᵛ Rᵛ T) / pᵛ⁺`.
"""
function saturation_specific_humidity_field(T, ρ, constants)
    Rᵛ = vapor_gas_constant(constants)
    surface = PlanarLiquidSurface()

    # Compute saturation vapor pressure
    pᵛ⁺ = saturation_vapor_pressure.(T, Ref(constants), Ref(surface))
    
    # Saturation specific humidity: qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T)
    qᵛ⁺ = pᵛ⁺ ./ (ρ .* Rᵛ .* T)

    return qᵛ⁺
end

# Validation for constant/scalar relative humidity
function validate_relative_humidity_for_saturation_adjustment(ℋ::Number)
    if ℋ > 1
        throw(ArgumentError("Cannot set relative humidity ℋ = $ℋ > 1 with " *
                            "SaturationAdjustment microphysics. The saturation adjustment " *
                            "would immediately reduce the relative humidity to 1. " *
                            "Use qᵗ instead if you want to set supersaturated conditions."))
    end
    return nothing
end

# Validation for functions: we can't validate until it's evaluated, so skip
validate_relative_humidity_for_saturation_adjustment(ℋ::Function) = nothing

# Validation for arrays/fields: check max value
function validate_relative_humidity_for_saturation_adjustment(ℋ::AbstractArray)
    ℋ_max = maximum(ℋ)
    if ℋ_max > 1
        throw(ArgumentError("Cannot set relative humidity with maximum value ℋ = $ℋ_max > 1 with " *
                            "SaturationAdjustment microphysics. The saturation adjustment " *
                            "would immediately reduce the relative humidity to 1. " *
                            "Use qᵗ instead if you want to set supersaturated conditions."))
    end
    return nothing
end
