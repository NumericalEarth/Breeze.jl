# Microphysics Interface

This document describes the microphysics interface for implementing custom microphysics schemes in Breeze. The interface consists of eight functions that must be implemented for any microphysics scheme to work with `AtmosphereModel`.

## Overview

The microphysics interface allows developers to implement custom schemes for modeling cloud microphysical processes, including phase changes (vapor ↔ liquid ↔ ice), precipitation formation, and particle sedimentation. The interface is designed to be GPU-compatible and type-stable for optimal performance.

## Getting Started: Building a OneMomentCloudMicrophysics Scheme

To illustrate the interface, we'll walk through building a `OneMomentCloudMicrophysics` scheme using CloudMicrophysics.jl. This scheme includes prognostic precipitation fields (rain and snow) and demonstrates all aspects of the interface.

```julia
using Breeze
using CloudMicrophysics
using Oceananigans

# Load the CloudMicrophysics extension
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

# Create a one-moment microphysics scheme
microphysics = OneMomentCloudMicrophysics()
```

## Interface Functions

### `prognostic_field_names(microphysics)`

**Signature:** `prognostic_field_names(microphysics) -> Tuple{Symbol, ...}`

**Purpose:** Returns a tuple of symbols naming the prognostic microphysical fields that must be advected by the model.

**Returns:** A tuple of symbols, e.g., `(:ρqʳ,)` for warm-phase one-moment scheme or `(:ρqʳ, :ρqˢ)` for mixed-phase.

**Example:**

```julia
# For OneMomentCloudMicrophysics with mixed-phase saturation adjustment
prognostic_field_names(microphysics)
# Returns: (:ρqʳ, :ρqˢ)

# For SaturationAdjustment (no prognostic fields)
prognostic_field_names(SaturationAdjustment())
# Returns: ()
```

**Implementation Notes:**
- Return `tuple()` for schemes with no prognostic microphysical fields (e.g., saturation adjustment schemes)
- Field names should use the notation from `docs/src/appendix/notation.md`
- Prognostic fields are typically density-weighted (e.g., `ρqʳ` rather than `qʳ`)

### `materialize_microphysical_fields(microphysics, grid, boundary_conditions)`

**Signature:** `materialize_microphysical_fields(microphysics, grid, boundary_conditions) -> NamedTuple`

**Purpose:** Creates the field containers for all microphysical fields (both prognostic and diagnostic) on the given grid.

**Parameters:**
- `microphysics`: The microphysics scheme
- `grid`: An Oceananigans grid
- `boundary_conditions`: A NamedTuple of boundary conditions (may be empty)

**Returns:** A NamedTuple with field names as keys and `CenterField` objects as values.

**Example:**

```julia
using Breeze
using Oceananigans

# Create a column grid (1D vertical)
grid = RectilinearGrid(size=(1, 1, 64), extent=(1000, 1000, 1000), 
                       topology=(Flat, Flat, Bounded))
boundary_conditions = NamedTuple()

# For OneMomentCloudMicrophysics (mixed-phase)
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

microphysics = OneMomentCloudMicrophysics()
fields = materialize_microphysical_fields(microphysics, grid, boundary_conditions)

# Returns a NamedTuple with:
# - qᵛ: vapor mass fraction (diagnostic)
# - qᶜˡ: cloud liquid mass fraction (diagnostic)  
# - qᶜⁱ: cloud ice mass fraction (diagnostic)
# - ρqʳ: rain density (prognostic)
# - ρqˢ: snow density (prognostic)

# Inspect the fields
println("Field names: ", keys(fields))
println("qᵛ field type: ", typeof(fields.qᵛ))
println("ρqʳ field type: ", typeof(fields.ρqʳ))
```

**Implementation Notes:**
- Use `center_field_tuple(grid, names...)` helper function from `Breeze.Microphysics` to create fields
- All microphysical fields are located at cell centers (`CenterField`)
- Diagnostic fields store quantities computed from the thermodynamic state
- Prognostic fields are advected by the model and updated by tendencies

### `update_microphysical_fields!(microphysical_fields, microphysics, i, j, k, grid, density, state, thermo)`

**Signature:** `update_microphysical_fields!(microphysical_fields, microphysics, i, j, k, grid, density, state, thermo) -> Nothing`

**Purpose:** Updates diagnostic microphysical fields from the thermodynamic state after saturation adjustment or other microphysical processes.

**Parameters:**
- `microphysical_fields`: NamedTuple of microphysical fields (mutated)
- `microphysics`: The microphysics scheme
- `i, j, k`: Grid indices
- `grid`: The model grid
- `density`: Reference density field (for computing mass fractions from densities)
- `state`: The adjusted thermodynamic state (`AbstractThermodynamicState`)
- `thermo`: Thermodynamic constants

**Example:**

```julia
using Breeze
using Oceananigans

# Create a model with OneMomentCloudMicrophysics
grid = RectilinearGrid(size=(8, 8, 8), extent=(1000, 1000, 1000))
microphysics = OneMomentCloudMicrophysics()
model = AtmosphereModel(grid; microphysics)

# Set initial conditions (avoiding set! to demonstrate manual update)
# After computing thermodynamic state, update microphysical fields
for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
    # Get thermodynamic state at this point
    𝒰 = diagnose_thermodynamic_state(i, j, k, grid, model.formulation, 
                                     microphysics, model.microphysical_fields,
                                     model.thermodynamics, 
                                     model.energy_density, model.moisture_density)
    
    # Adjust state (saturation adjustment)
    𝒰_adjusted = maybe_adjust_thermodynamic_state(𝒰, microphysics, model.thermodynamics)
    
    # Update microphysical fields
    update_microphysical_fields!(model.microphysical_fields, microphysics, 
                                i, j, k, grid, 
                                model.formulation.reference_state.density,
                                𝒰_adjusted, model.thermodynamics)
end
```

!!! note "Automatic Updates"
    When using `set!(model, ...)` to set initial conditions, `update_microphysical_fields!` is called automatically during `update_state!(model)`. The example above demonstrates manual usage for clarity.

**Implementation Notes:**
- Must be marked `@inline` for GPU compatibility
- Use `@inbounds` when accessing field arrays
- Extract moisture mass fractions from `state.moisture_mass_fractions`
- For schemes with precipitation, combine cloud and precipitation species (e.g., `qᶜˡ = qˡ + qʳ`)

### `maybe_adjust_thermodynamic_state(state, microphysics, thermo)`

**Signature:** `maybe_adjust_thermodynamic_state(state::AbstractThermodynamicState, microphysics, thermo) -> AbstractThermodynamicState`

**Purpose:** Adjusts the thermodynamic state according to microphysical processes (e.g., saturation adjustment, nucleation).

**Parameters:**
- `state`: Initial thermodynamic state
- `microphysics`: The microphysics scheme
- `thermo`: Thermodynamic constants

**Returns:** Adjusted thermodynamic state with updated moisture mass fractions.

**Example:**

This function is already extensively documented in the [saturation adjustment documentation](@ref). For saturation adjustment schemes, it solves for the equilibrium temperature and partitions moisture between vapor and condensates.

**Implementation Notes:**
- Must be marked `@inline` for GPU compatibility
- Should handle edge cases (e.g., absolute zero temperature)
- For schemes with no state adjustment, return the input state unchanged

### `compute_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, μ)`

**Signature:** `compute_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, μ) -> MoistureMassFractions`

**Purpose:** Computes the total moisture partitioning (vapor, liquid, ice) from microphysical fields, including both cloud and precipitation species.

**Parameters:**
- `i, j, k`: Grid indices
- `grid`: The model grid
- `microphysics`: The microphysics scheme
- `ρ`: Air density at this location
- `qᵗ`: Total moisture mass fraction
- `μ`: NamedTuple of microphysical fields

**Returns:** `MoistureMassFractions` with total vapor, liquid, and ice mass fractions.

**Example:**

For `OneMomentCloudMicrophysics`, this function combines cloud and precipitation species:

```julia
# For mixed-phase one-moment scheme
@inline @inbounds function compute_moisture_fractions(i, j, k, grid, bμp::MP1M, ρ, qᵗ, μ)
    # Extract cloud and precipitation species
    qᶜˡ = μ.qᶜˡ[i, j, k]  # Cloud liquid
    qᶜⁱ = μ.qᶜⁱ[i, j, k]  # Cloud ice
    qʳ = μ.ρqʳ[i, j, k] / ρ  # Rain (convert density to mass fraction)
    qˢ = μ.ρqˢ[i, j, k] / ρ  # Snow
    
    # Combine to get total liquid and ice
    qᵛ = μ.qᵛ[i, j, k]
    qˡ = qᶜˡ + qʳ  # Total liquid = cloud + rain
    qⁱ = qᶜⁱ + qˢ  # Total ice = cloud + snow
    
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end
```

**Implementation Notes:**
- Must be marked `@inline` for GPU compatibility
- Use `@inbounds` when accessing field arrays
- Convert density-weighted fields (e.g., `ρqʳ`) to mass fractions by dividing by `ρ`
- Return `MoistureMassFractions(qᵗ)` for schemes that treat all moisture as vapor

### `microphysical_velocities(microphysics, name)`

**Signature:** `microphysical_velocities(microphysics, name) -> Union{Nothing, NamedTuple{(:u, :v, :w), ...}}`

**Purpose:** Returns the fall velocities (sedimentation velocities) for a microphysical species, used in advection calculations.

**Parameters:**
- `microphysics`: The microphysics scheme
- `name`: Symbol naming the microphysical field (e.g., `:ρqʳ`, `:ρqˢ`)

**Returns:** 
- `nothing` if the species has no fall velocity
- A `NamedTuple` with `u`, `v`, `w` components (typically `u=0`, `v=0`, `w=<fall_velocity>`) if sedimentation is modeled

**Example:**

For precipitation species, fall velocities can be computed from CloudMicrophysics.jl:

```julia
using CloudMicrophysics.Microphysics1M: terminal_velocity

@inline function microphysical_velocities(bμp::OneMomentCloudMicrophysics, name::Symbol)
    if name === :ρqʳ
        # Return a function that computes rain fall velocity
        # This would be called during advection with microphysical properties
        return (u=0, v=0, w=terminal_velocity_rain)  # Simplified example
    elseif name === :ρqˢ
        return (u=0, v=0, w=terminal_velocity_snow)
    else
        return nothing
    end
end
```

!!! note "Current Status"
    Microphysical velocities are not yet fully wired up for `OneMomentCloudMicrophysics` in Breeze. This functionality is planned for future implementation. For now, schemes typically return `nothing`.

**Implementation Notes:**
- Must be marked `@inline` for GPU compatibility
- Fall velocities are typically only non-zero in the vertical (`w`) direction
- Velocities should be functions of microphysical properties (mass fraction, density, etc.)
- Return `nothing` for non-precipitating species or schemes without sedimentation

### `microphysical_tendency(i, j, k, grid, microphysics, args...)`

**Signature:** `microphysical_tendency(i, j, k, grid, microphysics, args...) -> Number`

**Purpose:** Computes the tendency (time derivative) of a prognostic microphysical field due to microphysical processes.

**Parameters:**
- `i, j, k`: Grid indices
- `grid`: The model grid
- `microphysics`: The microphysics scheme
- Additional arguments vary by implementation (typically include microphysical fields, thermodynamic state, etc.)

**Returns:** The tendency value (same units as the field per unit time).

**Example:**

For `ZeroMomentCloudMicrophysics`, the tendency removes precipitation:

```julia
@inline @inbounds function microphysical_tendency(i, j, k, grid, bμp::ZMCM, ::Val{:ρqᵗ}, μ, p, T, q, thermo)
    # Compute precipitation removal rate
    pᵣ = p[i, j, k]
    surface = equilibrated_surface(bμp.nucleation.equilibrium, T)
    ρ = density(pᵣ, T, q, thermo)
    qᵛ⁺ = saturation_specific_humidity(T, ρ, thermo, surface)
    qˡ = μ.qˡ[i, j, k]
    qⁱ = μ.qⁱ[i, j, k]
    ρᵣ = reference_density[i, j, k]
    
    # Remove precipitable condensate
    return ρᵣ * remove_precipitation(bμp.categories, qˡ, qⁱ, qᵛ⁺)
end
```

**Implementation Notes:**
- Must be marked `@inline` for GPU compatibility
- Use `@inbounds` when accessing field arrays
- Tendencies are added to advection terms in the prognostic equation
- Return `zero(grid)` for schemes with no prognostic fields or no tendencies
- The function signature is still evolving; check the source code for the current signature

### `compute_temperature(state, microphysics, thermo)`

**Signature:** `compute_temperature(state::AbstractThermodynamicState, microphysics, thermo) -> Number`

**Purpose:** Computes the temperature associated with a thermodynamic state after microphysical adjustment.

**Parameters:**
- `state`: Thermodynamic state (may be unadjusted)
- `microphysics`: The microphysics scheme
- `thermo`: Thermodynamic constants

**Returns:** Temperature in Kelvin.

**Implementation:**

This function typically calls `maybe_adjust_thermodynamic_state` and then extracts temperature:

```julia
@inline function compute_temperature(𝒰₀::AbstractThermodynamicState, microphysics, thermo)
    𝒰₁ = maybe_adjust_thermodynamic_state(𝒰₀, microphysics, thermo)
    return temperature(𝒰₁, thermo)
end
```

**Implementation Notes:**
- Must be marked `@inline` for GPU compatibility
- For schemes with no state adjustment, simply return `temperature(state, thermo)`

## Implementation Guidelines

### GPU Compatibility

All interface functions must be GPU-compatible:

1. **Use `@inline`**: All functions should be marked `@inline` to enable inlining in GPU kernels
2. **Use `@inbounds`**: When accessing field arrays, use `@inbounds` to skip bounds checking
3. **Avoid allocations**: Functions should not allocate memory
4. **Type stability**: Ensure functions are type-stable for optimal performance

### Type Stability

- Use concrete types where possible
- Avoid `Union` types in function signatures
- Let Julia infer types when possible; only annotate for dispatch

### Field Naming Conventions

Follow the notation from `docs/src/appendix/notation.md`:
- `qᵗ`: Total moisture mass fraction
- `qᵛ`: Vapor mass fraction
- `qˡ`: Liquid mass fraction
- `qⁱ`: Ice mass fraction
- `ρqʳ`: Rain density (prognostic)
- `ρqˢ`: Snow density (prognostic)
- `qᶜˡ`: Cloud liquid mass fraction (diagnostic)
- `qᶜⁱ`: Cloud ice mass fraction (diagnostic)

## Currently Implemented Schemes

| Scheme | Prognostic Fields | Description |
|--------|------------------|-------------|
| `Nothing` | `()` | No microphysics; all moisture treated as vapor |
| `SaturationAdjustment` | `()` | Instantaneous saturation adjustment (warm or mixed phase) |
| `ZeroMomentCloudMicrophysics` | `()` | Saturation adjustment + instant precipitation removal |
| `OneMomentCloudMicrophysics` (warm) | `(:ρqʳ,)` | Saturation adjustment + prognostic rain |
| `OneMomentCloudMicrophysics` (mixed) | `(:ρqʳ, :ρqˢ)` | Saturation adjustment + prognostic rain and snow |

## See Also

- [Example Implementation](@ref microphysics_example-section): A complete example of implementing a custom microphysics scheme
- [Saturation Adjustment](@ref saturation_adjustment-section): Detailed documentation on saturation adjustment algorithms
- [Microphysics Overview](@ref): User-facing documentation on available microphysics schemes

