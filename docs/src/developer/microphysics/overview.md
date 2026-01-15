# Microphysics Interface Overview

This document describes the interface for embedding microphysical processes into [`AtmosphereModel`](@ref).
The interface enables cloud microphysics schemes to work seamlessly in both grid-based LES simulations
and Lagrangian parcel models.

## Core Abstraction

The central abstraction is the **microphysical state** (`ℳ`), which encapsulates local microphysical
variables (specific humidities, number concentrations, etc.) at a single point. This state-based
design enables the same tendency and moisture fraction functions to work across different dynamics
without modification.

## Interface Structure

### State Construction

| Function | Arguments | Description |
|----------|-----------|-------------|
| [`microphysical_state`](@ref) | `(microphysics, ρ, μ, 𝒰)` | **Primary interface**. Build scheme-specific state from scalars. |
| `grid_microphysical_state` | `(i, j, k, grid, microphysics, μ_fields, ρ, 𝒰)` | **Generic wrapper**. Extracts prognostics then calls gridless version. |

**Design principle**: Schemes implement the gridless `microphysical_state`; the grid-indexed version is generic.

Arguments:
- `microphysics`: The microphysics scheme
- `ρ`: Air density
- `μ`: NamedTuple of density-weighted prognostic scalars (e.g., `(ρqᶜˡ=..., ρqʳ=...)`)
- `𝒰`: Thermodynamic state

### Tendency Computation

| Function | Arguments | Description |
|----------|-----------|-------------|
| [`microphysical_tendency`](@ref) | `(microphysics, name, ρ, ℳ, 𝒰, constants)` | **State-based**. Compute tendency for variable `name`. |
| [`grid_microphysical_tendency`](@ref) | `(i, j, k, grid, microphysics, name, ρ, fields, 𝒰, constants)` | **Generic wrapper**. Builds `ℳ` and dispatches to state-based version. |

**Design principle**: Schemes implement the state-based version; grid-indexed is generic.

The `name` argument is a `Val` type (e.g., `Val(:ρqᶜˡ)`) that dispatches to the appropriate tendency.

### Moisture Fraction Computation

| Function | Arguments | Description |
|----------|-----------|-------------|
| [`moisture_fractions`](@ref) | `(microphysics, ℳ, qᵗ)` | **State-based**. Partition moisture into vapor, liquid, ice. |
| [`grid_moisture_fractions`](@ref) | `(i, j, k, grid, microphysics, ρ, qᵗ, μ_fields)` | **Generic wrapper**. Builds state and dispatches. |

**Note**: Non-equilibrium schemes don't need `𝒰` to build their state (they use prognostic fields).
Saturation adjustment schemes override `grid_moisture_fractions` directly since they read cloud
condensate from diagnostic fields.

### Thermodynamic Adjustment

| Function | Arguments | Description |
|----------|-----------|-------------|
| [`maybe_adjust_thermodynamic_state`](@ref) | `(𝒰, microphysics, qᵗ, constants)` | Apply saturation adjustment if scheme uses it. |

This function is fully gridless—it takes only scalar thermodynamic arguments.
Non-equilibrium schemes simply return `𝒰` unchanged. Saturation adjustment schemes perform
iterative adjustment to partition moisture between vapor and condensate.

### Auxiliary Field Updates

| Function | Arguments | Description |
|----------|-----------|-------------|
| [`update_microphysical_auxiliaries!`](@ref) | `(μ, i, j, k, grid, microphysics, ℳ, ρ, 𝒰, constants)` | **Single interface** for writing all auxiliary fields. |
| [`update_microphysical_fields!`](@ref) | `(μ, i, j, k, grid, microphysics, ρ, 𝒰, constants)` | **Orchestrating function**. Builds `ℳ` and calls the above. |

**Why `i, j, k` is needed**: Grid indices cannot be eliminated because:
1. Fields must be written at specific grid points
2. Some schemes need grid-dependent logic (e.g., `k == 1` for bottom boundary conditions in sedimentation)

**Argument ordering convention**:
- Mutating functions: mutated object first (`μ`), then indices (`i, j, k, grid`), then other arguments
- All mutating functions `return nothing`

### Field Materialization

| Function | Arguments | Description |
|----------|-----------|-------------|
| [`prognostic_field_names`](@ref) | `(microphysics)` | Return tuple of prognostic field names (e.g., `(:ρqᶜˡ, :ρqʳ)`) |
| [`materialize_microphysical_fields`](@ref) | `(microphysics, grid, bcs)` | Create all microphysical fields (prognostic + auxiliary) |

**Field categories created by `materialize_microphysical_fields`**:

| Category | Grid Location | Boundary Conditions | Examples |
|----------|---------------|---------------------|----------|
| Prognostic | `CenterField` | User-provided via `bcs` | `ρqᶜˡ`, `ρqʳ`, `ρnᶜˡ` |
| Auxiliary/Diagnostic | `CenterField` | None needed | `qᵛ`, `qˡ`, `qᶜˡ`, `qʳ` |
| Velocities | `ZFaceField` | `bottom=nothing` | `wʳ`, `wᶜˡ`, `wʳₙ` |

### Velocity and Humidity Functions

| Function | Arguments | Description |
|----------|-----------|-------------|
| [`microphysical_velocities`](@ref) | `(microphysics, μ_fields, name)` | Return terminal velocities for advection of tracer `name` |
| [`specific_humidity`](@ref) | `(microphysics, model)` | Return vapor mass fraction field |

## Scheme Implementation Checklist

### Required Functions

| Function | Purpose |
|----------|---------|
| `microphysical_state(microphysics, ρ, μ, 𝒰)` | Build state from prognostics |
| `microphysical_tendency(microphysics, name, ρ, ℳ, 𝒰, constants)` | Compute tendencies |
| `prognostic_field_names(microphysics)` | List prognostic variables |
| `materialize_microphysical_fields(microphysics, grid, bcs)` | Create fields |
| `update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics, ℳ, ρ, 𝒰, constants)` | Update auxiliary fields |

### Often Needed

| Function | When to implement |
|----------|-------------------|
| `moisture_fractions(microphysics, ℳ, qᵗ)` | If generic version doesn't work for your scheme |

### Saturation Adjustment Schemes Only

| Function | Purpose |
|----------|---------|
| `grid_moisture_fractions(...)` | Override to read from diagnostic fields |
| `maybe_adjust_thermodynamic_state(...)` | Perform saturation adjustment |

## State Types

Built-in state types that schemes can use or extend:

| Type | Fields | Use case |
|------|--------|----------|
| `NothingMicrophysicalState{FT}` | None | No prognostic microphysics |
| `WarmRainState{FT}` | `qᶜˡ`, `qʳ` | Cloud liquid and rain |

Schemes may define their own state types inheriting from `AbstractMicrophysicalState{FT}`.

## Design Principles

1. **Gridless core**: Tendency and moisture fraction computations are gridless (state-based).
   Grid-indexed wrappers handle field extraction.

2. **Generic wrappers**: Most grid-indexed functions are generic and don't need scheme-specific
   implementations. Schemes only implement the gridless versions.

3. **Consistent argument ordering**: Mutating functions place the mutated object first, then
   grid indices, then other arguments.

4. **Explicit returns**: All mutating functions `return nothing`.
