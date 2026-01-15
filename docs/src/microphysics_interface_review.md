# Microphysics Interface Review

This document reviews the current microphysics interface in Breeze, describes its structure,
and identifies areas for potential improvement.

## Overview

The microphysics interface enables cloud microphysics schemes to work seamlessly in both
grid-based LES simulations and Lagrangian parcel models. The core abstraction is the
**microphysical state** (`ℳ`), which encapsulates local microphysical variables and
enables the same tendency functions to work across different dynamics.

## Interface Structure

### State Construction

| Function | Arguments | Description |
|----------|-----------|-------------|
| `microphysical_state(microphysics, ρ, μ, 𝒰)` | Scheme, density, prognostic NamedTuple, thermodynamic state | **Primary interface**. Build scheme-specific state from scalars. |
| `grid_microphysical_state(i, j, k, grid, microphysics, μ_fields, ρ, 𝒰)` | Grid indices, grid, scheme, fields, density, thermo state | **Generic wrapper**. Extracts prognostics via `extract_microphysical_prognostics` then calls gridless version. |

**Design principle**: Schemes implement the gridless version; the grid-indexed version is generic.

### Tendency Computation

| Function | Arguments | Description |
|----------|-----------|-------------|
| `microphysical_tendency(microphysics, name, ρ, ℳ, 𝒰, constants)` | Scheme, variable name (`Val`), density, state, thermo, constants | **State-based**. Compute tendency for variable `name`. |
| `grid_microphysical_tendency(i, j, k, grid, microphysics, name, ρ, fields, 𝒰, constants)` | Grid indices first, then same as above with fields | **Generic wrapper**. Builds `ℳ` and dispatches to state-based version. |

**Design principle**: Schemes implement the state-based version; grid-indexed is generic.

### Moisture Fraction Computation

| Function | Arguments | Description |
|----------|-----------|-------------|
| `moisture_fractions(microphysics, ℳ, qᵗ)` | Scheme, state, total moisture | **State-based**. Partition moisture into vapor, liquid, ice. |
| `grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, μ_fields)` | Grid indices first, then scheme, density, total moisture, fields | **Generic wrapper**. Calls `microphysical_state(microphysics, ρ, μ, nothing)` then dispatches to `moisture_fractions`. |

**Note**: Non-equilibrium schemes don't need `𝒰` to build their state (they use prognostic fields).
Saturation adjustment schemes override `grid_moisture_fractions` directly since they read cloud
condensate from diagnostic fields.

### Auxiliary Field Updates

| Function | Arguments | Description |
|----------|-----------|-------------|
| `update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics, ℳ, ρ, 𝒰, constants)` | Fields (mutated), indices, scheme, state, density, thermo, constants | **Single interface** for writing all auxiliary fields. |
| `update_microphysical_fields!(μ, i, j, k, grid, microphysics, ρ, 𝒰, constants)` | Fields (mutated), indices, grid, scheme, density, thermo, constants | **Orchestrating function**. Builds `ℳ` and calls `update_microphysical_auxiliaries!`. |

**Why `i, j, k` is needed**: Grid indices cannot be eliminated because:
1. Fields must be written at specific grid points
2. Some schemes need grid-dependent logic (e.g., `k == 1` for bottom boundary conditions in sedimentation)

**Argument ordering convention**:
- Mutating functions: mutated object first (`μ`), then indices (`i, j, k, grid`), then other arguments
- All mutating functions `return nothing`

### Other Interface Functions

| Function | Description |
|----------|-------------|
| `prognostic_field_names(microphysics)` | Return tuple of prognostic field names (e.g., `(:ρqᶜˡ, :ρqʳ)`) |
| `materialize_microphysical_fields(microphysics, grid, bcs)` | Create all microphysical fields (prognostic + diagnostic) |
| `maybe_adjust_thermodynamic_state(state, microphysics, qᵗ, constants)` | Apply saturation adjustment if scheme uses it |
| `microphysical_velocities(microphysics, μ_fields, name)` | Return terminal velocities for advection of tracer `name` |
| `specific_humidity(microphysics, model)` | Return vapor mass fraction field |

## Scheme Implementation Guide

### What to implement for a new scheme

| Required? | Function | Purpose |
|-----------|----------|---------|
| **Required** | `microphysical_state(microphysics, ρ, μ, 𝒰)` | Build state from prognostics |
| **Required** | `microphysical_tendency(microphysics, name, ρ, ℳ, 𝒰, constants)` | Compute tendencies |
| **Required** | `prognostic_field_names(microphysics)` | List prognostic variables |
| **Required** | `materialize_microphysical_fields(microphysics, grid, bcs)` | Create fields |
| **Required** | `update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics, ℳ, ρ, 𝒰, constants)` | Update auxiliary fields |
| Often needed | `moisture_fractions(microphysics, ℳ, qᵗ)` | Partition moisture (if generic doesn't work) |
| SA schemes | `grid_moisture_fractions(...)` | Override for diagnostic-field reading |
| SA schemes | `maybe_adjust_thermodynamic_state(...)` | Perform saturation adjustment |

### State Types

Built-in state types:
- `NothingMicrophysicalState{FT}`: No prognostic microphysics
- `WarmRainState{FT}`: Cloud liquid (`qᶜˡ`) and rain (`qʳ`)

Schemes may define their own state types inheriting from `AbstractMicrophysicalState{FT}`.

## Areas for Improvement

### 1. Consolidate Redundant State Types

**Issue**: `WarmRainState` (in `microphysics_interface.jl`) and `WarmPhaseOneMomentState`
(in CloudMicrophysics extension) are nearly identical.

**Recommendation**: Use `WarmRainState` consistently, or merge them.

### 2. Reduce Number of Interface Functions

**Issue**: There are many interface functions, some of which may be redundant or could be
combined. The current interface has ~15 functions.

**Recommendation**: Continue consolidation. The recent merge of three hooks into
`update_microphysical_auxiliaries!` was a step in the right direction.

### 3. Unify Grid-Indexed Wrappers

**Current pattern**: Each function has separate grid-indexed and state-based versions.
The grid-indexed versions are mostly boilerplate.

**Recommendation**: Consider a macro or metaprogramming approach to generate grid-indexed
wrappers automatically. However, this may reduce readability.

### 4. Document the Saturation Adjustment Exception

**Issue**: Saturation adjustment schemes have a fundamentally different structure:
- Cloud condensate is diagnostic (from `𝒰`), not prognostic
- They must override `grid_moisture_fractions` to read diagnostic fields
- `maybe_adjust_thermodynamic_state` performs the adjustment

**Recommendation**: Add clear documentation explaining why SA schemes are different and
which functions they must override.

### 5. Consider Removing `microphysical_velocities`

**Issue**: Terminal velocities are written to fields in `update_microphysical_auxiliaries!`,
so `microphysical_velocities` might be redundant.

**Investigation needed**: Clarify the relationship between the velocity fields updated
in `update_microphysical_auxiliaries!` and the `microphysical_velocities` function used
for advection.

## Summary

The microphysics interface is well-structured around the core abstraction of a gridless
microphysical state (`ℳ`). The recent consolidation of field update hooks into a single
`update_microphysical_auxiliaries!` function simplified the interface.

**Key strengths**:
- Clear separation between state-based (gridless) and grid-indexed functions
- Generic wrappers reduce code duplication for most operations
- Consistent argument ordering for mutating functions

**Remaining complexity**:
- Saturation adjustment schemes require special handling
- Multiple state types for similar purposes (e.g., `WarmRainState` vs `WarmPhaseOneMomentState`)

The interface is parsimonious for non-equilibrium schemes (implement 5 functions) but
more complex for saturation adjustment schemes due to their fundamentally different
structure.
