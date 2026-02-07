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
| `microphysical_state` | `(microphysics, ρ, μ, 𝒰)` | **Primary interface**. Build scheme-specific state from scalars. |
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
| `microphysical_tendency` | `(microphysics, name, ρ, ℳ, 𝒰, constants)` | **State-based**. Compute tendency for variable `name`. |
| `grid_microphysical_tendency` | `(i, j, k, grid, microphysics, name, ρ, fields, 𝒰, constants)` | **Generic wrapper**. Builds `ℳ` and dispatches to state-based version. |

**Design principle**: Schemes implement the state-based version; grid-indexed is generic.

The `name` argument is a `Val` type (e.g., `Val(:ρqᶜˡ)`) that dispatches to the appropriate tendency.

### Moisture Fraction Computation

| Function | Arguments | Description |
|----------|-----------|-------------|
| `moisture_fractions` | `(microphysics, ℳ, qᵗ)` | **State-based**. Partition moisture into vapor, liquid, ice. |
| `grid_moisture_fractions` | `(i, j, k, grid, microphysics, ρ, qᵗ, μ_fields)` | **Generic wrapper**. Builds state and dispatches. |

**Note**: Non-equilibrium schemes don't need `𝒰` to build their state (they use prognostic fields).
Saturation adjustment schemes override `grid_moisture_fractions` directly since they read cloud
condensate from diagnostic fields.

### Thermodynamic Adjustment

| Function | Arguments | Description |
|----------|-----------|-------------|
| `maybe_adjust_thermodynamic_state` | `(𝒰, microphysics, qᵗ, constants)` | Apply saturation adjustment if scheme uses it. |

This function is fully gridless—it takes only scalar thermodynamic arguments.
Non-equilibrium schemes simply return `𝒰` unchanged. Saturation adjustment schemes perform
iterative adjustment to partition moisture between vapor and condensate.

### Auxiliary Field Updates

| Function | Arguments | Description |
|----------|-----------|-------------|
| `update_microphysical_auxiliaries!` | `(μ, i, j, k, grid, microphysics, ℳ, ρ, 𝒰, constants)` | **Single interface** for writing all auxiliary fields. |
| `update_microphysical_fields!` | `(μ, i, j, k, grid, microphysics, ρ, 𝒰, constants)` | **Orchestrating function**. Builds `ℳ` and calls the above. |

**Why `i, j, k` is needed**: Grid indices cannot be eliminated because:
1. Fields must be written at specific grid points
2. Some schemes need grid-dependent logic (e.g., `k == 1` for bottom boundary conditions in sedimentation)

**Argument ordering convention**:
- Mutating functions: mutated object first (`μ`), then indices (`i, j, k, grid`), then other arguments
- All mutating functions `return nothing`

### Field Materialization

| Function | Arguments | Description |
|----------|-----------|-------------|
| `prognostic_field_names` | `(microphysics)` | Return tuple of prognostic field names (e.g., `(:ρqᶜˡ, :ρqʳ)`) |
| `materialize_microphysical_fields` | `(microphysics, grid, bcs)` | Create all microphysical fields (prognostic + auxiliary) |

**Field categories created by `materialize_microphysical_fields`**:

| Category | Grid Location | Boundary Conditions | Examples |
|----------|---------------|---------------------|----------|
| Prognostic | `CenterField` | User-provided via `bcs` | `ρqᶜˡ`, `ρqʳ`, `ρnᶜˡ` |
| Auxiliary/Diagnostic | `CenterField` | None needed | `qᵛ`, `qˡ`, `qᶜˡ`, `qʳ` |
| Velocities | `ZFaceField` | `bottom=nothing` | `wʳ`, `wᶜˡ`, `wʳₙ` |

### Sedimentation Speed and Bulk Sedimentation Velocities

#### Notation and concepts

Individual hydrometeor sedimentation speeds (`𝕎ᶜˡ`, `𝕎ʳ`, `𝕎ᶜⁱ`, `𝕎ˢ`) are terminal velocities
parameterized by the microphysics scheme, stored as **positive magnitudes** (downward).
The term "sedimentation speed" is more general than "terminal velocity": a particle is always
sedimenting at its sedimentation speed, but reaches terminal velocity only after an acceleration
period. In practice, most microphysics parameterizations return terminal velocities, so the
individual values (`𝕎ʳ`, etc.) _are_ terminal velocities from a parameterization.

The **effective total water sedimentation speed** `𝕎ᵗ` is a mass-weighted average of the individual
sedimentation speeds, representing the aggregate sedimentation rate of total water.

#### The `sedimentation_speed` interface

| Function | Arguments | Description |
|----------|-----------|-------------|
| `sedimentation_speed` | `(microphysics, microphysical_fields, name)` | **Primary interface**: return positive sedimentation speed field for tracer `name`, or `nothing` |
| `total_water_sedimentation_speed_components` | `(microphysics, microphysical_fields)` | Return `(speed_field, humidity_field)` tuples for aggregate computation |
| `microphysical_velocities` | `(microphysics, microphysical_fields, name)` | **Generic wrapper** (don't override): converts sedimentation speed to negative velocity tuple via `NegatedField` |

**Design principle**: Schemes implement `sedimentation_speed`; the generic `microphysical_velocities`
wrapper calls `sedimentation_speed` and constructs a `(u=ZeroField(), v=ZeroField(), w=NegatedField(fs))`
tuple for the advection operator.

#### From individual sedimentation speeds to aggregate velocity

The effective total water sedimentation speed is a mass-weighted average of the liquid and ice
sedimentation speeds (cf. CliMA documentation, Section 3.4):

```math
\mathbb{W}^t = \frac{q^l \, \mathbb{W}^l + q^i \, \mathbb{W}^i}{q^t}
```

where the liquid and ice sedimentation speeds are themselves mass-weighted averages of their
sub-components:

```math
\mathbb{W}^l = \frac{q^{cl} \, \mathbb{W}^{cl} + q^r \, \mathbb{W}^r}{q^l}, \qquad
\mathbb{W}^i = \frac{q^{ci} \, \mathbb{W}^{ci} + q^s \, \mathbb{W}^s}{q^i}
```

In general, the kernel computes:

```math
\mathbb{W}^t = \frac{\sum_i \mathbb{W}_i \, q_i}{q^t}
```

where the sum runs over the `(speed_field, humidity_field)` pairs returned by
`total_water_sedimentation_speed_components`.

#### Concrete example: 2M warm-phase scheme

In the two-moment warm-phase scheme, both cloud liquid and rain contribute to total water
sedimentation:

- `sedimentation_speed(scheme, μ, Val(:ρqᶜˡ))` returns `μ.wᶜˡ` (cloud liquid mass-weighted terminal velocity)
- `sedimentation_speed(scheme, μ, Val(:ρqʳ))` returns `μ.wʳ` (rain mass-weighted terminal velocity)
- `total_water_sedimentation_speed_components(scheme, μ)` returns `((μ.wᶜˡ, μ.qᶜˡ), (μ.wʳ, μ.qʳ))`

The kernel then computes:

```math
\mathbb{W}^t = \frac{\mathbb{W}^{cl} \, q^{cl} + \mathbb{W}^r \, q^r}{q^t}
```

This produces a physically meaningful aggregate where rain dominates when ``q^r \gg q^{cl}``.

#### Model-level bulk sedimentation velocities

Precomputed aggregate sedimentation velocities are stored on the model as
`model.bulk_sedimentation_velocities`, a `NamedTuple` of velocity tuples. Currently this contains
only the total water velocity:

```julia
(ρqᵗ = (u=ZeroField(), v=ZeroField(), w=wᵗ),)
```

where `wᵗ` is a `ZFaceField` storing **negative** values (downward velocity, consistent with the
advection operator's convention). This field is updated during `update_state!` via
`update_bulk_sedimentation_velocities!`, which calls the `_compute_bulk_sedimentation_velocity!`
kernel.

### Specific Humidity

| Function | Arguments | Description |
|----------|-----------|-------------|
| `specific_humidity` | `(microphysics, model)` | Return vapor mass fraction field |

## Scheme Implementation Checklist

The interface is designed so that a **minimal implementation** enables parcel model support,
while **additional functions** are needed for full Eulerian (grid-based LES) support.

### Core Functions (Parcel Model)

These functions are sufficient to use a microphysics scheme with [`ParcelModel`](@ref):

| Function | Purpose |
|----------|---------|
| `microphysical_state(microphysics, ρ, μ, 𝒰)` | Build state from prognostics |
| `microphysical_tendency(microphysics, name, ρ, ℳ, 𝒰, constants)` | Compute tendencies |
| `moisture_fractions(microphysics, ℳ, qᵗ)` | Partition moisture (if generic doesn't work) |
| `prognostic_field_names(microphysics)` | List prognostic variables |

**Why this works**: Parcel models operate on scalar states at a single point.
They don't need grid indexing, field materialization, or auxiliary field updates.
The gridless interface is exactly what parcel dynamics requires.

### Eulerian-Only Functions (Grid-Based LES)

These additional functions are required for full [`AtmosphereModel`](@ref) support:

| Function | Purpose |
|----------|---------|
| `materialize_microphysical_fields(microphysics, grid, bcs)` | Create prognostic + auxiliary fields |
| `update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics, ℳ, ρ, 𝒰, constants)` | Update auxiliary fields at grid points |
| `sedimentation_speed(microphysics, μ_fields, name)` | Positive sedimentation speed for tracer advection |
| `total_water_sedimentation_speed_components(microphysics, μ_fields)` | Component `(speed, humidity)` pairs for aggregate velocity |

**Why these are Eulerian-only**:
- **Field materialization**: Parcel models don't have fields; they store scalars directly in `ParcelState`.
- **Auxiliary updates**: Parcel models recompute derived quantities on-the-fly; they don't store them in fields.
- **Sedimentation speeds**: Sedimentation is a grid-based concept (advection through space). In parcel models,
  sedimentation would be modeled as a mass sink in `microphysical_tendency`, not as spatial transport.

### Summary Table

| Function | Parcel | Eulerian | Notes |
|----------|:------:|:--------:|-------|
| `microphysical_state` | ✓ | ✓ | Core interface |
| `microphysical_tendency` | ✓ | ✓ | Core interface |
| `moisture_fractions` | ✓ | ✓ | Often use generic fallback |
| `prognostic_field_names` | ✓ | ✓ | Required for both |
| `materialize_microphysical_fields` | — | ✓ | Fields for grid storage |
| `update_microphysical_auxiliaries!` | — | ✓ | Write to diagnostic fields |
| `sedimentation_speed` | — | ✓ | Positive sedimentation speed per tracer |
| `total_water_sedimentation_speed_components` | — | ✓ | Component pairs for aggregate velocity |
| `grid_microphysical_state` | — | — | Generic wrapper (don't override) |
| `grid_microphysical_tendency` | — | — | Generic wrapper (don't override) |
| `microphysical_velocities` | — | — | Generic wrapper (don't override) |
| `grid_moisture_fractions` | — | ✓* | Override for saturation adjustment |
| `maybe_adjust_thermodynamic_state` | — | ✓* | Override for saturation adjustment |

*Only needed for saturation adjustment schemes.

### Saturation Adjustment Schemes

Saturation adjustment schemes have some additional requirements:

| Function | Purpose |
|----------|---------|
| `grid_moisture_fractions(...)` | Override to read from diagnostic fields |
| `maybe_adjust_thermodynamic_state(...)` | Perform saturation adjustment |

These are needed because saturation adjustment schemes diagnose cloud condensate from
thermodynamic state rather than prognosing it.

## State Types

Built-in state types that schemes can use or extend:

| Type | Fields | Use case |
|------|--------|----------|
| `NothingMicrophysicalState{FT}` | None | No prognostic microphysics |
| `WarmRainState{FT}` | `qᶜˡ`, `qʳ` | Cloud liquid and rain |

Schemes may define their own state types inheriting from `AbstractMicrophysicalState{FT}`.

## Design Principles

1. **Gridless core**: Tendency and moisture fraction computations are gridless (state-based).
   Grid-indexed wrappers handle field extraction. This enables parcel model support with
   minimal implementation.

2. **Layered complexity**: The interface is structured so that:
   - **Minimal implementation** (4 functions) → parcel model support
   - **Full implementation** (7+ functions) → Eulerian LES support

   This allows rapid prototyping of new schemes in parcel models before investing in
   full grid infrastructure.

3. **Generic wrappers**: Most grid-indexed functions are generic and don't need scheme-specific
   implementations. Schemes only implement the gridless versions.

4. **Consistent argument ordering**: Mutating functions place the mutated object first, then
   grid indices, then other arguments.

5. **Explicit returns**: All mutating functions `return nothing`.

6. **Sedimentation is Eulerian**: Sedimentation speeds (`sedimentation_speed`,
   `total_water_sedimentation_speed_components`) are only meaningful for grid-based simulations
   where tracers advect through space. In parcel models, precipitation loss should be modeled
   as a sink term in `microphysical_tendency`.
