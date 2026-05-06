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
| `microphysical_state` | `(microphysics, ρ, μ, 𝒰, velocities)` | **Primary interface**. Build scheme-specific state from scalars. |
| `grid_microphysical_state` | `(i, j, k, grid, microphysics, μ_fields, ρ, 𝒰, velocities)` | **Generic wrapper**. Extracts prognostics then calls gridless version. |

**Design principle**: Schemes implement the gridless `microphysical_state`; the grid-indexed version is generic.

Arguments:
- `microphysics`: The microphysics scheme
- `ρ`: Air density
- `μ`: NamedTuple of density-weighted prognostic scalars (e.g., `(ρqᶜˡ=..., ρqʳ=...)`)
- `𝒰`: Thermodynamic state
- `velocities`: NamedTuple of velocity components `(; u, v, w)` [m/s]. Used by schemes with aerosol activation (which depends on vertical velocity).

### Tendency Computation

| Function | Arguments | Description |
|----------|-----------|-------------|
| `microphysical_tendency` | `(microphysics, name, ρ, ℳ, 𝒰, constants)` | **State-based**. Compute tendency for variable `name`. |
| `compute_microphysical_tendencies!` | `(microphysics, model)` | **Model entry point**. Adds microphysics contributions to `Gⁿ`. |

**Design principle**: `compute_microphysical_tendencies!` is the only call the atmosphere model
makes into microphysics during tendency assembly — it runs *after* the per-tracer dynamics
kernels (advection + diffusion + forcing) have written `Gⁿ`, and adds microphysics on top via `+=`.

Schemes plug in by extending one of two methods:

- **Per-name (typical)** — extend `microphysical_tendency(microphysics, Val(name), ρ, ℳ, 𝒰, constants)`.
  The default `compute_microphysical_tendencies!` launches a single fused kernel that builds `ℳ`
  and `𝒰` once per cell and `+=`s `microphysical_tendency` for each prognostic name into the
  corresponding `G` field. This is the right extension point when the per-name tendencies don't
  share intermediate work.
- **Fused (bundle schemes)** — override `compute_microphysical_tendencies!(microphysics, model)`
  directly. Use this when a single bundle of process rates (e.g. ~14 rates in mixed-phase 1M)
  feeds multiple prognostic tendencies; computing the bundle once per cell rather than once per
  prognostic is a substantial GPU win.

The `name` argument is a `Val` type (e.g., `Val(:ρqᶜˡ)`) that dispatches to the appropriate tendency.
Velocity components are interpolated from cell faces to cell centers and passed as a NamedTuple
`(; u, v, w)` to the microphysical state for aerosol activation and other velocity-dependent processes.

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

### Velocity and Humidity Functions

| Function | Arguments | Description |
|----------|-----------|-------------|
| `microphysical_velocities` | `(microphysics, μ_fields, name)` | Return terminal velocities for advection of tracer `name` |
| `specific_humidity` | `(microphysics, model)` | Return vapor mass fraction field |

## Scheme Implementation Checklist

The interface is designed so that a **minimal implementation** enables parcel model support,
while **additional functions** are needed for full Eulerian (grid-based LES) support.

### Core Functions (Parcel Model)

These functions are sufficient to use a microphysics scheme with [`ParcelModel`](@ref):

| Function | Purpose |
|----------|---------|
| `microphysical_state(microphysics, ρ, μ, 𝒰, velocities)` | Build state from prognostics |
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
| `microphysical_velocities(microphysics, μ_fields, name)` | Terminal velocities for tracer advection |

**Why these are Eulerian-only**:
- **Field materialization**: Parcel models don't have fields; they store scalars directly in `ParcelState`.
- **Auxiliary updates**: Parcel models recompute derived quantities on-the-fly; they don't store them in fields.
- **Terminal velocities**: Sedimentation is a grid-based concept (advection through space). In parcel models,
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
| `microphysical_velocities` | — | ✓ | Sedimentation advection |
| `grid_microphysical_state` | — | — | Generic wrapper (don't override) |
| `compute_microphysical_tendencies!` | — | ✓* | Override for fused bundle schemes |
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

6. **Sedimentation is Eulerian**: Terminal velocities (`microphysical_velocities`) are only
   meaningful for grid-based simulations where tracers advect through space. In parcel models,
   precipitation loss should be modeled as a sink term in `microphysical_tendency`.
