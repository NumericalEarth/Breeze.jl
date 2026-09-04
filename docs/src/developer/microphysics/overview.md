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
  share intermediate work. See [Per-name Implementation](@ref) for a worked
  example.
- **Fused (bundle schemes)** — override `compute_microphysical_tendencies!(microphysics, model)`
  directly. Use this when a single bundle of process rates (e.g. ~14 rates in mixed-phase 1M)
  feeds multiple prognostic tendencies; computing the bundle once per cell rather than once per
  prognostic is a substantial GPU win. See
  [Fused-kernel Microphysics Implementation](@ref) for a worked example.

The `name` argument is a `Val` type (e.g., `Val(:ρqᶜˡ)`) that dispatches to the appropriate tendency.
Velocity components are interpolated from cell faces to cell centers and passed as a NamedTuple
`(; u, v, w)` to the microphysical state for aerosol activation and other velocity-dependent processes.

### Moisture Fraction Computation

| Function | Arguments | Description |
|----------|-----------|-------------|
| `moisture_fractions` | `(microphysics, ℳ, qᵛᵉ)` | **State-based**. Partition moisture into vapor, liquid, ice. |
| `grid_moisture_fractions` | `(i, j, k, grid, microphysics, ρ, qᵛᵉ, μ_fields)` | **Generic wrapper**. Builds state and dispatches. |

The argument `qᵛᵉ` is the scheme-dependent specific moisture: vapor (``qᵛ``) for
non-equilibrium schemes, or equilibrium moisture (``qᵉ = qᵛ + qᶜˡ``) for saturation
adjustment schemes.

**Note**: Non-equilibrium schemes don't need `𝒰` to build their state (they use prognostic fields).
Saturation adjustment schemes override `grid_moisture_fractions` directly since they read cloud
condensate from diagnostic fields.

### Thermodynamic Adjustment

| Function | Arguments | Description |
|----------|-----------|-------------|
| `maybe_adjust_thermodynamic_state` | `(𝒰, microphysics, qᵛᵉ, constants)` | Apply saturation adjustment if scheme uses it. |

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
| Velocities | `ZFaceField` | `bottom=nothing` | `wʳ`, `wᶜˡ`, `wⁿʳ` |

### Sedimentation

| Function | Arguments | Description |
|----------|-----------|-------------|
| `sedimentation_velocity` | `(microphysics, microphysical_fields, name)` | **Primary interface**: return the vertical sedimentation velocity field for tracer `name`, or `nothing` |
| `condensate_phase` | `(microphysics, name)` | Return `Val(:liquid)`, `Val(:ice)`, or a liquid fraction, the thermodynamic phase of condensate mass `name`; required for every sedimenting mass |
| `microphysical_velocities` | `(microphysics, microphysical_fields, name)` | **Generic wrapper** (don't override): wraps the sedimentation velocity in a velocity tuple |

**Design principle**: Schemes implement `sedimentation_velocity` (how fast a tracer falls) and
`condensate_phase` (which latent heat its mass carries); the generic `microphysical_velocities`
wrapper calls `sedimentation_velocity` and constructs a `(u=ZeroField(), v=ZeroField(), w=w)`
tuple for the advection operator.

CloudMicrophysics returns positive downward terminal-speed magnitudes `𝕎ˣ`. Breeze uses a
signed vertical coordinate that is positive upward, so the corresponding velocity is
`wˣ = -𝕎ˣ` and falling hydrometeors have `wˣ < 0`. `write_sedimentation_velocity!` stores the
signed velocity at the cell's bottom face and applies the precipitation boundary condition at
`k = 1`.

#### Sedimentation constituents

At construction the model resolves, once, a tuple of `(; w, q, ρq, phase, advection)`
constituents, one for every name in `condensate_field_names` with a `sedimentation_velocity`: the
velocity field, the specific-humidity field and the prognostic density behind it, the phase tag,
and the advection scheme that transports the tracer (`model.sedimentation_constituents`, `()`
when nothing sediments). Number tracers (e.g.
`:ρnᶜˡ`) and non-additive particle properties (P3's rime mass `ρqᶠ`, a portion of `ρqⁱ`, and
rime volume `ρbᶠ`) are not condensate masses and are never consulted. Condensate that does not
sediment, such as cloud condensate diagnosed by saturation adjustment, moves no mass and needs
no declaration. A sedimenting mass without a `condensate_phase` is an error at construction,
since its latent heat would otherwise be left behind.

!!! note "Velocity and phase are independent"
    P3's liquid on ice `ρqʷⁱ` is liquid water riding on an ice particle: it falls at `wⁱ`, yet
    `moisture_fractions` counts it in the thermodynamic liquid fraction `qˡ = qᶜˡ + qʳ + qʷⁱ`
    because no fusion enthalpy has been released for it. P3 therefore declares
    `sedimentation_velocity(…, Val(:ρqʷⁱ)) = μ.wⁱ` and `condensate_phase(…, Val(:ρqʷⁱ)) = Val(:liquid)`:
    one tracer with two independent properties.

!!! note "Mixed-phase particles"
    A part-ice, part-liquid particle is two condensate masses sharing a fall speed, which is what
    the independence above buys: P3 carries ice `ρqⁱ` and the liquid on it `ρqʷⁱ`, both falling at
    `wⁱ`, each declaring the phase whose enthalpy it holds. Splitting the mass beats blending the
    enthalpy, since freezing that liquid is a process with its own rate that must move mass
    between them.

    A single mass of mixed composition may instead declare a liquid fraction,
    `condensate_phase(…, ::Val{:ρqˣ}) = 0.3`. That is exact, not an interpolation: the content is
    a directional derivative in composition space, so a mass leaving along `f eˡ + (1 - f) eⁱ`
    carries `f χˡ + (1 - f) χⁱ`. No scheme needs this yet.

#### Sedimentation of the thermodynamic variables

The *mass* sediments by ordinary tracer advection, untouched by any of this: `scalar_tendency`
adds the fall velocity to the transport velocity and hands the sum to `div_ρUc` with the tracer's
own scheme, so bounds- and positivity-preserving schemes limit the falling humidity exactly as
they would without sedimentation. What follows weights those same fluxes to carry the
*thermodynamic variable's* share; it forms no flux of its own.

The thermodynamic-variable tendencies consume the constituents: each constituent's
sedimentation mass flux — the advective flux of its humidity at the combined resolved and fall
velocity minus the flux at the resolved velocity alone, computed with the same advection scheme
that transports the tracer's mass (for bounds-preserving WENO, from the same per-cell limited
reconstructions the tracer operator uses, so the limiter never separates heat from water at
cloud and precipitation edges) — is binned by its phase and weighted by the content it delivers
to the cell. The falling mass carries its enthalpy and each cell converts what it gains or loses
locally: a flux out of a cell removes the cell's own ``χˣ``, the partial derivative of the
specific variable with respect to that condensate mass fraction at fixed temperature, so the
cell the condensate leaves keeps its temperature; a flux in delivers ``χˣ`` plus ``∂φ/∂h`` times
the enthalpy ``hˣ - hʳ`` the arriving mass brings in excess of the receiving cell's. What takes
up the departed mass is the dynamics' call (`sedimentation_replacement`): dry air on the
anelastic core, whose total density is fixed so that ``qᵈ`` absorbs the change; the local
mixture on the compressible core, whose prognostic dry density has no sedimentation source, so
that the diagnosed total density falls with the condensate and every mass fraction
renormalizes. The enthalpy is then ``(cˣ - cʳ) T - (ℒˣᵣ - ℒʳ)`` (``(cˣ - cᵖᵈ) T - ℒˣᵣ`` against
dry air, ``hˣ - (s - g z)`` against the mixture), which is also the content of `ρs`; with
``∂s/∂h = 1`` the sum collapses to the flux form and ``∫ρs`` is conserved. For `ρθ` the content
is ``∂θˡⁱ/∂qˣ`` along the same composition change (to leading order ``-ℒˣᵣ / (cᵖᵐ Π)``) with
``∂θˡⁱ/∂h = 1 / (cᵖᵐ Π)``, and must not collapse: that Jacobian varies with the Exner function,
so moving it between pressure levels would conserve ``∫ρθ``, which precipitation does not (heat
released at one pressure and absorbed at another). Both formulations respond in temperature
identically. The content fluxes ride the total-density-weighted mass flux the tracer tendency
applies, and the cell's coupling-to-total density ratio (one on the anelastic core,
``qᵈ = ρᵈ / ρ`` on the compressible core) converts the change of the specific variable into that
of the coupling-weighted prognostic. Under adaptive
implicit vertical advection the tendency carries the content of the explicit fraction of each
mass flux only; between the tracers' implicit solves of a stage and the thermodynamic variable's
own, the time steppers call `implicit_sedimentation_step!`, which moves the content of the
remainder from the first-order fluxes the solves actually applied, at the solved state, so the
heat follows the mass at any fall Courant number and takes the same implicit transport and
diffusion as the rest of the field. Rain-out thus leaves latent warming aloft and pre-cools the
layer that later evaporates the arriving rain, the mechanism that builds cold pools.

### Bottom Precipitation Flux

`bottom_precipitation_flux(model)` returns the flux of precipitating moisture through the
bottom boundary [kg m⁻² s⁻¹, positive downward]. A scheme that implements
`sedimentation_velocity` and `condensate_phase` gets it for free: the default method sums the
bottom-face flux of every sedimentation constituent, evaluating each with the advection scheme
that transports that tracer, so the diagnostic agrees with the boundary flux the tendency
operator applies. Schemes that move precipitation by their own internal means (such as
`DCMIP2016KM`) override `bottom_precipitation_flux` directly instead.

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
| `microphysical_state(microphysics, ρ, μ, 𝒰, velocities)` | Build state from prognostics |
| `microphysical_tendency(microphysics, name, ρ, ℳ, 𝒰, constants)` | Compute tendencies |
| `moisture_fractions(microphysics, ℳ, qᵛᵉ)` | Partition moisture (if generic doesn't work) |
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
| `sedimentation_velocity(microphysics, μ_fields, name)` | Vertical sedimentation velocity per tracer |
| `condensate_phase(microphysics, name)` | Thermodynamic phase of each sedimenting condensate mass (`:liquid` or `:ice`) |

**Why these are Eulerian-only**:
- **Field materialization**: Parcel models don't have fields; they store scalars directly in `ParcelState`.
- **Auxiliary updates**: Parcel models recompute derived quantities on-the-fly; they don't store them in fields.
- **Sedimentation velocities**: Sedimentation is a grid-based concept (advection through space). In parcel models,
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
| `sedimentation_velocity` | — | ✓§ | Vertical sedimentation velocity per tracer |
| `condensate_phase` | — | ✓§ | Thermodynamic phase of each sedimenting condensate mass |
| `grid_microphysical_state` | — | — | Generic wrapper (don't override) |
| `compute_microphysical_tendencies!` | — | ✓† | Override for fused bundle schemes |
| `microphysical_velocities` | — | — | Generic wrapper (don't override) |
| `grid_moisture_fractions` | — | ✓‡ | Override for saturation adjustment |
| `maybe_adjust_thermodynamic_state` | — | ✓‡ | Override for saturation adjustment |

† Only needed for bundle/fused-kernel schemes (e.g. mixed-phase 1M).
‡ Only needed for saturation adjustment schemes.
§ Only needed when one or more prognostic species sediments; non-sedimenting schemes need
neither.

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

## Which Path Should I Pick?

| Question | Per-name path | Bundled-rate path |
|----------|:-------------:|:-----------------:|
| Do per-name tendencies share intermediate work? | No | Yes |
| Used from a `ParcelModel` or per-name unit tests? | Required | Optional wrappers |
| Do you want to own the launch and kernel? | No | Yes |
| Number of prognostic tendencies | Any | Most useful when ``≥ 3`` |

**Start with the per-name path** in [Per-name Implementation](@ref). The default
`compute_microphysical_tendencies!` already builds ``ℳ`` and ``𝒰`` once per cell, so the
per-name interface is not paying for redundant state.
Move to the [bundled-rate path](@ref "Fused-kernel Microphysics Implementation") only when
profiling shows redundant intermediates *within* the tendencies dominate — the canonical
cases are `MPNE1M` and `WPNE2M`, where ~14 process rates collectively determine 5 prognostic
tendencies and computing the bundle once per cell is a substantial GPU win.

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

6. **Sedimentation is Eulerian**: Sedimentation velocities (`sedimentation_velocity`) are only
   meaningful for grid-based simulations where tracers advect through space. In parcel models,
   precipitation loss should be modeled as a sink term in `microphysical_tendency`.
