# Fused-kernel implementation

The [previous page](example.md) walked through implementing a microphysics scheme by extending
`microphysical_tendency` once per prognostic variable. The default
`compute_microphysical_tendencies!` then launches a single fused kernel that builds the
microphysical state ``ℳ`` and thermodynamic state ``𝒰`` once per cell and `+=`s each
per-name tendency into the corresponding ``Gⁿ`` field.

This page walks through the **alternative** extension point: overriding
`compute_microphysical_tendencies!` directly so a scheme can compute one bundle of process
rates per cell and write to all its tendency fields in a single kernel pass.

## When to reach for this path

Use this path when a scheme's per-name tendencies share substantial intermediate work that
the per-name interface would force you to recompute. The motivating example is mixed-phase
non-equilibrium one-moment microphysics (`MPNE1M`): roughly 14 process rates (autoconversion,
accretion, evaporation, deposition, sublimation, melting, …) collectively determine 5
prognostic tendencies (``ρqᵛ``, ``ρqᶜˡ``, ``ρqᶜⁱ``, ``ρqʳ``, ``ρqˢ``). Computing the whole
bundle once and distributing it across the five ``G`` fields is dramatically faster on the
GPU than dispatching `microphysical_tendency` five times per cell — particularly because
each per-name dispatch would re-traverse the whole rate bundle.

For schemes whose per-name tendencies factor cleanly, prefer the [per-name path](example.md):
it is simpler, requires no kernel of your own, and benefits from the same one-build-per-cell
optimization through the default `compute_microphysical_tendencies!`.

## Running example

To make the comparison concrete, we re-implement the same `ExplicitMicrophysics` scheme from
the [per-name walkthrough](example.md) via the override path. The physics is identical;
only the extension point differs.

The shared portions of the implementation — the struct, prognostic field names, field
materialization, the state type, and `microphysical_state` — are unchanged from the per-name
walkthrough. We list them here so the page is self-contained:

```julia
using Breeze
using Oceananigans: CenterField
using Breeze.AtmosphereModels: AbstractMicrophysicalState
import Breeze.AtmosphereModels: prognostic_field_names,
                                 materialize_microphysical_fields,
                                 microphysical_state

struct ExplicitMicrophysics{FT}
    vapor_to_liquid :: FT
    vapor_to_ice :: FT
end

prognostic_field_names(::ExplicitMicrophysics) = (:ρqᵛ, :ρqˡ, :ρqⁱ)

function materialize_microphysical_fields(::ExplicitMicrophysics, grid, boundary_conditions)
    ρqᵛ = CenterField(grid; boundary_conditions=boundary_conditions.ρqᵛ)
    ρqˡ = CenterField(grid; boundary_conditions=boundary_conditions.ρqˡ)
    ρqⁱ = CenterField(grid; boundary_conditions=boundary_conditions.ρqⁱ)
    qᵛ = CenterField(grid)
    return (; ρqᵛ, ρqˡ, ρqⁱ, qᵛ)
end

struct ExplicitMicrophysicsState{FT} <: AbstractMicrophysicalState{FT}
    qᵛ :: FT
    qˡ :: FT
    qⁱ :: FT
end

function microphysical_state(::ExplicitMicrophysics, ρ, μ::NamedTuple, 𝒰, velocities)
    qᵛ = μ.ρqᵛ / ρ
    qˡ = μ.ρqˡ / ρ
    qⁱ = μ.ρqⁱ / ρ
    return ExplicitMicrophysicsState(qᵛ, qˡ, qⁱ)
end
```

## Bundle the per-name tendencies

In the [per-name walkthrough](example.md), each of the three prognostic variables had its
own `microphysical_tendency` method. Notice that the liquid and ice tendencies each recompute
`temperature(𝒰, constants)` independently, and the vapor tendency calls back into the
liquid and ice methods — so a per-name kernel dispatch evaluates `temperature` three times
per cell. For this small scheme the overhead is modest, but the structure scales poorly:
each new prognostic adds another redundant pass through the same intermediates.

The override path begins by defining a single helper that returns all three tendencies as
a NamedTuple, with the shared intermediates computed exactly once:

```@example fused_microphysics_example
using Breeze.Thermodynamics: temperature, saturation_specific_humidity,
                              PlanarLiquidSurface, PlanarIceSurface

@inline function explicit_microphysics_tendencies(em::ExplicitMicrophysics, ρ, ℳ, 𝒰, constants)
    T = temperature(𝒰, constants)
    q⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    q⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())

    Sˡ = ρ * (ℳ.qᵛ - q⁺ˡ) / em.vapor_to_liquid
    Sⁱ = ρ * (ℳ.qᵛ - q⁺ⁱ) / em.vapor_to_ice
    Sᵛ = -Sˡ - Sⁱ

    return (ρqᵛ = Sᵛ, ρqˡ = Sˡ, ρqⁱ = Sⁱ)
end
```

The helper is gridless: it works equally well in a parcel model context. If a parcel model
needs the per-name interface, the helper can also be wrapped:

```julia
import Breeze.AtmosphereModels: microphysical_tendency

@inline microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqᵛ}, ρ, ℳ, 𝒰, c) =
    explicit_microphysics_tendencies(em, ρ, ℳ, 𝒰, c).ρqᵛ
@inline microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqˡ}, ρ, ℳ, 𝒰, c) =
    explicit_microphysics_tendencies(em, ρ, ℳ, 𝒰, c).ρqˡ
@inline microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqⁱ}, ρ, ℳ, 𝒰, c) =
    explicit_microphysics_tendencies(em, ρ, ℳ, 𝒰, c).ρqⁱ
```

Wrapping is optional: the override below is sufficient for `AtmosphereModel`. Provide the
per-name methods only if the scheme is also used from a [`ParcelModel`](@ref) or from
state-based unit tests.

## Write the fused kernel

The fused kernel is responsible for everything the default kernel does — reading density,
extracting the prognostic moisture, reconstructing the thermodynamic state ``𝒰``, building
the microphysical state ``ℳ`` — and then for `+=`ing the bundled tendencies into all
`Gⁿ` fields:

```julia
using KernelAbstractions: @kernel, @index

@kernel function _compute_explicit_microphysics_tendencies!(Gρqᵛ, Gρqˡ, Gρqⁱ,
                                                            grid, microphysics, dynamics, formulation,
                                                            constants, specific_prognostic_moisture,
                                                            microphysical_fields)
    i, j, k = @index(Global, NTuple)

    ρ_field = Breeze.AtmosphereModels.dynamics_density(dynamics)
    @inbounds ρ = ρ_field[i, j, k]
    @inbounds qᵛ = specific_prognostic_moisture[i, j, k]

    # Reconstruct moisture fractions and thermodynamic state at this cell.
    q = Breeze.AtmosphereModels.grid_moisture_fractions(i, j, k, grid, microphysics,
                                                         ρ, qᵛ, microphysical_fields)
    𝒰 = Breeze.AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

    # Build the microphysical state directly to avoid the velocity interpolation
    # in `grid_microphysical_state` (this scheme's `microphysical_state` does not use velocities).
    @inbounds qᵛˢ = microphysical_fields.ρqᵛ[i, j, k] / ρ
    @inbounds qˡ  = microphysical_fields.ρqˡ[i, j, k] / ρ
    @inbounds qⁱ  = microphysical_fields.ρqⁱ[i, j, k] / ρ
    ℳ = ExplicitMicrophysicsState(qᵛˢ, qˡ, qⁱ)

    G = explicit_microphysics_tendencies(microphysics, ρ, ℳ, 𝒰, constants)

    @inbounds Gρqᵛ[i, j, k] += G.ρqᵛ
    @inbounds Gρqˡ[i, j, k] += G.ρqˡ
    @inbounds Gρqⁱ[i, j, k] += G.ρqⁱ
end
```

Two points worth highlighting:

1. The kernel uses several non-public internals from `Breeze.AtmosphereModels`
   (`dynamics_density`, `grid_moisture_fractions`, `diagnose_thermodynamic_state`,
   `specific_prognostic_moisture`). These are stable enough to override against, but they
   are not part of the public API — track them by following `MPNE1M`'s override in
   `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl`.
2. We *do not* call `grid_microphysical_state` because we want to skip its velocity
   interpolation: this scheme's `microphysical_state` ignores velocities. Schemes that
   need velocity-dependent activation should call `grid_microphysical_state` instead,
   passing `transport_velocities(model)` from the override below.

## Override `compute_microphysical_tendencies!`

The override threads the model fields the kernel needs and launches it:

```julia
using Oceananigans.Utils: launch!
import Breeze.AtmosphereModels: compute_microphysical_tendencies!

function compute_microphysical_tendencies!(microphysics::ExplicitMicrophysics, model)
    grid = model.grid
    arch = grid.architecture
    G = model.timestepper.Gⁿ

    launch!(arch, grid, :xyz, _compute_explicit_microphysics_tendencies!,
            G.ρqᵛ, G.ρqˡ, G.ρqⁱ,
            grid, microphysics, model.dynamics, model.formulation,
            model.thermodynamic_constants,
            Breeze.AtmosphereModels.specific_prognostic_moisture(model),
            model.microphysical_fields)

    return nothing
end
```

The default `compute_microphysical_tendencies!` falls back to the per-name path; the more
specific dispatch on `::ExplicitMicrophysics` takes priority whenever the model carries this
scheme.

## Auxiliary, moisture, and saturation hooks (unchanged)

The remaining interface — `update_microphysical_auxiliaries!`, `moisture_fractions`, and
`maybe_adjust_thermodynamic_state` — is independent of how tendencies are dispatched, and
the implementations from the [per-name walkthrough](example.md) carry over verbatim:

```julia
using Breeze.Thermodynamics: MoistureMassFractions
import Breeze.AtmosphereModels: update_microphysical_auxiliaries!,
                                 moisture_fractions,
                                 maybe_adjust_thermodynamic_state

@inline function update_microphysical_auxiliaries!(μ, i, j, k, grid,
                                                    ::ExplicitMicrophysics,
                                                    ℳ::ExplicitMicrophysicsState,
                                                    ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    return nothing
end

@inline function moisture_fractions(::ExplicitMicrophysics, ℳ::ExplicitMicrophysicsState, qᵗ)
    return MoistureMassFractions(ℳ.qᵛ, ℳ.qˡ, ℳ.qⁱ)
end

@inline maybe_adjust_thermodynamic_state(𝒰, ::ExplicitMicrophysics, qᵗ, constants) = 𝒰
```

## Which path should I pick?

| Question | Per-name path | Override path |
|----------|:-------------:|:-------------:|
| Do per-name tendencies share intermediate work? | No | Yes |
| Is the scheme used from a `ParcelModel` or per-name unit tests? | Required | Optional wrappers |
| Do you want to own the launch and kernel? | No | Yes |
| Number of prognostic tendencies | Any | Most useful when ``\ge 3`` |

In practice: **start with the per-name path.** Move to the override path only when profiling
shows that redundant intermediates dominate the microphysics tendency cost, or when the
scheme structure is unambiguously a bundle of process rates feeding many prognostics —
`MPNE1M` and `WPNE2M` are the canonical cases. The two paths coexist deliberately, and a
scheme can supply per-name methods for parcel use *and* an override for grid use, as shown
above.
