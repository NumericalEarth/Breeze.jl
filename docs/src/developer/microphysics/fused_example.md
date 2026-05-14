# Fused-kernel Microphysics Implementation

This page walks through the **alternative** to the per-name extension point covered in
[Example Microphysics Implementation](example.md): overriding
`compute_microphysical_tendencies!` directly so a scheme can compute one bundle of
process rates per cell and write to all its tendency fields in a single kernel pass.

To make the comparison concrete, we re-implement the same `ExplicitMicrophysics` scheme
— droplet and ice particle nucleation with constant-rate relaxation of specific humidity
to saturation — via the override path. The physics is identical; only the extension
point differs.

## When to Reach for This Path

Use the override path when a scheme's per-name tendencies share substantial intermediate
work that the per-name interface would force you to recompute. The motivating example is
mixed-phase non-equilibrium one-moment microphysics (`MPNE1M`): roughly 14 process rates
(autoconversion, accretion, evaporation, deposition, sublimation, melting, ...)
collectively determine 5 prognostic tendencies (``ρqᵛ``, ``ρqᶜˡ``, ``ρqᶜⁱ``, ``ρqʳ``,
``ρqˢ``). Computing the whole bundle once and distributing it across the five ``G``
fields is dramatically faster on the GPU than dispatching `microphysical_tendency` five
times per cell, since each per-name dispatch would re-traverse the whole rate bundle.

For schemes whose per-name tendencies factor cleanly, prefer the
[per-name path](example.md): it is simpler, requires no kernel of your own, and benefits
from the same one-build-per-cell optimization through the default
`compute_microphysical_tendencies!`.

## Running Example

The struct, prognostic field names, field materialization, microphysical state type, and
`microphysical_state` method are unchanged from the [per-name walkthrough](example.md).
We list them here so the page is self-contained:

```julia
using Breeze
using Oceananigans: CenterField
using Breeze.AtmosphereModels: AtmosphereModels, AbstractMicrophysicalState

struct ExplicitMicrophysics{FT}
    vapor_to_liquid :: FT
    vapor_to_ice :: FT
end

AtmosphereModels.prognostic_field_names(::ExplicitMicrophysics) = (:ρqᵛ, :ρqˡ, :ρqⁱ)

function AtmosphereModels.materialize_microphysical_fields(::ExplicitMicrophysics, grid, boundary_conditions)
    # Prognostic fields (density-weighted)
    ρqᵛ = CenterField(grid; boundary_conditions=boundary_conditions.ρqᵛ)
    ρqˡ = CenterField(grid; boundary_conditions=boundary_conditions.ρqˡ)
    ρqⁱ = CenterField(grid; boundary_conditions=boundary_conditions.ρqⁱ)

    # Diagnostic field (specific humidity)
    qᵛ = CenterField(grid)

    return (; ρqᵛ, ρqˡ, ρqⁱ, qᵛ)
end

struct ExplicitMicrophysicsState{FT} <: AbstractMicrophysicalState{FT}
    qᵛ :: FT
    qˡ :: FT
    qⁱ :: FT
end

function AtmosphereModels.microphysical_state(::ExplicitMicrophysics, ρ, μ::NamedTuple, 𝒰, velocities)
    # Convert density-weighted prognostics to specific quantities
    qᵛ = μ.ρqᵛ / ρ
    qˡ = μ.ρqˡ / ρ
    qⁱ = μ.ρqⁱ / ρ
    return ExplicitMicrophysicsState(qᵛ, qˡ, qⁱ)
end
```

## Bundling Tendencies

The per-name walkthrough splits the three tendency functions across three
`microphysical_tendency` methods; the liquid and ice methods each compute
`temperature(𝒰, constants)`, and the vapor method delegates back to both, so the same
intermediates are evaluated redundantly across methods. For this small scheme the
overhead is modest, but the structure scales poorly: each new prognostic adds another
redundant pass through the same intermediates.

In the override path we package all three tendencies into a single helper, with the
shared intermediates computed exactly once:

```julia
using Breeze.Thermodynamics: temperature, saturation_specific_humidity,
                              PlanarLiquidSurface, PlanarIceSurface

@inline function explicit_microphysics_tendencies(em::ExplicitMicrophysics, ρ, ℳ, 𝒰, constants)
    T = temperature(𝒰, constants)
    q⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    q⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())

    τᵛˡ = em.vapor_to_liquid
    τᵛⁱ = em.vapor_to_ice

    # Relaxation toward liquid and ice saturation
    Sˡ = ρ * (ℳ.qᵛ - q⁺ˡ) / τᵛˡ
    Sⁱ = ρ * (ℳ.qᵛ - q⁺ⁱ) / τᵛⁱ
    # Vapor tendency closes by conservation
    Sᵛ = -Sˡ - Sⁱ

    return (ρqᵛ = Sᵛ, ρqˡ = Sˡ, ρqⁱ = Sⁱ)
end
```

The helper is gridless: it works equally well in a parcel model context. If a parcel
model needs the per-name interface, the helper can also be wrapped:

```julia
using Breeze.AtmosphereModels: AtmosphereModels

@inline AtmosphereModels.microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqᵛ}, ρ, ℳ, 𝒰, c) =
    explicit_microphysics_tendencies(em, ρ, ℳ, 𝒰, c).ρqᵛ
@inline AtmosphereModels.microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqˡ}, ρ, ℳ, 𝒰, c) =
    explicit_microphysics_tendencies(em, ρ, ℳ, 𝒰, c).ρqˡ
@inline AtmosphereModels.microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqⁱ}, ρ, ℳ, 𝒰, c) =
    explicit_microphysics_tendencies(em, ρ, ℳ, 𝒰, c).ρqⁱ
```

Wrapping is optional: the override below is sufficient for `AtmosphereModel`. Provide
the per-name methods only if the scheme is also used from a [`ParcelModel`](@ref) or
from state-based unit tests.

## Writing the Fused Kernel

The fused kernel is responsible for everything the default kernel does — reading
density, extracting the prognostic moisture, reconstructing the thermodynamic state
``𝒰``, building the microphysical state ``ℳ`` — and then for ``+=``ing the bundled
tendencies into all ``Gⁿ`` fields:

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

    # Reconstruct moisture fractions and thermodynamic state at this cell
    q = Breeze.AtmosphereModels.grid_moisture_fractions(i, j, k, grid, microphysics,
                                                         ρ, qᵛ, microphysical_fields)
    𝒰 = Breeze.AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

    # Build the microphysical state directly to skip the velocity interpolation in
    # `grid_microphysical_state` — this scheme's `microphysical_state` does not use velocities
    @inbounds qˡ = microphysical_fields.ρqˡ[i, j, k] / ρ
    @inbounds qⁱ = microphysical_fields.ρqⁱ[i, j, k] / ρ
    ℳ = ExplicitMicrophysicsState(qᵛ, qˡ, qⁱ)

    G = explicit_microphysics_tendencies(microphysics, ρ, ℳ, 𝒰, constants)

    @inbounds Gρqᵛ[i, j, k] += G.ρqᵛ
    @inbounds Gρqˡ[i, j, k] += G.ρqˡ
    @inbounds Gρqⁱ[i, j, k] += G.ρqⁱ
end
```

Two points worth highlighting:

1. The kernel uses several non-public internals from `Breeze.AtmosphereModels`
   (`dynamics_density`, `grid_moisture_fractions`, `diagnose_thermodynamic_state`,
   `specific_prognostic_moisture`). These are stable enough to override against, but
   they are not part of the public API — the canonical reference is `MPNE1M`'s override
   in `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl`.
2. We deliberately *do not* call `grid_microphysical_state`, because we want to skip its
   velocity interpolation: this scheme's `microphysical_state` ignores velocities.
   Schemes that need velocity-dependent activation should call `grid_microphysical_state`
   instead, passing `transport_velocities(model)` from the override below.

## Overriding `compute_microphysical_tendencies!`

The override threads the model fields the kernel needs and launches it:

```julia
using Oceananigans.Utils: launch!
using Breeze.AtmosphereModels: AtmosphereModels

function AtmosphereModels.compute_microphysical_tendencies!(microphysics::ExplicitMicrophysics, model)
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

The default `compute_microphysical_tendencies!` falls back to the per-name path; the
more specific dispatch on `::ExplicitMicrophysics` takes priority whenever the model
carries this scheme.

## Updating Auxiliary Fields

Independent of how tendencies are dispatched, so identical to the
[per-name walkthrough](example.md):

```julia
using Breeze.AtmosphereModels: AtmosphereModels

@inline function AtmosphereModels.update_microphysical_auxiliaries!(μ, i, j, k, grid,
                                                                    ::ExplicitMicrophysics,
                                                                    ℳ::ExplicitMicrophysicsState,
                                                                    ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    return nothing
end
```

## Computing Moisture Fractions

```julia
using Breeze.Thermodynamics: MoistureMassFractions
using Breeze.AtmosphereModels: AtmosphereModels

@inline function AtmosphereModels.moisture_fractions(::ExplicitMicrophysics, ℳ::ExplicitMicrophysicsState, qᵛᵉ)
    return MoistureMassFractions(ℳ.qᵛ, ℳ.qˡ, ℳ.qⁱ)
end
```

## Thermodynamic Adjustment

```julia
using Breeze.AtmosphereModels: AtmosphereModels

@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰, ::ExplicitMicrophysics, qᵛᵉ, constants) = 𝒰
```

## Summary

The override path adds three steps to the per-name implementation listed in
[Example Microphysics Implementation](example.md):

1. **Bundle the per-name tendencies** into a single gridless helper that computes all
   shared intermediates exactly once.
2. **Write a fused `@kernel`** that builds ``ℳ`` and ``𝒰`` per cell and ``+=``s every
   ``G`` field in a single pass.
3. **Override `compute_microphysical_tendencies!(scheme, model)`** to `launch!` the
   kernel with the model fields it needs.

Per-name `microphysical_tendency` methods become optional — provide them only if the
scheme is also driven by a [`ParcelModel`](@ref) or by state-based unit tests.

The remaining hooks (`update_microphysical_auxiliaries!`, `moisture_fractions`,
`maybe_adjust_thermodynamic_state`) are unchanged from the per-name walkthrough.

## Which Path Should I Pick?

| Question | Per-name path | Override path |
|----------|:-------------:|:-------------:|
| Do per-name tendencies share intermediate work? | No | Yes |
| Is the scheme used from a `ParcelModel` or per-name unit tests? | Required | Optional wrappers |
| Do you want to own the launch and kernel? | No | Yes |
| Number of prognostic tendencies | Any | Most useful when ``≥ 3`` |

In practice: **start with the per-name path**. Move to the override path only when
profiling shows that redundant intermediates dominate the microphysics tendency cost,
or when the scheme structure is unambiguously a bundle of process rates feeding many
prognostics — `MPNE1M` and `WPNE2M` are the canonical cases. The two paths coexist
deliberately, and a scheme can supply per-name methods for parcel use *and* an override
for grid use, as shown above.
