# Fused-kernel Microphysics Implementation

This page covers the **bundled-rate** alternative to the per-name extension point in
[Example Microphysics Implementation](@ref): a scheme overrides
`compute_microphysical_tendencies!` directly so it can compute one bundle of process
rates per cell and write all its ``G`` fields in a single kernel pass.

## Which Path Should I Pick?

| Question | Per-name path | Bundled-rate path |
|----------|:-------------:|:-----------------:|
| Do per-name tendencies share intermediate work? | No | Yes |
| Used from a `ParcelModel` or per-name unit tests? | Required | Optional wrappers |
| Do you want to own the launch and kernel? | No | Yes |
| Number of prognostic tendencies | Any | Most useful when ``≥ 3`` |

**Start with the per-name path** in [Example Microphysics Implementation](@ref). The
default `compute_microphysical_tendencies!` already builds ``ℳ`` and ``𝒰`` once per cell,
so the per-name interface is not paying for redundant state. Move to the bundled-rate
path only when profiling shows redundant intermediates *within* the tendencies dominate —
the canonical cases are `MPNE1M` and `WPNE2M`, where ~14 process rates collectively
determine 5 prognostic tendencies and computing the bundle once per cell is a substantial
GPU win.

## Running Example

We re-implement the `ExplicitMicrophysics` scheme from the
[per-name walkthrough](example.md) via the bundled-rate path. The struct,
`prognostic_field_names`, `materialize_microphysical_fields`, `ExplicitMicrophysicsState`,
and `microphysical_state` are unchanged — copy them straight from
[Example Microphysics Implementation](@ref). Only `microphysical_tendency` and
`compute_microphysical_tendencies!` change.

## Bundling Tendencies

In the per-name walkthrough each prognostic gets its own `microphysical_tendency`
method. The liquid and ice methods each call `saturation_specific_humidity`
independently, and the vapor method delegates back into the liquid and ice methods —
so a given ``saturation_specific_humidity`` is computed twice and the vapor method
re-invokes both other dispatches. The structure scales poorly: each new prognostic adds
another redundant pass.

In the bundled-rate path we package all three tendencies into one gridless helper, with
the shared intermediates computed exactly once:

```julia
using Breeze.Thermodynamics: temperature, saturation_specific_humidity,
                              PlanarLiquidSurface, PlanarIceSurface

@inline function explicit_microphysics_tendencies(em::ExplicitMicrophysics, ρ, ℳ, 𝒰, constants)
    T = temperature(𝒰, constants)
    q⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    q⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())

    τᵛˡ = em.vapor_to_liquid
    τᵛⁱ = em.vapor_to_ice

    Sˡ = ρ * (ℳ.qᵛ - q⁺ˡ) / τᵛˡ
    Sⁱ = ρ * (ℳ.qᵛ - q⁺ⁱ) / τᵛⁱ
    Sᵛ = -Sˡ - Sⁱ                       # closes by conservation

    return (ρqᵛ = Sᵛ, ρqˡ = Sˡ, ρqⁱ = Sⁱ)
end
```

If a parcel model or state-based unit tests need the per-name interface, wrap the
helper:

```julia
using Breeze.AtmosphereModels: AtmosphereModels

@inline AtmosphereModels.microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqᵛ}, ρ, ℳ, 𝒰, c) =
    explicit_microphysics_tendencies(em, ρ, ℳ, 𝒰, c).ρqᵛ
# Same shape for Val{:ρqˡ} and Val{:ρqⁱ}.
```

Otherwise the override below is sufficient for `AtmosphereModel`.

## Writing the Fused Kernel

!!! warning "Non-public internals"
    The kernel below calls four `Breeze.AtmosphereModels` symbols that are not part of
    the public API: `dynamics_density`, `grid_moisture_fractions`,
    `diagnose_thermodynamic_state`, and `specific_prognostic_moisture`. They are stable
    enough to override against today, but the canonical reference is `MPNE1M`'s override
    in `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl` — track it if the
    internal signatures shift.

The kernel does what the default kernel does — read density, build ``ℳ`` and ``𝒰`` —
then `+=`s the bundled tendencies into every ``G`` field:

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

    q = Breeze.AtmosphereModels.grid_moisture_fractions(i, j, k, grid, microphysics,
                                                         ρ, qᵛ, microphysical_fields)
    𝒰 = Breeze.AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

    # Skip `grid_microphysical_state` — its velocity interpolation is dead weight here
    @inbounds qˡ = microphysical_fields.ρqˡ[i, j, k] / ρ
    @inbounds qⁱ = microphysical_fields.ρqⁱ[i, j, k] / ρ
    ℳ = ExplicitMicrophysicsState(qᵛ, qˡ, qⁱ)

    G = explicit_microphysics_tendencies(microphysics, ρ, ℳ, 𝒰, constants)

    @inbounds Gρqᵛ[i, j, k] += G.ρqᵛ
    @inbounds Gρqˡ[i, j, k] += G.ρqˡ
    @inbounds Gρqⁱ[i, j, k] += G.ρqⁱ
end
```

Schemes that need velocity-dependent activation should call `grid_microphysical_state`
instead of building ``ℳ`` by hand, passing `transport_velocities(model)` from the
override below.

## Overriding `compute_microphysical_tendencies!`

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

The default `compute_microphysical_tendencies!` falls back to the per-name path;
dispatch on `::ExplicitMicrophysics` takes priority when the model carries this scheme.

`update_microphysical_auxiliaries!`, `moisture_fractions`, and
`maybe_adjust_thermodynamic_state` are unchanged from the
[per-name walkthrough](example.md) — copy them over verbatim.
