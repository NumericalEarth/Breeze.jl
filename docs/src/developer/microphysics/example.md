# Per-name Implementation

This page walks through the implementation of a simple microphysics scheme to illustrate the developer interface.
We implement a scheme that represents droplet and ice particle nucleation with constant-rate
relaxation of specific humidity to saturation.

## Defining the Scheme

First, we define a struct to hold the scheme parameters:

```@example microphysics_example
using Breeze
using Breeze.AtmosphereModels: AtmosphereModels

struct ExplicitMicrophysics{FT}
    vapor_to_liquid :: FT
    vapor_to_ice :: FT
end
```

## Prognostic Field Names

This scheme is fully prognostic, meaning we carry vapor, liquid, and ice density as prognostic
variables:

```@example microphysics_example
AtmosphereModels.prognostic_field_names(::ExplicitMicrophysics) = (:ρqᵛ, :ρqˡ, :ρqⁱ)
```

!!! note "Field naming conventions"
    The names of prognostic fields defined by `prognostic_field_names` are crucial to the user
    interface, because users can interact with them and
    `set!` their initial conditions. Names should be concise, mathematical forms consistent with
    Breeze conventions (see the notation appendix).

## Materializing Fields

When we materialize the microphysics fields, we must include all prognostic fields plus any
diagnostic fields needed:

```@example microphysics_example
using Oceananigans: CenterField

function AtmosphereModels.materialize_microphysical_fields(::ExplicitMicrophysics, grid, boundary_conditions)
    ρqᵛ = CenterField(grid; boundary_conditions=boundary_conditions.ρqᵛ)
    ρqˡ = CenterField(grid; boundary_conditions=boundary_conditions.ρqˡ)
    ρqⁱ = CenterField(grid; boundary_conditions=boundary_conditions.ρqⁱ)
    qᵛ = CenterField(grid)
    return (; ρqᵛ, ρqˡ, ρqⁱ, qᵛ)
end
```

Note we include the diagnostic field `qᵛ` (vapor mass fraction) in addition to the three
prognostic fields.

## Building the Microphysical State

The microphysical state encapsulates local values for tendency computation:

```@example microphysics_example
using Breeze.AtmosphereModels: AbstractMicrophysicalState

struct ExplicitMicrophysicsState{FT} <: AbstractMicrophysicalState{FT}
    qᵛ :: FT
    qˡ :: FT
    qⁱ :: FT
end

function AtmosphereModels.microphysical_state(::ExplicitMicrophysics, ρ, μ::NamedTuple, 𝒰, velocities)
    # `velocities` is part of the interface for schemes that need aerosol activation; unused here
    qᵛ = μ.ρqᵛ / ρ
    qˡ = μ.ρqˡ / ρ
    qⁱ = μ.ρqⁱ / ρ
    return ExplicitMicrophysicsState(qᵛ, qˡ, qⁱ)
end
```

## Computing Tendencies

Tendencies are computed from the microphysical state. Each prognostic variable needs a
tendency method:

```@example microphysics_example
using Breeze.Thermodynamics: temperature, saturation_specific_humidity,
                              PlanarLiquidSurface, PlanarIceSurface

# Relaxation toward liquid saturation
@inline function AtmosphereModels.microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqˡ}, ρ, ℳ, 𝒰, constants)
    T = temperature(𝒰, constants)
    q⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    τᵛˡ = em.vapor_to_liquid
    return ρ * (ℳ.qᵛ - q⁺ˡ) / τᵛˡ
end

# Relaxation toward ice saturation
@inline function AtmosphereModels.microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqⁱ}, ρ, ℳ, 𝒰, constants)
    T = temperature(𝒰, constants)
    q⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
    τᵛⁱ = em.vapor_to_ice
    return ρ * (ℳ.qᵛ - q⁺ⁱ) / τᵛⁱ
end

# Vapor closes by conservation
@inline function AtmosphereModels.microphysical_tendency(em::ExplicitMicrophysics, ::Val{:ρqᵛ}, ρ, ℳ, 𝒰, constants)
    Sˡ = AtmosphereModels.microphysical_tendency(em, Val(:ρqˡ), ρ, ℳ, 𝒰, constants)
    Sⁱ = AtmosphereModels.microphysical_tendency(em, Val(:ρqⁱ), ρ, ℳ, 𝒰, constants)
    return -Sˡ - Sⁱ
end
```

## Updating Auxiliary Fields

The `update_microphysical_auxiliaries!` function writes diagnostic fields:

```@example microphysics_example
@inline function AtmosphereModels.update_microphysical_auxiliaries!(μ, i, j, k, grid,
                                                                    ::ExplicitMicrophysics,
                                                                    ℳ::ExplicitMicrophysicsState,
                                                                    ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    return nothing
end
```

## Computing Moisture Fractions

The `moisture_fractions` function partitions the scheme-dependent specific moisture
``qᵛᵉ`` (see [Microphysics Interface Overview](@ref)) into vapor / liquid / ice components:

```@example microphysics_example
using Breeze.Thermodynamics: MoistureMassFractions

@inline function AtmosphereModels.moisture_fractions(::ExplicitMicrophysics, ℳ::ExplicitMicrophysicsState, qᵛᵉ)
    return MoistureMassFractions(ℳ.qᵛ, ℳ.qˡ, ℳ.qⁱ)
end
```

## Thermodynamic Adjustment

This is a fully prognostic scheme with no saturation adjustment, so we simply return
the state unchanged:

```@example microphysics_example
@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰, ::ExplicitMicrophysics, qᵛᵉ, constants) = 𝒰
```

## Summary

To implement a new microphysics scheme, we need to:

1. **Define a struct** to hold scheme parameters
2. **Implement `prognostic_field_names`** to list density-weighted prognostic variables
3. **Implement `materialize_microphysical_fields`** to create prognostic and diagnostic fields
4. **Define a state type** inheriting from `AbstractMicrophysicalState`
5. **Implement `microphysical_state`** to build state from prognostic scalars
6. **Implement `microphysical_tendency`** for each prognostic variable
7. **Implement `update_microphysical_auxiliaries!`** to update diagnostic fields
8. **Implement `moisture_fractions`** to partition moisture
9. **Implement `maybe_adjust_thermodynamic_state`** (trivial for non-equilibrium schemes)

For schemes with sedimentation, you would also implement velocity fields and the
`microphysical_velocities` function. For bundle schemes where many process rates feed
multiple prognostic tendencies, see
[Fused-kernel Microphysics Implementation](@ref).
