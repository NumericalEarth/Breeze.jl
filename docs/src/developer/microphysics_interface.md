# `AtmosphereModel` microphysics interface

This document describes the interface for embedding microphysical processes into [`AtmosphereModel`](@ref).
The interface consists of eight functions that must be implemented for any microphysics scheme to work with `AtmosphereModel`.

## Overview

The microphysics interface consists of seven functions, each of which must be implemented to complete
a microphysics implementation in `AtmosphereModel`:

* [`Breeze.AtmosphereModels.prognostic_field_names`](@ref)
    - Defines the names of the prognostic microphysical fields
* [`Breeze.AtmosphereModels.materialize_microphysical_fields`](@ref)
    - "Materializes" or generates, given the model `grid` and `boundary_conditions`,
      a `NamedTuple` of microphysical fields.
    - The `NamedTuple` of microphysical fields _must_ include prognostic fields,
      but can also include additional diagnostic fields.
    - Note, `boundary_conditions` can only be supplied to prognostic fields.
* [`Breeze.AtmosphereModels.update_microphysical_fields!`](@ref)
    - Update the diagnostic microphysics fields. This should not touch the prognostic fields.
* [`Breeze.AtmosphereModels.maybe_adjust_thermodynamic_state`](@ref)
    - Possibly adjust the thermodynamic state according to some constraint, such as
      saturation adjustment.
* [`Breeze.AtmosphereModels.compute_moisture_fractions`](@ref)
    - Given the model state, return a `MoistureMassFractions` object
* [`Breeze.AtmosphereModels.microphysical_velocities`](@ref)
    - Build the differential velocity field that microphysical tracers
      experience in addition to the bulk velocity (for example, the terminal velocity
      of falling hydrometeors)
* [`Breeze.AtmosphereModels.microphysical_tendency`](@ref)
    - Add additional tendency terms to the microphysical tracer equations representing
      for example, condensation, evaporation, or autoconversion of cloud liquid and
      ice content to snow or rain.

## Example implementation

To illustrate the development of a new microphysics scheme, we implement a 
simple microphysics scheme that represents droplet and ice particle nucleation
with constant-rate relaxation of specific humidity to saturation.

```@example microphysics_interface
using Breeze

struct ExplicitMicrophysics{FT}
    vapor_to_liquid :: FT
    vapor_to_ice :: FT
end
```

### Prognostic field names and materializing prognostic + diagnostic fields

This scheme is fully prognostic, which means we must carry around vapor, liquid
and ice density as prognostic variables,

```@example microphysics_interface
import Breeze.AtmosphereModels: prognostic_field_names

prognostic_field_names(::ExplicitMicrophysics) = (:ρqᵛ, :ρqˡ, :ρqⁱ)
```

!!! note
    The names of prognostic fields defined by `prognostic_field_names` 
    **are crucial to the user interface**, because users can interact them and
    `set!` their initial conditions. The names of variables should be carefully
    chosen to be concise, mathematical forms that are consistent with Breeze conventions.

When we materialize the microphysics fields, we must include all of the prognostic fields
in addition to diagnostic fields (this behavior may change in the future):

```@example microphysics_interface
import Breeze.AtmosphereModels: materialize_microphysical_fields

function materialize_microphysical_fields(::ExplicitMicrophysics, grid, boundary_conditions)
    ρqᵛ = CenterField(grid, boundary_conditions=boundary_Conditions.ρqᵛ)
    ρqˡ = CenterField(grid, boundary_conditions=boundary_Conditions.ρqˡ)
    ρqⁱ = CenterField(grid, boundary_conditions=boundary_Conditions.ρqⁱ)
    qᵛ = CenterField(grid)
    return (; ρqˡ, ρqⁱ, ρqᵛ, qᵛ)
end
```
The tendencies for 

```@example microphysics_interface
import Breeze.AtmosphereModels: microphysical_tendency

using Breeze.Thermodynamics:
    PlanarLiquidSurface,
    PlanarIceSurface

@inline function microphysical_tendency(i, j, k, grid, em::ExplicitMicrophysics, ::Val{:ρqˡ}, μ, 𝒰, constants)
    ρ = 1.2 # density
    T = temperature(𝒰, constants)
    q⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    τᵛˡ = em.vapor_to_liquid
    return @inbounds ρ * (μ.qᵛ[i, j, k] - q⁺ˡ) / τᵛˡ
end

@inline function microphysical_tendency(i, j, k, grid,
    em::ExplicitMicrophysics, ::Val{:ρqⁱ}, μ, 𝒰, constants)

    ρ = 1.2 # density
    q = MoistureMassFractions(qᵛ, qˡ, qⁱ)
    T = temperature(𝒰, constants)
    q⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
    τᵛⁱ = em.vapor_to_ice
    qᵛ = @inbounds μ.qᵛ[i, j, k]

    return ρ * (qᵛ - q⁺ⁱ) / τᵛⁱ
end

@inline function microphysical_tendency(i, j, k, grid,
    em::ExplicitMicrophysics, ::Val{:ρqᵛ}, μ, 𝒰, constants)

    Sᵛˡ = microphysical_tendency(i, j, k, grid, em, Val(:ρvˡ), μ, 𝒰, constants)
    Sᵛⁱ = microphysical_tendency(i, j, k, grid, em, Val(:ρvⁱ), μ, 𝒰, constants)
    return - Sᵛˡ - Sᵛⁱ
end

```


Note we have included the diagnostic field `qᵛ` (the vapor mass fraction, aka "specific humidity")
in addition to the three prognostic fields representing vapor, liquid and ice density.

### Prognostic field names and materializing prognostic + diagnostic fields

```@example microphysics_interface
import Breeze.AtmosphereModels:
    update_microphysical_fields!,
    compute_moisture_fraction

@inline update_microphysical_fields!(μ, em::ExplicitMicrophysics, i, j, k, grid, ρ, state, p′, constants, Δt) =
    @inbounds μ.qᵛ[i, j, k] = state.moisture_mass_fractions.vapor

@inline function compute_moisture_fractions(i, j, k, grid,
    ::ExplicitMicrophysics, ρ, qᵗ, microphysical_fields)

    @inbounds begin
        qᵛ = microphysical_fields.qᵛ[i, j, k]
        qˡ = microphysical_fields.ρqˡ[i, j, k] / ρ
        qⁱ = microphysical_fields.ρqⁱ[i, j, k] / ρ
    end

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end
```

This is a fully prognostic  scheme, so there is no adjustment,

```@example microphysics_interface
import Breeze.AtmosphereModels: maybe_adjust_thermodynamic_state

@inline maybe_adjust_thermodynamic_state(state, ::ExplicitMicrophysics, μ, qᵗ, constants) = state
```
