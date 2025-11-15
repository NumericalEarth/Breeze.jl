# [Example Microphysics Implementation](@id microphysics_example-section)

This document provides a complete example of implementing a custom microphysics scheme in Breeze. The example implements a simple bulk microphysics scheme with explicit nucleation, two categories (cloud liquid and cloud ice), and a zero-moment-style precipitation removal model.

## Scheme Overview

Our example scheme, called `SimpleBulkMicrophysics`, implements:

1. **Explicit nucleation**: Constant-rate conversion of vapor to cloud liquid and cloud ice
2. **Two categories**: Cloud liquid (`qᶜˡ`) and cloud ice (`qᶜⁱ`)
3. **Precipitation removal**: Instant removal of condensate above a threshold (zero-moment style)

The scheme has no prognostic microphysical fields (all processes are diagnostic), making it simpler than one-moment schemes but more complex than basic saturation adjustment.

## Complete Implementation

```julia
using Breeze
using Breeze.AtmosphereModels
using Breeze.Thermodynamics: AbstractThermodynamicState, MoistureMassFractions, 
                             total_moisture_mass_fraction, with_moisture, temperature, density
using Breeze.Microphysics: center_field_tuple
using Oceananigans: CenterField
using DocStringExtensions: TYPEDSIGNATURES

#####
##### SimpleBulkMicrophysics type definition
#####

"""
    SimpleBulkMicrophysics{FT}

A simple bulk microphysics scheme with explicit nucleation and precipitation removal.

Fields:
- `nucleation_rate_vapor_to_liquid`: Rate constant for vapor → cloud liquid nucleation (s⁻¹)
- `nucleation_rate_vapor_to_ice`: Rate constant for vapor → cloud ice nucleation (s⁻¹)
- `precipitation_threshold_liquid`: Liquid mass fraction threshold for precipitation removal (kg kg⁻¹)
- `precipitation_threshold_ice`: Ice mass fraction threshold for precipitation removal (kg kg⁻¹)
"""
struct SimpleBulkMicrophysics{FT}
   τᵛˡ :: FT
   τᵛⁱ :: FT
   qˡ★ :: FT
   qⁱ★ :: FT
end
```

#####
##### Interface implementation
#####

```julia
prognostic_field_names(::SimpleBulkMicrophysics) = (:ρqᵛ, :ρqˡ, :ρqⁱ)

function materialize_microphysical_fields(μp::SimpleBulkMicrophysics, grid, boundary_conditions)
    names = prognostic_field_names(μp)
    return center_field_tuple(grid, names...)
end

@inline update_microphysical_fields!(μ, μp::SimpleBulkMicrophysics, i, j, k, grid, density, 𝒰, thermo) = nothing

@inline @inbounds function compute_moisture_fractions(i, j, k, grid, μp::SimpleBulkMicrophysics, ρ, qᵗ, μ)
    qᵛ = μ.ρqᵛ[i, j, k] / ρ
    qˡ = μ.ρqˡ[i, j, k] / ρ
    qⁱ = μ.ρqⁱ[i, j, k] / ρ
    MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

@inline microphysical_velocities(::SimpleBulkMicrophysics, name) = nothing
@inline microphysical_tendency(i, j, k, grid, ::SimpleBulkMicrophysics, args...) = zero(grid)
```

## Usage Example

```julia
using Breeze
using Oceananigans

# Create grid
grid = RectilinearGrid(size=(64, 64, 64), extent=(1000, 1000, 1000))
microphysics = SimpleBulkMicrophysics(1e-4, 1e-5, 1e-3, 1e-3)
model = AtmosphereModel(grid; microphysics)

# Use the model normally
set!(model, qᵗ=0.01)  # Set initial moisture
time_step!(model, 1)  # Step forward
```

## Implementation Notes

### Design Decisions

1. **No Prognostic Fields**: All microphysical processes are diagnostic, simplifying the implementation but limiting the scheme's ability to represent precipitation explicitly.

2. **Explicit Nucleation**: The nucleation rates are constant, which is unrealistic but simple. A more sophisticated implementation would depend on supersaturation and aerosol properties.

3. **Precipitation Removal**: Excess condensate is instantly removed (returned to vapor), similar to zero-moment schemes. This is a simplification that doesn't represent precipitation transport.

4. **Conservation**: The scheme ensures total moisture conservation by normalizing moisture fractions if needed.

### Extending the Scheme

To make this scheme more realistic, consider:

1. **Add Prognostic Fields**: Implement `ρqʳ` and `ρqˢ` as prognostic fields to represent precipitation explicitly.

2. **Temperature-Dependent Nucleation**: Make nucleation rates depend on temperature and supersaturation.

3. **Sedimentation**: Implement `microphysical_velocities` to add fall velocities for precipitation.

4. **Tendencies**: Implement `microphysical_tendency` to add source/sink terms for prognostic fields.

5. **Phase Partitioning**: Improve the partitioning between liquid and ice based on temperature.

## See Also

- [Microphysics Interface](@ref): Complete documentation of all interface functions
- [Saturation Adjustment](@ref): Reference implementation of a more sophisticated scheme
- [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl): Advanced microphysics schemes that can be integrated with Breeze

