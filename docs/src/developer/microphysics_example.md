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
- `precipitation_removal_rate`: Fraction of excess condensate removed per time step
"""
struct SimpleBulkMicrophysics{FT}
    nucleation_rate_vapor_to_liquid :: FT
    nucleation_rate_vapor_to_ice :: FT
    precipitation_threshold_liquid :: FT
    precipitation_threshold_ice :: FT
    precipitation_removal_rate :: FT
end

function SimpleBulkMicrophysics(FT::DataType = Float64;
                               nucleation_rate_vapor_to_liquid = 1e-4,
                               nucleation_rate_vapor_to_ice = 1e-5,
                               precipitation_threshold_liquid = 1e-3,
                               precipitation_threshold_ice = 1e-3,
                               precipitation_removal_rate = 0.1)
    return SimpleBulkMicrophysics(FT(nucleation_rate_vapor_to_liquid),
                                  FT(nucleation_rate_vapor_to_ice),
                                  FT(precipitation_threshold_liquid),
                                  FT(precipitation_threshold_ice),
                                  FT(precipitation_removal_rate))
end

#####
##### Interface implementation
#####

"""
$(TYPEDSIGNATURES)

Return `tuple()` - SimpleBulkMicrophysics has no prognostic variables.
All microphysical processes are diagnostic.
"""
prognostic_field_names(::SimpleBulkMicrophysics) = tuple()

"""
$(TYPEDSIGNATURES)

Create microphysical fields for SimpleBulkMicrophysics.
Returns diagnostic fields for cloud liquid and cloud ice.
"""
function materialize_microphysical_fields(μp::SimpleBulkMicrophysics, grid, boundary_conditions)
    names = (:qᶜˡ, :qᶜⁱ)
    return center_field_tuple(grid, names...)
end

"""
$(TYPEDSIGNATURES)

Update microphysical fields from the thermodynamic state.
This function applies explicit nucleation and precipitation removal.
"""
@inline @inbounds function update_microphysical_fields!(μ, μp::SimpleBulkMicrophysics, 
                                                        i, j, k, grid, density, 𝒰, thermo)
    # Extract current moisture state
    qᵗ = total_moisture_mass_fraction(𝒰)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qˡ = 𝒰.moisture_mass_fractions.liquid
    qⁱ = 𝒰.moisture_mass_fractions.ice
    
    # Get current cloud fields
    qᶜˡ_old = μ.qᶜˡ[i, j, k]
    qᶜⁱ_old = μ.qᶜⁱ[i, j, k]
    
    # Explicit nucleation: convert vapor to cloud condensate
    # This is a simplified model - in reality, nucleation depends on supersaturation
    # and aerosol properties. Here we use constant rates.
    Δt_nuc = 1.0  # Time step for nucleation (would come from model in practice)
    Δqᵛ→ˡ = μp.nucleation_rate_vapor_to_liquid * qᵛ * Δt_nuc
    Δqᵛ→ⁱ = μp.nucleation_rate_vapor_to_ice * qᵛ * Δt_nuc
    
    # Update cloud fields (simplified - in practice, this would be part of tendency calculation)
    qᶜˡ_new = qᶜˡ_old + Δqᵛ→ˡ
    qᶜⁱ_new = qᶜⁱ_old + Δqᵛ→ⁱ
    
    # Precipitation removal: remove excess condensate above threshold
    if qᶜˡ_new > μp.precipitation_threshold_liquid
        excess = qᶜˡ_new - μp.precipitation_threshold_liquid
        qᶜˡ_new -= μp.precipitation_removal_rate * excess
    end
    
    if qᶜⁱ_new > μp.precipitation_threshold_ice
        excess = qᶜⁱ_new - μp.precipitation_threshold_ice
        qᶜⁱ_new -= μp.precipitation_removal_rate * excess
    end
    
    # Store updated fields
    μ.qᶜˡ[i, j, k] = qᶜˡ_new
    μ.qᶜⁱ[i, j, k] = qᶜⁱ_new
    
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute moisture fractions from microphysical fields.
For SimpleBulkMicrophysics, we combine cloud species with any remaining vapor.
"""
@inline @inbounds function compute_moisture_fractions(i, j, k, grid, μp::SimpleBulkMicrophysics, 
                                                      ρ, qᵗ, μ)
    qᶜˡ = μ.qᶜˡ[i, j, k]
    qᶜⁱ = μ.qᶜⁱ[i, j, k]
    
    # Total condensate
    qᶜ = qᶜˡ + qᶜⁱ
    
    # Vapor is remainder
    qᵛ = max(0, qᵗ - qᶜ)
    
    # Update liquid and ice to match cloud fields
    # (In a more sophisticated scheme, we might partition based on temperature)
    qˡ = qᶜˡ
    qⁱ = qᶜⁱ
    
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

"""
$(TYPEDSIGNATURES)

Compute thermodynamic state adjustment.
For SimpleBulkMicrophysics, we apply nucleation and precipitation removal.
"""
@inline function compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState, 
                                             μp::SimpleBulkMicrophysics, thermo)
    # Extract current state
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    q₀ = 𝒰₀.moisture_mass_fractions
    
    # Simplified nucleation model
    # In practice, this would be more sophisticated and depend on supersaturation
    qᵛ = q₀.vapor
    
    # Apply nucleation rates (simplified - assumes small changes)
    Δqᵛ→ˡ = μp.nucleation_rate_vapor_to_liquid * qᵛ
    Δqᵛ→ⁱ = μp.nucleation_rate_vapor_to_ice * qᵛ
    
    # Update moisture fractions
    qˡ_new = q₀.liquid + Δqᵛ→ˡ
    qⁱ_new = q₀.ice + Δqᵛ→ⁱ
    qᵛ_new = max(0, qᵗ - qˡ_new - qⁱ_new)
    
    # Apply precipitation removal
    if qˡ_new > μp.precipitation_threshold_liquid
        excess = qˡ_new - μp.precipitation_threshold_liquid
        qˡ_new -= μp.precipitation_removal_rate * excess
        qᵛ_new += μp.precipitation_removal_rate * excess  # Return to vapor
    end
    
    if qⁱ_new > μp.precipitation_threshold_ice
        excess = qⁱ_new - μp.precipitation_threshold_ice
        qⁱ_new -= μp.precipitation_removal_rate * excess
        qᵛ_new += μp.precipitation_removal_rate * excess  # Return to vapor
    end
    
    # Ensure conservation
    q_total = qᵛ_new + qˡ_new + qⁱ_new
    if q_total != qᵗ
        # Normalize to conserve total moisture
        scale = qᵗ / q_total
        qᵛ_new *= scale
        qˡ_new *= scale
        qⁱ_new *= scale
    end
    
    q₁ = MoistureMassFractions(qᵛ_new, qˡ_new, qⁱ_new)
    return with_moisture(𝒰₀, q₁)
end

"""
$(TYPEDSIGNATURES)

Compute temperature from thermodynamic state.
Delegates to compute_thermodynamic_state then extracts temperature.
"""
@inline function compute_temperature(𝒰₀::AbstractThermodynamicState, 
                                     μp::SimpleBulkMicrophysics, thermo)
    𝒰₁ = compute_thermodynamic_state(𝒰₀, μp, thermo)
    return temperature(𝒰₁, thermo)
end

"""
$(TYPEDSIGNATURES)

Return microphysical velocities.
SimpleBulkMicrophysics has no sedimentation (cloud particles are small).
"""
@inline microphysical_velocities(::SimpleBulkMicrophysics, name) = nothing

"""
$(TYPEDSIGNATURES)

Return microphysical tendency.
SimpleBulkMicrophysics has no prognostic fields, so tendencies are zero.
"""
@inline microphysical_tendency(i, j, k, grid, ::SimpleBulkMicrophysics, args...) = zero(grid)
```

## Usage Example

```julia
using Breeze
using Oceananigans

# Create grid
grid = RectilinearGrid(size=(64, 64, 64), extent=(1000, 1000, 1000))

# Create microphysics scheme
microphysics = SimpleBulkMicrophysics(
    nucleation_rate_vapor_to_liquid = 1e-4,
    nucleation_rate_vapor_to_ice = 1e-5,
    precipitation_threshold_liquid = 1e-3,
    precipitation_threshold_ice = 1e-3,
    precipitation_removal_rate = 0.1
)

# Create model
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

