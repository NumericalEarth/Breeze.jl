#####
##### Compressible buoyancy force
#####
##### Implements the buoyancy_forceᶜᶜᶜ interface function for compressible dynamics.
#####

"""
$(TYPEDSIGNATURES)

Compute the buoyancy force density for compressible dynamics at cell center `(i, j, k)`.

For the fully compressible formulation, the buoyancy/gravity force is simply:

```math
ρ b = -g ρ
```

where `ρ` is the prognostic density field.

Note: In the compressible formulation, the full gravitational force appears directly
in the momentum equation without subtraction of a reference state.
"""
@inline function buoyancy_forceᶜᶜᶜ(i, j, k, grid,
                                   dynamics::CompressibleDynamics,
                                   temperature,
                                   specific_moisture,
                                   microphysics,
                                   microphysical_fields,
                                   constants)

    ρ_field = dynamics_density(dynamics)
    @inbounds ρ = ρ_field[i, j, k]
    g = constants.gravitational_acceleration

    return -g * ρ
end

