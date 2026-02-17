#####
##### Compressible buoyancy force
#####
##### Implements the buoyancy_forceᶜᶜᶜ interface function for compressible dynamics.
#####

"""
$(TYPEDSIGNATURES)

Compute the buoyancy force density for compressible dynamics at cell center `(i, j, k)`.

When a reference state is provided, the buoyancy force is computed as a perturbation:

```math
ρ b = -g (ρ - ρ_r)
```

where ``ρ_r`` is the reference density in discrete hydrostatic balance. This eliminates
the ``O(Δz^2)`` truncation error from the near-cancellation of ``∂p/∂z`` and ``gρ``,
which is essential for stability with acoustic substepping at large time steps.

Without a reference state, the full gravitational force ``-gρ`` is used.
"""
@inline function AtmosphereModels.buoyancy_forceᶜᶜᶜ(i, j, k, grid,
                                                    dynamics::CompressibleDynamics,
                                                    temperature,
                                                    specific_moisture,
                                                    microphysics,
                                                    microphysical_fields,
                                                    constants)

    ρ_field = dynamics_density(dynamics)
    @inbounds ρ = ρ_field[i, j, k]
    g = constants.gravitational_acceleration
    ρᵣ = reference_densityᶜᶜᶜ(i, j, k, grid, dynamics.reference_state)

    return -g * (ρ - ρᵣ)
end

@inline reference_densityᶜᶜᶜ(i, j, k, grid, ::Nothing) = 0
@inline reference_densityᶜᶜᶜ(i, j, k, grid, ref::ExnerReferenceState) = @inbounds ref.density[i, j, k]
