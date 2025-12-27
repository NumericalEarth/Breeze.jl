#####
##### Anelastic buoyancy force
#####
##### Implements the buoyancy_forceᶜᶜᶜ interface function for anelastic dynamics.
#####

"""
$(TYPEDSIGNATURES)

Compute the buoyancy force density for anelastic dynamics at cell center `(i, j, k)`.

For the anelastic formulation, the buoyancy force is:

```math
ρ b = -g (ρ - ρᵣ)
```

where `ρ` is the in-situ density computed from the ideal gas law using the
current temperature and pressure, and `ρᵣ` is the reference density.

The density is computed as:
```math
ρ = \\frac{pᵣ}{Rᵐ T}
```
where `pᵣ` is the reference pressure, `Rᵐ` is the mixture gas constant, and `T` is temperature.
"""
@inline function buoyancy_forceᶜᶜᶜ(i, j, k, grid,
                                   dynamics::AnelasticDynamics,
                                   formulation,
                                   temperature,
                                   specific_moisture,
                                   microphysics,
                                   microphysical_fields,
                                   constants)

    @inbounds begin
        qᵗ = specific_moisture[i, j, k]
        pᵣ = dynamics.reference_state.pressure[i, j, k]
        ρᵣ = dynamics.reference_state.density[i, j, k]
        T = temperature[i, j, k]
    end

    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵗ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)
    ρ = pᵣ / (Rᵐ * T)
    g = constants.gravitational_acceleration
    ρ′ = ρ - ρᵣ

    return - g * ρ′
end

