#####
##### Anelastic buoyancy force
#####
##### Implements the buoyancy_forceᶜᶜᶜ interface function for anelastic dynamics.
#####

"""
$(TYPEDSIGNATURES)

Compute the buoyancy force density for anelastic dynamics at cell center `(i, j, k)`.

The anelastic buoyancy force is the gravitational force on the density anomaly:

```math
-g ρ' = -g (ρ - ρ_r)
```

where ``ρ = p_r / (R^m T)`` is the in-situ density from the ideal gas law,
and ``ρ_r = p_r / (R^m_r T_r)`` is the reference density. Substituting:

```math
\\rho' = \\frac{p_r}{R^m T} - \\frac{p_r}{R^m_r T_r}
       = \\frac{p_r}{R^m_r T_r} \\left( \\frac{R^m_r T_r}{R^m T} - 1 \\right)
       = \\rho_r \\left( \\frac{R^m_r T_r}{R^m T} - 1 \\right)
```

This "perturbation form" avoids subtracting two large, nearly-equal numbers
(``p_r / (R^m T) - ρ_r``), which causes catastrophic cancellation when ``T ≈ T_r``.
Instead, the ratio ``R^m_r T_r / (R^m T)`` is close
to 1, and the subtraction of 1 preserves relative precision.

Here, ``R^m = q^d R^d + q^v R^v`` is the mixture gas constant for the current
moisture state and ``R^m_r`` is the mixture gas constant for the reference
moisture state.
"""
@inline function AtmosphereModels.buoyancy_forceᶜᶜᶜ(i, j, k, grid,
                                                    dynamics::AnelasticDynamics,
                                                    temperature,
                                                    specific_prognostic_moisture,
                                                    microphysics,
                                                    microphysical_fields,
                                                    constants)

    ρᵣ = @inbounds dynamics.reference_state.density[i, j, k]
    RᵐT, RᵐᵣTᵣ = gas_constant_temperature_products(i, j, k, grid, dynamics, temperature, specific_prognostic_moisture,
                                                   microphysics, microphysical_fields, constants)

    # Perturbation buoyancy: ρ' = ρ - ρᵣ = ρᵣ (Rᵐᵣ Tᵣ / (Rᵐ T) - 1)
    g = constants.gravitational_acceleration
    ρ′ = ρᵣ * (RᵐᵣTᵣ / RᵐT - 1)

    return - g * ρ′
end

# The products Rᵐ T and Rᵐᵣ Tᵣ of the current and reference states, which at the shared pressure pᵣ are
# proportional to their specific volumes
@inline function gas_constant_temperature_products(i, j, k, grid, dynamics, temperature, specific_prognostic_moisture,
                                                   microphysics, microphysical_fields, constants)
    ref = dynamics.reference_state

    @inbounds begin
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        ρᵣ = ref.density[i, j, k]
        Tᵣ = ref.temperature[i, j, k]
        T = temperature[i, j, k]
        qᵛᵣ = ref.vapor_mass_fraction[i, j, k]
        qˡᵣ = ref.liquid_mass_fraction[i, j, k]
        qⁱᵣ = ref.ice_mass_fraction[i, j, k]
    end

    qᵣ = MoistureMassFractions(qᵛᵣ, qˡᵣ, qⁱᵣ)
    Rᵐᵣ = mixture_gas_constant(qᵣ, constants)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵛᵉ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)

    return Rᵐ * T, Rᵐᵣ * Tᵣ
end

# A parcel at the reference pressure has Dh/Dt = α Dpᵣ/Dt = - (α / αᵣ) g w, so Ds/Dt = D(h + g z)/Dt = - b w with
# b = g (α / αᵣ - 1). This is exact, and is what makes ∂z s + b = cᵖ Π ∂z θ in dry air. The buoyancy force above
# has b = g (1 - αᵣ / α) instead, which differs at second order in T - Tᵣ.
@inline function AtmosphereModels.parcel_static_energy_buoyancyᶜᶜᶜ(i, j, k, grid, dynamics::AnelasticDynamics,
                                                                   temperature, specific_prognostic_moisture,
                                                                   microphysics, microphysical_fields, constants)

    ρᵣ = @inbounds dynamics.reference_state.density[i, j, k]
    RᵐT, RᵐᵣTᵣ = gas_constant_temperature_products(i, j, k, grid, dynamics, temperature, specific_prognostic_moisture,
                                                   microphysics, microphysical_fields, constants)

    return ρᵣ * constants.gravitational_acceleration * (RᵐT / RᵐᵣTᵣ - 1)
end
