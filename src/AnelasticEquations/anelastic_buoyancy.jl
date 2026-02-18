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
-g \\rho' = -g (\\rho - \\rho_r)
```

where ``\\rho = p_r / (R^m T)`` is the in-situ density from the ideal gas law,
and ``\\rho_r = p_r / (R^m_r T_r)`` is the reference density. Substituting:

```math
\\rho' = \\frac{p_r}{R^m T} - \\frac{p_r}{R^m_r T_r}
       = \\frac{p_r}{R^m_r T_r} \\left( \\frac{R^m_r T_r}{R^m T} - 1 \\right)
       = \\rho_r \\left( \\frac{R^m_r T_r}{R^m T} - 1 \\right)
```

This "perturbation form" avoids subtracting two large, nearly-equal numbers
(``p_r / (R^m T) - \\rho_r``), which causes catastrophic cancellation when ``T \\approx T_r``.
Instead, the ratio ``R^m_r T_r / (R^m T)`` is close
to 1, and the subtraction of 1 preserves relative precision.

Here ``R^m = q^d R^d + q^v R^v`` is the mixture gas constant for the current
moisture state and ``R^m_r`` is the mixture gas constant for the reference
moisture state.
"""
@inline function AtmosphereModels.buoyancy_forceᶜᶜᶜ(i, j, k, grid,
                                                    dynamics::AnelasticDynamics,
                                                    temperature,
                                                    specific_moisture,
                                                    microphysics,
                                                    microphysical_fields,
                                                    constants)

    ref = dynamics.reference_state

    @inbounds begin
        qᵛ = specific_moisture[i, j, k]
        ρᵣ = ref.density[i, j, k]
        Tᵣ = ref.temperature[i, j, k]
        T = temperature[i, j, k]
    end

    # Reference moisture fractions for Rᵐᵣ
    @inbounds begin
        qᵛᵣ = ref.vapor_mass_fraction[i, j, k]
        qˡᵣ = ref.liquid_mass_fraction[i, j, k]
        qⁱᵣ = ref.ice_mass_fraction[i, j, k]
    end

    qᵣ = MoistureMassFractions(qᵛᵣ, qˡᵣ, qⁱᵣ)
    Rᵐᵣ = mixture_gas_constant(qᵣ, constants)

    # Current moisture fractions for Rᵐ
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵣ, qᵛ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)

    # Perturbation buoyancy: ρ' = ρ - ρᵣ = ρᵣ (Rᵐᵣ Tᵣ / (Rᵐ T) - 1)
    g = constants.gravitational_acceleration
    ρ′ = ρᵣ * (Rᵐᵣ * Tᵣ / (Rᵐ * T) - 1)

    return - g * ρ′
end
