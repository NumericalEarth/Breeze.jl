#####
##### Time stepping for CompressibleDynamics
#####

#####
##### Model initialization
#####

# No default initialization for compressible models
initialize_model_thermodynamics!(model::CompressibleModel) = nothing

#####
##### Pressure correction (no-op for compressible dynamics)
#####

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - pressure is computed diagnostically from the equation of state.
"""
TimeSteppers.compute_pressure_correction!(model::CompressibleModel, Δt) = nothing

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - no pressure projection is needed.
"""
TimeSteppers.make_pressure_correction!(model::CompressibleModel, Δt) = nothing

#####
##### Pressure solver (no-op)
#####

"""
    solve_for_pressure!(model::CompressibleModel)

No-op for `CompressibleDynamics` - pressure is computed from the equation of state, not solved.
"""
solve_for_pressure!(model::CompressibleModel) = nothing

#####
##### Auxiliary dynamics variables (temperature and pressure for compressible)
#####

"""
$(TYPEDSIGNATURES)

Compute temperature and pressure jointly for `CompressibleModel`.

For compressible dynamics with potential temperature thermodynamics, temperature and
pressure are coupled via the ideal gas law and the potential temperature definition:

```math
θ = T (p₀/p)^κ \\quad \\text{and} \\quad p = ρ R^m T
```

Eliminating the circular dependency gives the direct formula:

```math
T = θ^γ \\left(\\frac{ρ R^m}{p₀}\\right)^{γ-1}
```

where ``γ = c_p / c_v`` is the heat capacity ratio. Once temperature is known,
pressure is computed from the ideal gas law ``p = ρ R^m T``.

This joint computation is necessary because, unlike anelastic dynamics where pressure
comes from a reference state, compressible dynamics requires solving for both
temperature and pressure simultaneously.
"""
function compute_auxiliary_dynamics_variables!(model::CompressibleModel)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics

    # Ensure halos are filled (may have been async from update_state!)
    # These fields are needed for pressure computation via equation of state
    fill_halo_regions!(dynamics.density)
    fill_halo_regions!(prognostic_fields(model.formulation))

    launch!(arch, grid, :xyz,
            _compute_temperature_and_pressure!,
            model.temperature,
            dynamics.pressure,
            dynamics.density,
            model.formulation,
            dynamics,
            model.specific_moisture,
            grid,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    fill_halo_regions!(model.temperature)
    fill_halo_regions!(dynamics.pressure)

    return nothing
end

@kernel function _compute_temperature_and_pressure!(temperature_field, pressure_field,
                                                    density, formulation, dynamics,
                                                    specific_moisture, grid, microphysics,
                                                    microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = density[i, j, k]
        qᵗ = specific_moisture[i, j, k]
    end

    # Compute moisture fractions
    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)
    cₚᵐ = mixture_heat_capacity(q, constants)
    cᵥᵐ = cₚᵐ - Rᵐ
    γ = cₚᵐ / cᵥᵐ

    # Compute temperature and pressure jointly
    T, p = temperature_and_pressure(i, j, k, formulation, dynamics, ρ, Rᵐ, γ, q, constants)

    @inbounds begin
        temperature_field[i, j, k] = T
        pressure_field[i, j, k] = p
    end
end

# Dispatch on formulation type for the coupled temperature-pressure computation

@inline function temperature_and_pressure(i, j, k,
                                          formulation::LiquidIcePotentialTemperatureFormulation,
                                          dynamics, ρ, Rᵐ, γ, q, constants)
    # Note: potential_temperature_density is ρθ (prognostic), potential_temperature is θ (diagnostic)
    ρθ = @inbounds formulation.potential_temperature_density[i, j, k]
    θ = ρθ / ρ
    p₀ = standard_pressure(dynamics)

    # Direct formula: T = θ^γ (ρ Rᵐ / p₀)^(γ-1)
    # For moist air with condensate, adjust for latent heat
    qˡ = q.liquid
    qⁱ = q.ice
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    cₚᵐ = mixture_heat_capacity(q, constants)

    # T = θ^γ (ρ Rᵐ / p₀)^(γ-1) + latent heat correction
    T_dry = θ^γ * (ρ * Rᵐ / p₀)^(γ - 1)
    T = T_dry + (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cₚᵐ

    # Ideal gas law: p = ρ Rᵐ T
    p = ρ * Rᵐ * T

    return T, p
end

