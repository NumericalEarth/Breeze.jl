#####
##### Time stepping for CompressibleDynamics
#####

#####
##### Model initialization
#####

# No default initialization for compressible models
AtmosphereModels.initialize_model_thermodynamics!(::CompressibleModel) = nothing

#####
##### Pressure correction (no-op for compressible dynamics)
#####

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - pressure is computed diagnostically from the equation of state.
"""
AtmosphereModels.compute_pressure_correction!(::CompressibleModel, Δt) = nothing

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - no pressure projection is needed.
"""
AtmosphereModels.make_pressure_correction!(::CompressibleModel, Δt) = nothing

#####
##### Pressure solver (no-op)
#####

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - pressure is computed from the equation of state, not solved.
"""
solve_for_pressure!(::CompressibleModel) = nothing

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
function AtmosphereModels.compute_auxiliary_dynamics_variables!(model::CompressibleModel)
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
            specific_prognostic_moisture(model),
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
                                                    specific_prognostic_moisture, grid, microphysics,
                                                    microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = density[i, j, k]
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
    end

    # Compute moisture fractions
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    cᵛᵐ = cᵖᵐ - Rᵐ
    γ = cᵖᵐ / cᵛᵐ

    # Compute temperature and pressure jointly
    T, p = temperature_and_pressure(i, j, k, grid, formulation, dynamics, ρ, Rᵐ, γ, q, constants)

    @inbounds begin
        temperature_field[i, j, k] = T
        pressure_field[i, j, k] = p
    end
end

# Dispatch on formulation type for the coupled temperature-pressure computation

@inline function temperature_and_pressure(i, j, k, grid,
                                          formulation::LiquidIcePotentialTemperatureFormulation,
                                          dynamics, ρ, Rᵐ, γ, q, constants)
    # Note: potential_temperature_density is ρθ (prognostic), potential_temperature is θ (diagnostic)
    ρθ = @inbounds formulation.potential_temperature_density[i, j, k]
    θ = ρθ / ρ
    pˢᵗ = standard_pressure(dynamics)

    qˡ = q.liquid
    qⁱ = q.ice
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q, constants)
    κ   = Rᵐ / cᵖᵐ

    # Dry-adiabatic inversion of θ from the ideal gas law (exact when there is no
    # condensate): T_dry = θ^γ (ρ Rᵐ / pˢᵗ)^(γ-1), equivalently T_dry = Π θ with
    # the Exner function Π evaluated at the dry pressure ρ Rᵐ T_dry.
    T_dry = θ^γ * (ρ * Rᵐ / pˢᵗ)^(γ - 1)

    # Latent-heat coupling for moist air. The liquid-ice potential temperature
    # satisfies  θ = (T - L) (pˢᵗ/p)^κ  with L = (ℒˡ qˡ + ℒⁱ qⁱ)/cᵖᵐ and the
    # ideal gas law p = ρ Rᵐ T. Naively adding L to `T_dry` evaluates the Exner
    # function at the *dry* pressure ρ Rᵐ T_dry instead of the actual pressure
    # ρ Rᵐ T, which under-heats condensate-laden parcels by ≈ (γ-1) L and yields
    # a buoyancy inconsistent with the anelastic core (which inverts the same θ
    # definition exactly). Solving the coupled (T, p) system gives the fixed point
    #   T = D T^κ + L ,   D ≡ θ (ρ Rᵐ / pˢᵗ)^κ ,
    # which we iterate to convergence (contraction rate ≈ κ ≈ 0.28). For dry/rest
    # cells (L = 0) the result is `T_dry` bit-for-bit, preserving the exact
    # discrete rest atmosphere.
    L = (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ
    D = θ * (ρ * Rᵐ / pˢᵗ)^κ
    T = T_dry + L
    for _ in 1:8
        T = D * T^κ + L
    end
    T = ifelse(L == 0, T_dry, T)

    # Ideal gas law: p = ρ Rᵐ T
    p = ρ * Rᵐ * T

    return T, p
end
