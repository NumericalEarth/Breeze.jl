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
##### Auxiliary dynamics variables (pressure from equation of state)
#####

"""
$(TYPEDSIGNATURES)

Compute pressure from the equation of state for `CompressibleModel`.

The pressure is computed from the ideal gas law:

```math
p = ρ R^m T
```

where ``ρ`` is the density, ``R^m`` is the mixture gas constant, and ``T`` is the temperature.
"""
function compute_auxiliary_dynamics_variables!(model::CompressibleModel)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics

    launch!(arch, grid, :xyz,
            _compute_pressure!,
            dynamics.pressure,
            dynamics.density,
            model.temperature,
            model.specific_moisture,
            grid,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    fill_halo_regions!(dynamics.pressure)

    return nothing
end

@kernel function _compute_pressure!(pressure, density, temperature,
                                    specific_moisture, grid, microphysics,
                                    microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = density[i, j, k]
        T = temperature[i, j, k]
        qᵗ = specific_moisture[i, j, k]
    end

    # Compute moisture fractions for the mixture gas constant
    q = compute_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵗ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)

    # Ideal gas law: p = ρ R^m T
    @inbounds pressure[i, j, k] = ρ * Rᵐ * T
end

