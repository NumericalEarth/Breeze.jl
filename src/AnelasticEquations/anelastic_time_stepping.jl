#####
##### Pressure correction time stepping for AnelasticDynamics
#####

#####
##### Model initialization
#####

"""
$(TYPEDSIGNATURES)

Initialize thermodynamic state for anelastic models.
Sets the initial potential temperature to the reference state value.
"""
function AtmosphereModels.initialize_model_thermodynamics!(model::AnelasticModel)
    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model, θ=θ₀)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute the pressure correction for anelastic dynamics by solving the pressure Poisson equation.
"""
function AtmosphereModels.compute_pressure_correction!(model::AnelasticModel, Δt)
    # Mask immersed velocities
    foreach(mask_immersed_field!, model.momentum)
    fill_halo_regions!(model.momentum, model.clock, fields(model))

    # Enforce mass conservation at open boundaries (solvability condition for pressure)
    enforce_open_boundary_mass_conservation!(model, model.boundary_mass_fluxes)

    dynamics = model.dynamics
    ρŨ = model.momentum
    solver = model.pressure_solver
    αᵣp′ = dynamics.pressure_anomaly  # kinematic pressure p'/ρᵣ
    solve_for_anelastic_pressure!(αᵣp′, solver, ρŨ, Δt)
    fill_halo_regions!(αᵣp′)

    return nothing
end

#####
##### Momentum pressure correction
#####

@kernel function _pressure_correct_momentum!(M, grid, Δt, αᵣ_pₙ, ρᵣ)
    i, j, k = @index(Global, NTuple)

    ρᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρᵣ)
    ρᶜ = @inbounds ρᵣ[i, j, k]

    @inbounds M.ρu[i, j, k] -= ρᶜ * Δt * ∂xᶠᶜᶜ(i, j, k, grid, αᵣ_pₙ)
    @inbounds M.ρv[i, j, k] -= ρᶜ * Δt * ∂yᶜᶠᶜ(i, j, k, grid, αᵣ_pₙ)
    @inbounds M.ρw[i, j, k] -= ρᶠ * Δt * ∂zᶜᶜᶠ(i, j, k, grid, αᵣ_pₙ)
end

"""
$(TYPEDSIGNATURES)

Update the predictor momentum ``(ρu, ρv, ρw)`` with the non-hydrostatic pressure via

```math
(\\rho\\boldsymbol{u})^{n+1} = (\\rho\\boldsymbol{u})^n - \\Delta t \\, \\rho_r \\boldsymbol{\\nabla} \\left( \\alpha_r p_{nh} \\right)
```
"""
function AtmosphereModels.make_pressure_correction!(model::AnelasticModel, Δt)
    dynamics = model.dynamics

    launch!(model.architecture, model.grid, :xyz,
            _pressure_correct_momentum!,
            model.momentum,
            model.grid,
            Δt,
            dynamics.pressure_anomaly,  # kinematic pressure p'/ρᵣ
            dynamics.reference_state.density)

    return nothing
end
