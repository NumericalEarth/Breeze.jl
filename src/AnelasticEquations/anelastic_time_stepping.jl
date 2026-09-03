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
    fill_halo_regions!(model.momentum, boundary_condition_args(model)...)

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
    kernel_Δt = kernel_time_step(model.architecture, model.grid, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _pressure_correct_momentum!,
            model.momentum,
            model.grid,
            kernel_Δt,
            dynamics.pressure_anomaly,  # kinematic pressure p'/ρᵣ
            dynamics.reference_state.density)

    return nothing
end

#####
##### Single-column mode: no pressure solve, no vertical-velocity stepping (w ≡ 0)
#####
#
# On a `SingleColumnGrid` the anelastic mass constraint ∂z(ρᵣ w) = 0 together with rigid top/bottom
# boundaries (w = 0 there) forces `w ≡ 0` throughout the column — there is no elliptic problem left
# to solve. We therefore skip the pressure correction entirely and hold the vertical-momentum
# tendency at zero, so `ρw` (initialized to 0) never moves. Vertical transport is carried by the
# turbulence closure and prescribed large-scale forcing (e.g. subsidence). This mirrors
# `Oceananigans.HydrostaticFreeSurfaceModel`'s single-column mode.

AtmosphereModels.compute_pressure_correction!(::AnelasticSingleColumnModel, Δt) = nothing
AtmosphereModels.make_pressure_correction!(::AnelasticSingleColumnModel, Δt) = nothing

# Hold Gρw ≡ 0 so `ρw` stays at its initial value of zero (`w ≡ 0`). `compute_z_momentum_tendency!`
# overwrites Gρw, so zeroing here — rather than launching it — is what omits vertical stepping.
function AtmosphereModels.compute_vertical_momentum_tendency!(::AnelasticSingleColumnModel, Gρw, w_args)
    fill!(parent(Gρw), 0)
    return nothing
end
