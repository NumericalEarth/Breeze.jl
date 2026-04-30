#####
##### Tendencies for CompressibleDynamics
#####

using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, div_xyᶜᶜᶜ

##### Pressure gradient
@inline AtmosphereModels.x_pressure_gradient(i, j, k, grid, d::CompressibleDynamics) = ∂xᶠᶜᶜ(i, j, k, grid, d.pressure)
@inline AtmosphereModels.y_pressure_gradient(i, j, k, grid, d::CompressibleDynamics) = ∂yᶜᶠᶜ(i, j, k, grid, d.pressure)

@inline function AtmosphereModels.z_pressure_gradient(i, j, k, grid, d::CompressibleDynamics)
    ∂z_p = ∂zᶜᶜᶠ(i, j, k, grid, d.pressure)
    ∂z_pᵣ = ∂z_reference_pressureᶜᶜᶠ(i, j, k, grid, d.reference_state)
    return ∂z_p - ∂z_pᵣ
end

# For SplitExplicitTimeDiscretization: zero ONLY the vertical PGF and buoyancy.
# These are computed from the linearized pp formula (§5-7) in _convert_slow_tendencies!,
# using Uᴸ values frozen across all RK stages (matching MPAS tend_w_euler at rk_step=1).
#
# The HORIZONTAL PGF is NOT zeroed — it flows through from the dynamics kernel
# using the current state's pressure, matching MPAS which recomputes pp (and thus
# the horizontal PGF tend_u_euler) at every RK stage.
@inline AtmosphereModels.explicit_z_pressure_gradient(i, j, k, grid,
        d::CompressibleDynamics{<:SplitExplicitTimeDiscretization}) = zero(grid)
@inline AtmosphereModels.explicit_buoyancy_forceᶜᶜᶠ(i, j, k, grid,
        d::CompressibleDynamics{<:SplitExplicitTimeDiscretization}, args...) = zero(grid)

@inline ∂z_reference_pressureᶜᶜᶠ(i, j, k, grid, ::Nothing) = false
@inline ∂z_reference_pressureᶜᶜᶠ(i, j, k, grid, ref::ExnerReferenceState) = ∂zᶜᶜᶠ(i, j, k, grid, ref.pressure)

##### Density tendency
"""
$(TYPEDSIGNATURES)

Compute the density tendency for compressible dynamics using the continuity equation.

The density evolves according to:
```math
\\partial_t \\rho = -\\boldsymbol{\\nabla \\cdot \\,} (\\rho \\boldsymbol{u})
```

Since momentum `ρu` is already available, this is simply the negative divergence of momentum.
"""
function AtmosphereModels.compute_dynamics_tendency!(model::CompressibleModel)
    grid = model.grid
    arch = grid.architecture
    Gρ = model.timestepper.Gⁿ.ρ
    momentum = model.momentum
    td = model.dynamics.time_discretization
    launch!(arch, grid, :xyz, _compute_density_tendency!, Gρ, grid, momentum, td)
    return nothing
end

@kernel function _compute_density_tendency!(Gρ, grid, momentum, td)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρ[i, j, k] = - divᶜᶜᶜ(i, j, k, grid, momentum.ρu, momentum.ρv, momentum.ρw)
end
