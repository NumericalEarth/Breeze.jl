#####
##### Tendencies for CompressibleDynamics
#####
##### The compressible Euler equations in conservation form include:
#####
##### Continuity: ∂ρ/∂t + ∇·(ρu) = 0
##### Momentum:   ∂(ρu)/∂t + ... + ∇p = ...
#####
##### The pressure gradient is added to the momentum equations via the
##### x/y/z_pressure_gradient interface functions.
#####

using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, div_xyᶜᶜᶜ

#####
##### Pressure gradient for compressible dynamics
#####
##### When a reference state exists, the vertical pressure gradient subtracts the
##### reference pressure gradient ∂z(p_ref). Combined with the reference-subtracted
##### buoyancy -g(ρ - ρ_ref), this eliminates the O(Δz²) hydrostatic truncation error.
##### The reference pressure is in discrete hydrostatic balance with the reference
##### density, so ∂z(p_ref) + g*ℑz(ρ_ref) = 0 exactly at the discrete level.
#####
##### Horizontal pressure gradients are unaffected because the reference state
##### is a column (varies only in z).
#####

@inline AtmosphereModels.x_pressure_gradient(i, j, k, grid, d::CompressibleDynamics) = ∂xᶠᶜᶜ(i, j, k, grid, d.pressure)
@inline AtmosphereModels.y_pressure_gradient(i, j, k, grid, d::CompressibleDynamics) = ∂yᶜᶠᶜ(i, j, k, grid, d.pressure)

@inline function AtmosphereModels.z_pressure_gradient(i, j, k, grid, d::CompressibleDynamics)
    ∂z_p = ∂zᶜᶜᶠ(i, j, k, grid, d.pressure)
    ∂z_pᵣ = ∂z_reference_pressureᶜᶜᶠ(i, j, k, grid, d.reference_state)
    return ∂z_p - ∂z_pᵣ
end

@inline ∂z_reference_pressureᶜᶜᶠ(i, j, k, grid, ::Nothing) = 0
@inline ∂z_reference_pressureᶜᶜᶠ(i, j, k, grid, ref::ExnerReferenceState) = ∂zᶜᶜᶠ(i, j, k, grid, ref.pressure)

#####
##### Density tendency from continuity equation
#####

"""
$(TYPEDSIGNATURES)

Compute the density tendency for compressible dynamics using the continuity equation.

The density evolves according to:
```math
\\partial_t \\rho = -\\boldsymbol{\\nabla \\cdot} (\\rho \\boldsymbol{u})
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

## Full 3D divergence for explicit and split-explicit time stepping
@kernel function _compute_density_tendency!(Gρ, grid, momentum, td)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρ[i, j, k] = - divᶜᶜᶜ(i, j, k, grid, momentum.ρu, momentum.ρv, momentum.ρw)
end

## Horizontal-only divergence for VITS — vertical div_z(ρw) is part of fᴵ,
## evaluated via evaluate_implicit_tendency! and accumulated by the ARK predictor.
@kernel function _compute_density_tendency!(Gρ, grid, momentum, ::VerticallyImplicitTimeStepping)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρ[i, j, k] = - div_xyᶜᶜᶜ(i, j, k, grid, momentum.ρu, momentum.ρv)
end

