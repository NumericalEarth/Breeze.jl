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

using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ

#####
##### Pressure gradient for compressible dynamics
#####

@inline x_pressure_gradient(i, j, k, grid, d::CompressibleDynamics) = ∂xᶠᶜᶜ(i, j, k, grid, d.pressure)
@inline y_pressure_gradient(i, j, k, grid, d::CompressibleDynamics) = ∂yᶜᶠᶜ(i, j, k, grid, d.pressure)
@inline z_pressure_gradient(i, j, k, grid, d::CompressibleDynamics) = ∂zᶜᶜᶠ(i, j, k, grid, d.pressure)

#####
##### Density tendency from continuity equation
#####

"""
$(TYPEDSIGNATURES)

Compute the density tendency for compressible dynamics using the continuity equation.

The density evolves according to:
```math
\\frac{\\partial \\rho}{\\partial t} = -\\nabla \\cdot (\\rho \\mathbf{u})
```

Since momentum `ρu` is already available, this is simply the negative divergence of momentum.
"""
function compute_dynamics_tendency!(model::CompressibleModel)
    grid = model.grid
    arch = grid.architecture
    Gρ = model.timestepper.Gⁿ.ρ
    momentum = model.momentum

    launch!(arch, grid, :xyz, _compute_density_tendency!, Gρ, grid, momentum)

    return nothing
end

@kernel function _compute_density_tendency!(Gρ, grid, momentum)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρ[i, j, k] = - divᶜᶜᶜ(i, j, k, grid, momentum.ρu, momentum.ρv, momentum.ρw)
end
