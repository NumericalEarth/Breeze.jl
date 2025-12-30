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

"""
    x_pressure_gradient(i, j, k, grid, dynamics::CompressibleDynamics)

Return the x-component of the pressure gradient force at (Face, Center, Center).

For compressible dynamics, returns `-∂p/∂x`.
"""
@inline function x_pressure_gradient(i, j, k, grid, dynamics::CompressibleDynamics)
    return -∂xᶠᶜᶜ(i, j, k, grid, dynamics.pressure)
end

"""
    y_pressure_gradient(i, j, k, grid, dynamics::CompressibleDynamics)

Return the y-component of the pressure gradient force at (Center, Face, Center).

For compressible dynamics, returns `-∂p/∂y`.
"""
@inline function y_pressure_gradient(i, j, k, grid, dynamics::CompressibleDynamics)
    return -∂yᶜᶠᶜ(i, j, k, grid, dynamics.pressure)
end

"""
    z_pressure_gradient(i, j, k, grid, dynamics::CompressibleDynamics)

Return the z-component of the pressure gradient force at (Center, Center, Face).

For compressible dynamics, returns `-∂p/∂z`.
"""
@inline function z_pressure_gradient(i, j, k, grid, dynamics::CompressibleDynamics)
    return -∂zᶜᶜᶠ(i, j, k, grid, dynamics.pressure)
end

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
    @inbounds Gρ[i, j, k] = density_tendency(i, j, k, grid, momentum)
end

"""
    density_tendency(i, j, k, grid, momentum)

Compute the density tendency at cell center (i, j, k).

The tendency is the negative divergence of momentum:
```math
G_\\rho = -\\nabla \\cdot (\\rho \\mathbf{u}) = -\\left(\\frac{\\partial (\\rho u)}{\\partial x} + \\frac{\\partial (\\rho v)}{\\partial y} + \\frac{\\partial (\\rho w)}{\\partial z}\\right)
```
"""
@inline function density_tendency(i, j, k, grid, momentum)
    return -divᶜᶜᶜ(i, j, k, grid, momentum.ρu, momentum.ρv, momentum.ρw)
end

