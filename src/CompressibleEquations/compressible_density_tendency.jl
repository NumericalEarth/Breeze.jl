#####
##### Density tendency for CompressibleDynamics
#####
##### The continuity equation in conservation form:
##### ∂ρ/∂t + ∇·(ρu) = 0
#####
##### The tendency is:
##### Gρ = -∇·(ρu) = -(∂(ρu)/∂x + ∂(ρv)/∂y + ∂(ρw)/∂z)
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

