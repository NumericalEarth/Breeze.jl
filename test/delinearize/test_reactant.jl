using Reactant
using Enzyme
using KernelAbstractions
using Statistics: mean

Reactant.set_default_backend("cpu")

mutable struct MinimalGrid{T}
    Nx::T
    Ny::T
    Nz::T
    Hx::T
end

@kernel function fill_periodic_halo!(c, grid)
    j, k = @index(Global, NTuple)
    H = grid.Hx
    N = grid.Nx
    @inbounds for i = 1:H
        c[i, j, k]     = c[N+i, j, k]
        c[N+H+i, j, k] = c[H+i, j, k]
    end
end

function fill_halos!(c, grid)
    Ny_total = grid.Ny + 2 * grid.Hx
    Nz_total = grid.Nz + 2 * grid.Hx
    kernel! = fill_periodic_halo!(KernelAbstractions.CPU(), (4, 4))
    kernel!(c, grid; ndrange=(Ny_total, Nz_total))
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    return nothing
end

function loss(c, grid)
    fill_halos!(c, grid)
    return mean(c.^2)
end

function grad(c, dc, grid)
    dc .= 0
    _, lv = Enzyme.autodiff(Enzyme.ReverseWithPrimal, loss, Enzyme.Active,
        Enzyme.Duplicated(c, dc), Enzyme.Const(grid))
    return lv
end

halo = 2
Nx, Ny, Nz = 4, 4, 4
grid = MinimalGrid(Nx, Ny, Nz, halo)
c = Reactant.to_rarray(zeros(Nx + 2*halo, Ny + 2*halo, Nz + 2*halo))
dc = Reactant.to_rarray(zeros(Nx + 2*halo, Ny + 2*halo, Nz + 2*halo))

@info "Compiling with halo=$halo..."
compiled = Reactant.@compile raise_first=true raise=true sync=true grad(c, dc, grid)
