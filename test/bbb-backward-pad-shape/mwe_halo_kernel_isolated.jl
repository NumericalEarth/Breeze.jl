# B.6.10 kernel isolation: backward pass through fill_halo_regions! on BBB Face field
# Strips away full model — just halo fill + Enzyme on a single (Face,Center,Center) field.
# Run: julia --check-bounds=no --project -e 'include("test/bbb-backward-pad-shape/mwe_halo_kernel_isolated.jl")'

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
                       topology=(Bounded, Bounded, Bounded))
FT = eltype(grid)

u = Field{Face, Center, Center}(grid)
set!(u, (x, y, z) -> FT(sin(x) * cos(y) * sin(z)))

du = Enzyme.make_zero(u)

function loss_halo(u)
    fill_halo_regions!(u)
    return mean(interior(u).^2)
end

function grad_halo(u, du)
    parent(du) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_halo, Enzyme.Active,
        Enzyme.Duplicated(u, du))
    return du, lv
end

@info "Compiling isolated halo-fill backward on Face field (BBB)..."
compiled = Reactant.@compile raise=true raise_first=true sync=true grad_halo(u, du)

du_out, lv = compiled(u, du)
@info "Loss: $lv  max|∇u|: $(maximum(abs, interior(du_out)))"
