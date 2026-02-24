# B.6.10 progressive isolation: find minimal trigger for BBB backward pad-shape error
# Run: julia --check-bounds=no -O0 --project -e 'include("test/bbb-backward-pad-shape/mwe_progressive.jl")'

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
                       topology=(Bounded, Bounded, Bounded))
FT = eltype(grid)

u  = Field{Face, Center, Center}(grid)
v  = Field{Center, Face, Center}(grid)
c  = CenterField(grid)

set!(u, (x, y, z) -> FT(sin(x)))
set!(v, (x, y, z) -> FT(cos(y)))
set!(c, (x, y, z) -> FT(1))

du = Enzyme.make_zero(u)
dv = Enzyme.make_zero(v)
dc = Enzyme.make_zero(c)

# ── L1: multiple fields, halo fills, no loop ──

function loss_L1(u, v, c)
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    fill_halo_regions!(c)
    return mean(interior(u).^2) + mean(interior(v).^2)
end

function grad_L1(u, du, v, dv, c, dc)
    parent(du) .= 0; parent(dv) .= 0; parent(dc) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L1, Enzyme.Active,
        Enzyme.Duplicated(u, du), Enzyme.Duplicated(v, dv), Enzyme.Duplicated(c, dc))
    return lv
end

@info "L1: multiple fields, halo fills, no loop"
@time compiled_L1 = Reactant.@compile raise=true raise_first=true sync=true grad_L1(u, du, v, dv, c, dc)
lv = compiled_L1(u, du, v, dv, c, dc)
@info "L1 passed — loss=$lv"

# ── L2: single Face field + halo fill inside @trace loop ──

function loss_L2(u)
    @trace track_numbers=false for _ in 1:4
        fill_halo_regions!(u)
    end
    return mean(interior(u).^2)
end

function grad_L2(u, du)
    parent(du) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L2, Enzyme.Active,
        Enzyme.Duplicated(u, du))
    return lv
end

@info "L2: single Face field + halo fill inside @trace loop"
@time compiled_L2 = Reactant.@compile raise=true raise_first=true sync=true grad_L2(u, du)
lv = compiled_L2(u, du)
@info "L2 passed — loss=$lv"

# ── L3: computation + halo fill inside @trace loop ──

function loss_L3(u, c)
    @trace track_numbers=false for _ in 1:4
        parent(u) .= parent(u) .+ FT(0.001) .* parent(c)
        fill_halo_regions!(u)
    end
    return mean(interior(u).^2)
end

function grad_L3(u, du, c, dc)
    parent(du) .= 0; parent(dc) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L3, Enzyme.Active,
        Enzyme.Duplicated(u, du), Enzyme.Duplicated(c, dc))
    return lv
end

@info "L3: computation + halo fill in @trace loop"
@time compiled_L3 = Reactant.@compile raise=true raise_first=true sync=true grad_L3(u, du, c, dc)
lv = compiled_L3(u, du, c, dc)
@info "L3 passed — loss=$lv"

# ── L4: multiple fields + computation + halo fills inside @trace loop ──

function loss_L4(u, v, c)
    @trace track_numbers=false for _ in 1:4
        parent(u) .= parent(u) .+ FT(0.001) .* parent(c)
        parent(v) .= parent(v) .+ FT(0.001) .* parent(c)
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        fill_halo_regions!(c)
    end
    return mean(interior(u).^2) + mean(interior(v).^2)
end

function grad_L4(u, du, v, dv, c, dc)
    parent(du) .= 0; parent(dv) .= 0; parent(dc) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L4, Enzyme.Active,
        Enzyme.Duplicated(u, du), Enzyme.Duplicated(v, dv), Enzyme.Duplicated(c, dc))
    return lv
end

@info "L4: multiple fields + computation + halo fills in @trace loop"
@time compiled_L4 = Reactant.@compile raise=true raise_first=true sync=true grad_L4(u, du, v, dv, c, dc)
lv = compiled_L4(u, du, v, dv, c, dc)
@info "L4 passed — loss=$lv"

# ── L5: same as L4 + checkpointing ──

function loss_L5(u, v, c)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:4
        parent(u) .= parent(u) .+ FT(0.001) .* parent(c)
        parent(v) .= parent(v) .+ FT(0.001) .* parent(c)
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        fill_halo_regions!(c)
    end
    return mean(interior(u).^2) + mean(interior(v).^2)
end

function grad_L5(u, du, v, dv, c, dc)
    parent(du) .= 0; parent(dv) .= 0; parent(dc) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_L5, Enzyme.Active,
        Enzyme.Duplicated(u, du), Enzyme.Duplicated(v, dv), Enzyme.Duplicated(c, dc))
    return lv
end

@info "L5: same as L4 + checkpointing"
@time compiled_L5 = Reactant.@compile raise=true raise_first=true sync=true grad_L5(u, du, v, dv, c, dc)
lv = compiled_L5(u, du, v, dv, c, dc)
@info "L5 passed — loss=$lv"

@info "All levels passed."
