# B.6.10 minimal reproduction: checkpointing + fill_halo_regions! on asymmetric field
# No Breeze dependency — pure Oceananigans + Reactant + Enzyme
#
# Run: julia --check-bounds=no -O0 --project -e 'include("test/bbb-backward-pad-shape/mwe_checkpoint_halo.jl")'

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean

Reactant.set_default_backend("cpu")

FT = Float64
grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1),
                       topology=(Bounded, Bounded, Bounded))

# ── Test 1: Face,Center,Center + checkpointed loop ──
# This is the hypothesis: asymmetric field + checkpointing → stablehlo.pad failure

function loss_fcc_ckpt(u)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:2
        fill_halo_regions!(u)
        parent(u) .= parent(u) .* FT(0.99)
    end
    return mean(interior(u).^2)
end

function grad_fcc_ckpt(u, du)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_fcc_ckpt, Enzyme.Active,
        Enzyme.Duplicated(u, du))
    return lv
end

@info "Test 1: Face,Center,Center + checkpointed loop (expected FAIL)"
let u = Field{Face, Center, Center}(grid)
    set!(u, (x, y, z) -> sin(x))
    du = Enzyme.make_zero(u)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_fcc_ckpt(u, du)
    lv = compiled(u, du)
    @info "Test 1 PASSED — loss=$lv"
end

# ── Test 2: Face,Center,Center WITHOUT checkpointing (control) ──

function loss_fcc_nockpt(u)
    @trace track_numbers=false for _ in 1:2
        fill_halo_regions!(u)
        parent(u) .= parent(u) .* FT(0.99)
    end
    return mean(interior(u).^2)
end

function grad_fcc_nockpt(u, du)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_fcc_nockpt, Enzyme.Active,
        Enzyme.Duplicated(u, du))
    return lv
end

@info "Test 2: Face,Center,Center WITHOUT checkpointing (expected PASS)"
let u = Field{Face, Center, Center}(grid)
    set!(u, (x, y, z) -> sin(x))
    du = Enzyme.make_zero(u)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_fcc_nockpt(u, du)
    lv = compiled(u, du)
    @info "Test 2 PASSED — loss=$lv"
end

# ── Test 3: Center,Center,Center + checkpointed loop (symmetry control) ──

function loss_ccc_ckpt(c)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:2
        fill_halo_regions!(c)
        parent(c) .= parent(c) .* FT(0.99)
    end
    return mean(interior(c).^2)
end

function grad_ccc_ckpt(c, dc)
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_ccc_ckpt, Enzyme.Active,
        Enzyme.Duplicated(c, dc))
    return lv
end

@info "Test 3: Center,Center,Center + checkpointed loop (expected PASS)"
let c = CenterField(grid)
    set!(c, (x, y, z) -> sin(x))
    dc = Enzyme.make_zero(c)
    @time compiled = Reactant.@compile raise=true raise_first=true sync=true grad_ccc_ckpt(c, dc)
    lv = compiled(c, dc)
    @info "Test 3 PASSED — loss=$lv"
end

@info "All tests done."
