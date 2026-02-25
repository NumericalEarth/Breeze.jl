#####
##### MWE: stablehlo.pad shape mismatch for BBB topology (backward pass)
#####
# Reproduces: 'stablehlo.pad' op inferred type(s) 'tensor<0x6x7xf64>' are
#              incompatible with return type(s) of operation 'tensor<1x6x7xf64>'
#
# Root cause: the adjoint of the _fill_west_and_east_halo! KA kernel for a
# Center-Center-Face field on a Bounded-Bounded-Bounded grid computes the
# leading dimension of the pad output as 0 instead of 1.
#
# The west halo fill writes c[0, j, k] for j ∈ 1:Ny, k ∈ 1:Nz_face, producing
# a (1, Ny, Nz_face) update slice. In the Enzyme adjoint, the stablehlo.pad
# reconstructing this slice miscalculates the first dimension.
#
# Hypothesis: Reactant/Enzyme off-by-one in first-dimension pad computation
# when the update slice has size 1 in the leading dimension.

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme

Reactant.set_default_backend("cpu")

N = 6
grid = RectilinearGrid(ReactantState();
    size = (N, N, N),
    extent = (1e3, 1e3, 1e3),
    topology = (Bounded, Bounded, Bounded))

# --- Test 1: CCC field (all NoFlux BCs) ---
# West/east halo fill produces (1, 6, 6) slice — may or may not trigger the bug.
c = CenterField(grid)
set!(c, 1.0)

function loss_ccc(c)
    fill_halo_regions!(c)
    return sum(interior(c).^2)
end

@info "Compiling forward (CCC)..."
compiled_loss_ccc = Reactant.@compile raise=true loss_ccc(c)

@info "Compiling backward (CCC)..."
dc = Enzyme.make_zero(c)
try
    compiled_grad_ccc = Reactant.@compile raise=true Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_ccc, Enzyme.Active,
        Enzyme.Duplicated(c, dc))
    @info "CCC backward compilation succeeded"
catch e
    @warn "CCC backward compilation failed" exception=(e, catch_backtrace())
end

# --- Test 2: CCF field (NoFlux on west/east and south/north, Nothing on bottom/top) ---
# West/east halo fill produces (1, 6, 7) slice — this is the suspected trigger.
w = Field{Center, Center, Face}(grid)
set!(w, 1.0)

function loss_ccf(w)
    fill_halo_regions!(w)
    return sum(interior(w).^2)
end

@info "Compiling forward (CCF)..."
compiled_loss_ccf = Reactant.@compile raise=true loss_ccf(w)

@info "Compiling backward (CCF)..."
dw = Enzyme.make_zero(w)
try
    compiled_grad_ccf = Reactant.@compile raise=true Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_ccf, Enzyme.Active,
        Enzyme.Duplicated(w, dw))
    @info "CCF backward compilation succeeded"
catch e
    @warn "CCF backward compilation failed" exception=(e, catch_backtrace())
end

# --- Test 3: CFC field (NoFlux on west/east and bottom/top, Nothing on south/north) ---
# West/east halo fill produces (1, 7, 6) slice — tests the asymmetry.
v = Field{Center, Face, Center}(grid)
set!(v, 1.0)

function loss_cfc(v)
    fill_halo_regions!(v)
    return sum(interior(v).^2)
end

@info "Compiling forward (CFC)..."
compiled_loss_cfc = Reactant.@compile raise=true loss_cfc(v)

@info "Compiling backward (CFC)..."
dv = Enzyme.make_zero(v)
try
    compiled_grad_cfc = Reactant.@compile raise=true Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_cfc, Enzyme.Active,
        Enzyme.Duplicated(v, dv))
    @info "CFC backward compilation succeeded"
catch e
    @warn "CFC backward compilation failed" exception=(e, catch_backtrace())
end

# --- Test 4: PBB control (x-Periodic avoids west/east non-periodic halo fill) ---
grid_pbb = RectilinearGrid(ReactantState();
    size = (N, N, N),
    extent = (1e3, 1e3, 1e3),
    topology = (Periodic, Bounded, Bounded))

w_pbb = Field{Center, Center, Face}(grid_pbb)
set!(w_pbb, 1.0)

function loss_pbb(w)
    fill_halo_regions!(w)
    return sum(interior(w).^2)
end

@info "Compiling forward (PBB-CCF)..."
compiled_loss_pbb = Reactant.@compile raise=true loss_pbb(w_pbb)

@info "Compiling backward (PBB-CCF, control)..."
dw_pbb = Enzyme.make_zero(w_pbb)
try
    compiled_grad_pbb = Reactant.@compile raise=true Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_pbb, Enzyme.Active,
        Enzyme.Duplicated(w_pbb, dw_pbb))
    @info "PBB-CCF backward compilation succeeded (control passes as expected)"
catch e
    @warn "PBB-CCF backward compilation failed (unexpected)" exception=(e, catch_backtrace())
end
