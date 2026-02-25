#####
##### MWE: Isolating checkpointing + halo fill interaction on BBB
#####
# The single-step halo fill adjoint works fine (mwe_bbb_pad_bug.jl).
# The full model with @trace checkpointing=true nsteps=4 fails (mwe_bbb_backward.jl).
# This MWE strips the model away and tests @trace + checkpointing + fill_halo_regions!
# on individual fields to pinpoint the trigger.
#
# Run: julia --project -e 'include("test/bbb-backward-pad-shape/mwe_checkpoint_halo.jl")'

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean

Reactant.set_default_backend("cpu")

N = 6
nsteps = 4  # perfect square for checkpointing

grid_bbb = RectilinearGrid(ReactantState();
    size = (N, N, N), extent = (1e3, 1e3, 1e3),
    topology = (Bounded, Bounded, Bounded))

grid_pbb = RectilinearGrid(ReactantState();
    size = (N, N, N), extent = (1e3, 1e3, 1e3),
    topology = (Periodic, Bounded, Bounded))

# ──────────────────────────────────────────────────────────────────
# Helper: loss with checkpointed @trace loop
# ──────────────────────────────────────────────────────────────────

function loss_ckpt(field, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        parent(field) .*= 0.99
        fill_halo_regions!(field)
    end
    return mean(interior(field).^2)
end

function grad_ckpt(field, dfield, nsteps)
    parent(dfield) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_ckpt, Enzyme.Active,
        Enzyme.Duplicated(field, dfield),
        Enzyme.Const(nsteps))
    return dfield, lv
end

# ──────────────────────────────────────────────────────────────────
# Helper: loss WITHOUT checkpointing (control)
# ──────────────────────────────────────────────────────────────────

function loss_no_ckpt(field, nsteps)
    @trace mincut=true checkpointing=false track_numbers=false for _ in 1:nsteps
        parent(field) .*= 0.99
        fill_halo_regions!(field)
    end
    return mean(interior(field).^2)
end

function grad_no_ckpt(field, dfield, nsteps)
    parent(dfield) .= 0
    _, lv = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_no_ckpt, Enzyme.Active,
        Enzyme.Duplicated(field, dfield),
        Enzyme.Const(nsteps))
    return dfield, lv
end

# ──────────────────────────────────────────────────────────────────
# Test matrix
# ──────────────────────────────────────────────────────────────────

function run_test(label, field, dfield, ns, use_ckpt)
    tag = use_ckpt ? "checkpointed" : "no-checkpoint"
    @info "  [$label] Compiling backward ($tag, nsteps=$ns)..."
    try
        if use_ckpt
            compiled = Reactant.@compile raise=true raise_first=true sync=true grad_ckpt(field, dfield, ns)
        else
            compiled = Reactant.@compile raise=true raise_first=true sync=true grad_no_ckpt(field, dfield, ns)
        end
        @info "  [$label] ✓ $tag compilation succeeded"
        return true
    catch e
        @warn "  [$label] ✗ $tag compilation FAILED" exception=(e, catch_backtrace())
        return false
    end
end

results = Dict{String, Bool}()

# ── Test 1: CCF on BBB, WITH checkpointing (expected to FAIL) ──
w_bbb = Field{Center, Center, Face}(grid_bbb); set!(w_bbb, 1.0)
dw_bbb = Enzyme.make_zero(w_bbb)
results["CCF-BBB-ckpt"] = run_test("CCF-BBB", w_bbb, dw_bbb, nsteps, true)

# ── Test 2: CCC on BBB, WITH checkpointing ──
c_bbb = CenterField(grid_bbb); set!(c_bbb, 1.0)
dc_bbb = Enzyme.make_zero(c_bbb)
results["CCC-BBB-ckpt"] = run_test("CCC-BBB", c_bbb, dc_bbb, nsteps, true)

# ── Test 3: CFC on BBB, WITH checkpointing ──
v_bbb = Field{Center, Face, Center}(grid_bbb); set!(v_bbb, 1.0)
dv_bbb = Enzyme.make_zero(v_bbb)
results["CFC-BBB-ckpt"] = run_test("CFC-BBB", v_bbb, dv_bbb, nsteps, true)

# ── Test 4: FCC on BBB, WITH checkpointing ──
u_bbb = Field{Face, Center, Center}(grid_bbb); set!(u_bbb, 1.0)
du_bbb = Enzyme.make_zero(u_bbb)
results["FCC-BBB-ckpt"] = run_test("FCC-BBB", u_bbb, du_bbb, nsteps, true)

# ── Test 5: CCF on BBB, WITHOUT checkpointing (control) ──
w_bbb2 = Field{Center, Center, Face}(grid_bbb); set!(w_bbb2, 1.0)
dw_bbb2 = Enzyme.make_zero(w_bbb2)
results["CCF-BBB-no-ckpt"] = run_test("CCF-BBB-ctrl", w_bbb2, dw_bbb2, nsteps, false)

# ── Test 6: CCF on PBB, WITH checkpointing (control) ──
w_pbb = Field{Center, Center, Face}(grid_pbb); set!(w_pbb, 1.0)
dw_pbb = Enzyme.make_zero(w_pbb)
results["CCF-PBB-ckpt"] = run_test("CCF-PBB-ctrl", w_pbb, dw_pbb, nsteps, true)

# ── Test 7: CCC on BBB, WITHOUT checkpointing (control) ──
c_bbb2 = CenterField(grid_bbb); set!(c_bbb2, 1.0)
dc_bbb2 = Enzyme.make_zero(c_bbb2)
results["CCC-BBB-no-ckpt"] = run_test("CCC-BBB-ctrl", c_bbb2, dc_bbb2, nsteps, false)

# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────

println("\n", "="^60)
println("RESULTS SUMMARY")
println("="^60)
for (k, v) in sort(collect(results))
    status = v ? "✓ PASS" : "✗ FAIL"
    println("  $status  $k")
end
println("="^60)
