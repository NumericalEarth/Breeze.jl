#=
MWE: Slice Bounds Error Isolation — B.6.4 Bug 2
=================================================
Reactant v0.2.221+, Enzyme v0.13.118+, Julia 1.11+
Assumes the :xyz fix for compute_velocities! is in place (Bug 1 resolved).

PURPOSE: Determine whether the "limit index N+1 > dimension size N" slice
bounds error is truly a scale/complexity threshold issue OR whether it can
be triggered with a single fill_halo_regions! call on a Face field.

ROOT CAUSE HYPOTHESIS (from halo-fill-kernel-sizing.md):
  Oceananigans' fill_halo_size() returns a Symbol (:xz, :yz, :xy) for
  non-periodic BCs with Colon indices. This Symbol flows into work_layout(),
  which calls size(grid) → (Nx, Ny, Nz), ignoring field location. Face
  fields on Bounded have N+1 interior points, so the kernel worksize is
  wrong: StaticSize{(N,)} instead of StaticSize{(N+1,)}.

QUESTION: If the worksize is statically wrong, why does the MLIR error
only appear at 3x update_state!? This MWE tests escalating complexity
to find the exact trigger.

Levels (each tested with forward-only AND gradient):
  1. Single fill_halo_regions!(u) — Face field, no loop
  2. fill_halo_regions!(u) inside a @trace loop
  3. fill_halo_regions! on Face + Center fields together (mixed locations)
  4. Breeze update_state! × 1 per loop iteration
  5. Breeze update_state! × 2 per loop iteration
  6. Breeze update_state! × 3 per loop iteration (known fail)

Run:  julia --project=test test/bounded-segfault/mwe_slice_bounds_isolation.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, set!
using Oceananigans.TimeSteppers: update_state!
using Breeze
using Breeze: CompressibleDynamics
using Reactant
using Enzyme
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

const Nx = 5
const Ny = 5
const Hx = 3
const Hy = 3

grid = RectilinearGrid(ReactantState();
    size=(Nx, Ny), extent=(1e3, 1e3), halo=(Hx, Hy),
    topology=(Bounded, Bounded, Flat))

nsteps = 2

# ── Helpers ──────────────────────────────────────────────────────────────────

function run_test(label, fn)
    print("  $label: ")
    flush(stdout)
    try
        result = fn()
        println("PASS (result=$result)")
        flush(stdout)
        return :pass
    catch e
        e isa InterruptException && rethrow()
        msg = sprint(showerror, e)
        first_line = first(split(msg, '\n'))
        if contains(msg, "limit index") || contains(msg, "dimension size")
            println("SLICE BOUNDS ERROR: $first_line")
        elseif contains(msg, "SinkDUS") || contains(msg, "signal (11)")
            println("SEGFAULT/SinkDUS: $first_line")
        else
            println("ERROR: $first_line")
        end
        flush(stdout)
        return :fail
    end
end

function print_field_info(name, f)
    d = parent(f)
    println("  $name: loc=$(Oceananigans.Fields.location(f)), parent_size=$(size(d))")
end

# ═════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — Single fill_halo_regions! on a Face field (no loop, no AD)
# ═════════════════════════════════════════════════════════════════════════════
# If the worksize bug is real, this should fail at the MLIR level:
# the south/north kernel gets StaticSize{(Nx, 1)} = (5, 1) but the Face
# field in x has Nx+1 = 6 interior points.

u = Field{Face, Center, Center}(grid)
set!(u, (x, y) -> x + y)
du = Field{Face, Center, Center}(grid)
set!(du, 0)

c = CenterField(grid)
set!(c, (x, y) -> x * y)
dc = CenterField(grid)
set!(dc, 0)

println_header(s) = println("\n", "─" ^ 72, "\n", s, "\n", "─" ^ 72)

println("=" ^ 72)
println("Slice Bounds Error Isolation MWE")
println("Grid: Nx=$Nx, Ny=$Ny, Hx=$Hx, Hy=$Hy, topology=(Bounded, Bounded, Flat)")
println("=" ^ 72)

print_field_info("u (Face,Center,Center)", u)
print_field_info("c (Center,Center,Center)", c)
println()

# ── Level 1a: Forward only ─────────────────────────────────────────────────
println_header("LEVEL 1 — Single fill_halo_regions! (no loop)")

run_test("1a  Forward: fill_halo!(u)  [Face field]", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true fill_halo_regions!(u)
    compiled(u)
    "compiled"
end)

run_test("1b  Forward: fill_halo!(c)  [Center field, control]", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true fill_halo_regions!(c)
    compiled(c)
    "compiled"
end)

# ── Level 1b: With AD ─────────────────────────────────────────────────────
loss_single_halo(f) = (fill_halo_regions!(f); mean(interior(f) .^ 2))

function grad_single_halo(f, df)
    Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_single_halo, Enzyme.Active, Enzyme.Duplicated(f, df))
end

run_test("1c  Gradient: fill_halo!(u) [Face field]", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_single_halo(u, du)
    compiled(u, du)
end)

run_test("1d  Gradient: fill_halo!(c) [Center field, control]", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_single_halo(c, dc)
    compiled(c, dc)
end)

# ═════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — fill_halo_regions! inside a @trace loop
# ═════════════════════════════════════════════════════════════════════════════
println_header("LEVEL 2 — fill_halo_regions! inside @trace loop")

function loss_halo_loop(f, n)
    @trace track_numbers=false for _ in 1:n
        fill_halo_regions!(f)
        parent(f) .= parent(f) .* 0.99
    end
    return mean(interior(f) .^ 2)
end

function grad_halo_loop(f, df, n)
    Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_halo_loop, Enzyme.Active,
        Enzyme.Duplicated(f, df), Enzyme.Const(n))
end

run_test("2a  Forward: loop(fill_halo!(u))  [Face field]", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true loss_halo_loop(u, nsteps)
    compiled(u, nsteps)
end)

run_test("2b  Forward: loop(fill_halo!(c))  [Center field]", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true loss_halo_loop(c, nsteps)
    compiled(c, nsteps)
end)

run_test("2c  Gradient: loop(fill_halo!(u)) [Face field]", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_halo_loop(u, du, nsteps)
    compiled(u, du, nsteps)
end)

run_test("2d  Gradient: loop(fill_halo!(c)) [Center field]", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_halo_loop(c, dc, nsteps)
    compiled(c, dc, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# LEVEL 3 — Mixed Face + Center fields together (interaction hypothesis)
# ═════════════════════════════════════════════════════════════════════════════
# Key hypothesis: the error requires BOTH Face and Center fields on the
# same Bounded axes, because the kernel worksizes differ (N vs N+1).
println_header("LEVEL 3 — Mixed Face + Center fields in same loop")

function loss_mixed_halo(u, c, n)
    @trace track_numbers=false for _ in 1:n
        fill_halo_regions!(u)
        fill_halo_regions!(c)
        parent(u) .= parent(u) .* 0.99
        parent(c) .= parent(c) .* 0.98
    end
    return mean(interior(u) .^ 2) + mean(interior(c) .^ 2)
end

function grad_mixed_halo(u, du, c, dc, n)
    Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_mixed_halo, Enzyme.Active,
        Enzyme.Duplicated(u, du), Enzyme.Duplicated(c, dc),
        Enzyme.Const(n))
end

run_test("3a  Forward: loop(fill_halo!(u) + fill_halo!(c))", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true loss_mixed_halo(u, c, nsteps)
    compiled(u, c, nsteps)
end)

run_test("3b  Gradient: loop(fill_halo!(u) + fill_halo!(c))", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_mixed_halo(u, du, c, dc, nsteps)
    compiled(u, du, c, dc, nsteps)
end)

# Extra: try 3x repetitions inside the loop to test complexity scaling
function loss_mixed_halo_3x(u, c, n)
    @trace track_numbers=false for _ in 1:n
        fill_halo_regions!(u); fill_halo_regions!(c)
        parent(u) .= parent(u) .* 0.99; parent(c) .= parent(c) .* 0.98
        fill_halo_regions!(u); fill_halo_regions!(c)
        parent(u) .= parent(u) .* 0.99; parent(c) .= parent(c) .* 0.98
        fill_halo_regions!(u); fill_halo_regions!(c)
        parent(u) .= parent(u) .* 0.99; parent(c) .= parent(c) .* 0.98
    end
    return mean(interior(u) .^ 2) + mean(interior(c) .^ 2)
end

function grad_mixed_halo_3x(u, du, c, dc, n)
    Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_mixed_halo_3x, Enzyme.Active,
        Enzyme.Duplicated(u, du), Enzyme.Duplicated(c, dc),
        Enzyme.Const(n))
end

run_test("3c  Gradient: loop(3× fill_halo!(u) + fill_halo!(c))", () -> begin
    compiled = Reactant.@compile raise_first=true raise=true sync=true grad_mixed_halo_3x(u, du, c, dc, nsteps)
    compiled(u, du, c, dc, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# LEVEL 4 — Breeze update_state! (1x, 2x, 3x) — the known threshold
# ═════════════════════════════════════════════════════════════════════════════
println_header("LEVEL 4 — Breeze update_state! (1x, 2x, 3x per iteration)")

model  = AtmosphereModel(grid; dynamics=CompressibleDynamics())
dmodel = Enzyme.make_zero(model)
θ  = CenterField(grid); set!(θ, (x, y) -> 300 + 0.01x)
dθ = CenterField(grid); set!(dθ, 0)

function loss_Nx_update(model, θ, n, num_updates)
    set!(model, θ=θ, ρ=1.0)
    @trace track_numbers=false for _ in 1:n
        for _ in 1:num_updates
            update_state!(model)
            parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
        end
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_Nx_update(model, dmodel, θ, dθ, n, num_updates)
    Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_Nx_update, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(θ, dθ),
        Enzyme.Const(n), Enzyme.Const(num_updates))
end

for num_updates in [1, 2, 3]
    run_test("4-$(num_updates)a  Forward: $(num_updates)x update_state!", () -> begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true loss_Nx_update(
            model, θ, nsteps, num_updates)
        compiled(model, θ, nsteps, num_updates)
    end)

    run_test("4-$(num_updates)b  Gradient: $(num_updates)x update_state!", () -> begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_Nx_update(
            model, dmodel, θ, dθ, nsteps, num_updates)
        compiled(model, dmodel, θ, dθ, nsteps, num_updates)
    end)
end

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("INTERPRETATION GUIDE:")
println()
println("If Level 1 fails → bug triggers with a single fill_halo! call;")
println("  the 3x threshold is unrelated, likely a Reactant version change")
println("  exposed stricter MLIR verification.")
println()
println("If Level 1-2 pass but Level 3 fails → the bug requires INTERACTION")
println("  between Face and Center fields on different Bounded axes.")
println("  The kernel worksize inconsistency only produces invalid MLIR")
println("  when Enzyme's reverse pass reconciles gradients from kernels")
println("  with different StaticSizes on the same tensor dimensions.")
println()
println("If Level 1-3 pass but Level 4 fails at 3x → the bug genuinely")
println("  requires Breeze-level complexity (many fields, compute kernels,")
println("  tendency calculations) to produce enough MLIR ops for the")
println("  worksize inconsistency to cause a compilation error.")
println()
println("Forward passes should always pass (the mismatch is only exposed")
println("  during Enzyme AD reverse pass gradient accumulation).")
println("=" ^ 72)
