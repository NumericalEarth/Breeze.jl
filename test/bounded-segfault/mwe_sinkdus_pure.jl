#=
MWE: SinkDUS Segfault (B.6.4)
==============================
Systematically isolates what triggers the SinkDUS crash.

Key finding from previous run:
  - Pure Reactant arrays with DUS/Slice patterns → NO CRASH (Test A passed)
  - Pure Reactant arrays with broadcasts in while → HANGS in CSE rewriter
  - The Breeze crash requires enzymexla.kernel_call ops in the while body

Strategy: Use Oceananigans Fields (not Breeze) on a Bounded grid.
Field operations (broadcast, set!, interior) produce kernel_call ops via
KernelAbstractions, matching the Breeze MLIR structure WITHOUT the full
AtmosphereModel complexity.

Tests in order of increasing complexity:
  1. Forward: simple field multiply in loop
  2. Forward: two-field operations in loop
  3. Gradient: simple field multiply (Enzyme AD)
  4. Gradient: two-field operations (closer to Breeze)
  5. Gradient: with set! inside loop (DUS inside while body)
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Reactant
using Enzyme
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

# ── MLIR dump setup ──────────────────────────────────────────────────────────
mlir_dump_dir = joinpath(@__DIR__, "mlir_dump_mwe")
mkpath(mlir_dump_dir)
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mlir_dump_dir

# ── Grid setup (Bounded topology, matching Breeze crash case) ────────────────
grid = RectilinearGrid(ReactantState();
    size = (4, 4),
    extent = (1.0, 1.0),
    halo = (3, 3),
    topology = (Bounded, Bounded, Flat))

nsteps = 2

# ── Helper ───────────────────────────────────────────────────────────────────
function run_test(label, fn)
    print("  $label: ")
    flush(stdout)
    try
        result = fn()
        println("OK (result=$result)")
        flush(stdout)
        return true
    catch e
        if e isa InterruptException
            rethrow()
        end
        # Print just first line of error
        msg = sprint(showerror, e)
        first_line = first(split(msg, '\n'))
        println("ERROR: $first_line")
        flush(stdout)
        return false
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Forward: single field, simple multiply
# ═══════════════════════════════════════════════════════════════════════════════
# MLIR: DUS (from set!) → While with kernel_call (a .= a .* 0.99) → Slice (interior)

function loss_1field(a, nsteps)
    set!(a, 0.01)
    @trace track_numbers=false for i in 1:nsteps
        a .= a .* 0.99
    end
    return mean(interior(a) .^ 2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Forward: two fields, cross-update
# ═══════════════════════════════════════════════════════════════════════════════
# More kernel_calls per iteration, more while carry variables

function loss_2field(a, b, nsteps)
    set!(a, 0.01)
    set!(b, 0.005)
    @trace track_numbers=false for i in 1:nsteps
        a .= a .* 0.9 .+ b .* 0.1
        b .= b .* 0.8 .+ a .* 0.2
    end
    return mean(interior(a) .^ 2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Gradient: single field (Enzyme reverse-mode AD)
# ═══════════════════════════════════════════════════════════════════════════════

function grad_loss_1field(a, da, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_1field, Enzyme.Active,
        Enzyme.Duplicated(a, da),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Gradient: two fields
# ═══════════════════════════════════════════════════════════════════════════════

function grad_loss_2field(a, da, b, db, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_2field, Enzyme.Active,
        Enzyme.Duplicated(a, da),
        Enzyme.Duplicated(b, db),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Gradient: set! inside loop (DUS inside while body)
# ═══════════════════════════════════════════════════════════════════════════════
# set! inside the traced loop produces DUS yield operands with external updates,
# which is exactly what SinkDUS matches on.

function loss_set_inside(a, init_val, nsteps)
    @trace track_numbers=false for i in 1:nsteps
        set!(a, init_val)
        a .= a .* 0.99
    end
    return mean(interior(a) .^ 2)
end

function grad_loss_set_inside(a, da, init_val, dinit_val, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_set_inside, Enzyme.Active,
        Enzyme.Duplicated(a, da),
        Enzyme.Duplicated(init_val, dinit_val),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run tests
# ═══════════════════════════════════════════════════════════════════════════════

println("=" ^ 72)
println("SinkDUS MWE — Oceananigans Fields on Bounded Grid")
println("Grid: $(summary(grid))")
println("MLIR dumps → $mlir_dump_dir")
println("nsteps = $nsteps")
println("=" ^ 72)

# ── Test 1: Forward, 1 field ─────────────────────────────────────────────────
println("\nTest 1 — Forward: 1 field, simple multiply:")
let a = CenterField(grid)
    run_test("Compile+Run", () -> begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true loss_1field(a, nsteps)
        compiled(a, nsteps)
    end)
end

# ── Test 2: Forward, 2 fields ────────────────────────────────────────────────
println("\nTest 2 — Forward: 2 fields, cross-update:")
let a = CenterField(grid), b = CenterField(grid)
    run_test("Compile+Run", () -> begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true loss_2field(a, b, nsteps)
        compiled(a, b, nsteps)
    end)
end

# ── Test 3: Gradient, 1 field ────────────────────────────────────────────────
println("\nTest 3 — Gradient: 1 field (Enzyme AD):")
let a = CenterField(grid)
    da = Enzyme.make_zero(a)
    run_test("Compile+Run", () -> begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss_1field(a, da, nsteps)
        compiled(a, da, nsteps)
    end)
end

# ── Test 4: Gradient, 2 fields ───────────────────────────────────────────────
println("\nTest 4 — Gradient: 2 fields (Enzyme AD):")
let a = CenterField(grid), b = CenterField(grid)
    da = Enzyme.make_zero(a)
    db = Enzyme.make_zero(b)
    run_test("Compile+Run", () -> begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss_2field(a, da, b, db, nsteps)
        compiled(a, da, b, db, nsteps)
    end)
end

# ── Test 5: Gradient, set! inside loop ────────────────────────────────────────
println("\nTest 5 — Gradient: set! inside loop (DUS inside while body):")
let a = CenterField(grid)
    da = Enzyme.make_zero(a)
    init_val = CenterField(grid)
    set!(init_val, (x, y) -> 0.01 * x)
    dinit_val = Enzyme.make_zero(init_val)
    run_test("Compile+Run", () -> begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss_set_inside(
            a, da, init_val, dinit_val, nsteps)
        compiled(a, da, init_val, dinit_val, nsteps)
    end)
end

println("\n" * "=" ^ 72)
println("Tests that CRASH identify the minimal trigger for the SinkDUS segfault.")
println("Tests that PASS show which patterns are safe.")
println("Check mlir_dump_mwe/ for the MLIR produced by each test.")
println("=" ^ 72)
