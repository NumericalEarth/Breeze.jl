#=
MWE: SinkDUS Segfault (B.6.4) — Pure Reactant + Enzyme
======================================================
No Oceananigans or Breeze. Targets the SinkDUS MLIR pattern directly.

Background:
  SinkDUS is an MLIR pattern that matches stablehlo.while ops. For each yield
  operand that is a dynamic_update_slice (DUS), it checks if:
    1. The DUS update value is defined OUTSIDE the while
    2. The DUS start indices are defined OUTSIDE the while
    3. The corresponding block argument is ONLY used by Slice, DUS (as source), and Return
  It then collects SliceOps on that block argument and checks overlap via
  mayReadMemoryWrittenTo(slice, DUS). The crash is a use-after-free in that check.

  This MWE creates the exact MLIR pattern SinkDUS matches on:
    - setindex! inside @trace for  →  DUS as yield operand, update defined outside
    - getindex  inside @trace for  →  Slice on the block argument
    - Block arg only used by those two ops + Return

Three test configurations:
  A. Inner pattern only:  DUS + Slice inside while body (directly triggers SinkDUS)
  B. Outer pattern only:  DUS before while + Slice after while (matches Breeze MLIR)
  C. Combined:            Both inner and outer patterns (maximum pattern interaction)

Each is tested with both forward compilation and Enzyme reverse-mode AD,
since the Breeze crash occurs during gradient compilation.
=#

using Reactant
using Enzyme

Reactant.set_default_backend("cpu")

# ── MLIR dump setup ──────────────────────────────────────────────────────────
mlir_dump_dir = joinpath(@__DIR__, "mlir_dump_mwe")
mkpath(mlir_dump_dir)
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mlir_dump_dir

# ── Array construction ───────────────────────────────────────────────────────
# Mimic Oceananigans fields with halo=3 around a 4×4 interior → 10×10×1 total
function make_padded(val=0.0)
    Reactant.ConcreteRArray(fill(Float64(val), 10, 10, 1))
end
function make_interior(val=0.0)
    Reactant.ConcreteRArray(fill(Float64(val), 4, 4, 1))
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST A — Inner pattern: DUS + Slice inside while body
# ═══════════════════════════════════════════════════════════════════════════════
#
# MLIR pattern produced:
#   while body {
#     %slice = stablehlo.slice %block_arg_a [3:7, 3:7, 0:1]       ← Slice of block arg
#     %add   = stablehlo.add %block_arg_accum, %slice              ← uses Slice result (not a's block arg)
#     %dus   = stablehlo.dynamic_update_slice %block_arg_a, %fill_val, [3,3,0]  ← DUS with external update
#     stablehlo.return ..., %dus, %add, ...
#   }
#
# Block arg `a` uses: {Slice, DUS-as-source} — satisfies SinkDUS constraints.
# DUS update = fill_val (defined outside while) — satisfies definedOutside check.
#

function loss_inner(a, accum, fill_val, nsteps)
    @trace track_numbers=false for i in 1:nsteps
        # Slice: read interior from block argument (4×4×1 from 10×10×1)
        interior = a[4:7, 4:7, 1:1]
        # Accumulate in accum (both 4×4×1 — uses Slice result, not a's block arg)
        accum .= accum .+ interior
        # DUS: overwrite interior with external value
        a[4:7, 4:7, 1:1] = fill_val
    end
    return sum(a .^ 2) + sum(accum .^ 2)
end

function grad_loss_inner(a, da, accum, daccum, fill_val, dfill_val, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_inner, Enzyme.Active,
        Enzyme.Duplicated(a, da),
        Enzyme.Duplicated(accum, daccum),
        Enzyme.Duplicated(fill_val, dfill_val),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST B — Outer pattern: DUS before while, Slice after while
# ═══════════════════════════════════════════════════════════════════════════════
#
# MLIR pattern produced (matches Breeze pre-pass MLIR):
#   %dus  = stablehlo.dynamic_update_slice %arg0, %init, [3,3,0]    ← DUS before while
#   %while = stablehlo.while(%dus, ...) { ... }                      ← DUS result → while init
#   %slice = stablehlo.slice %while#0 [3:7, 3:7, 0:1]               ← Slice of while result
#
# SinkDUS might not directly match this (it looks inside the body),
# but WhileDUS or WhileDUSDSSimplify might transform it first, then SinkDUS
# processes the result. This tests the pattern interaction path.
#

function loss_outer(a, b, init, nsteps)
    # DUS: write initial values into interior of halo-padded array
    a[4:7, 4:7, 1:1] = init
    b .= a .* 0.5

    @trace track_numbers=false for i in 1:nsteps
        a .= a .* 0.9 .+ b .* 0.1
        b .= b .* 0.8 .+ a .* 0.2
    end

    # Slice: extract interior from result
    return sum(a[4:7, 4:7, 1:1] .^ 2)
end

function grad_loss_outer(a, da, b, db, init, dinit, nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_outer, Enzyme.Active,
        Enzyme.Duplicated(a, da),
        Enzyme.Duplicated(b, db),
        Enzyme.Duplicated(init, dinit),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST C — Combined: both inner and outer patterns
# ═══════════════════════════════════════════════════════════════════════════════
#
# This combines Tests A and B to maximize pattern interaction:
#   - DUS before while (from init write)
#   - DUS inside while body as yield operand (from boundary fill)
#   - Slice inside while body on block argument (from interior read)
#   - Slice after while (from final interior extraction)
#   - Multiple arrays (a, b, c) → multiple SinkDUS candidates
#
# This is the closest to the Breeze crash scenario because it creates
# multiple interacting DUS/Slice patterns that different greedy rewriter
# patterns (WhileDUS, SinkDUS, WhileDUSDSSimplify) all try to match.
#

function loss_combined(a, b, c, accum, init, fill_a, fill_b, nsteps)
    # Outer DUS: write initial values
    a[4:7, 4:7, 1:1] = init
    b .= a .* 0.5
    c .= a .* 0.3

    @trace track_numbers=false for i in 1:nsteps
        # Inner Slices: read interiors from block arguments
        ia = a[4:7, 4:7, 1:1]
        ib = b[4:7, 4:7, 1:1]

        # Use sliced values (touches accum only, not a/b block args)
        accum .= accum .+ ia .+ ib

        # Inner DUS: write external boundary values back
        a[4:7, 4:7, 1:1] = fill_a
        b[4:7, 4:7, 1:1] = fill_b
    end

    # Outer Slice: extract interior from result
    return sum(a[4:7, 4:7, 1:1] .^ 2) + sum(accum .^ 2)
end

function grad_loss_combined(a, da, b, db, c, dc, accum, daccum,
                            init, dinit, fill_a, dfill_a, fill_b, dfill_b,
                            nsteps)
    _, val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_combined, Enzyme.Active,
        Enzyme.Duplicated(a, da),
        Enzyme.Duplicated(b, db),
        Enzyme.Duplicated(c, dc),
        Enzyme.Duplicated(accum, daccum),
        Enzyme.Duplicated(init, dinit),
        Enzyme.Duplicated(fill_a, dfill_a),
        Enzyme.Duplicated(fill_b, dfill_b),
        Enzyme.Const(nsteps))
    return val
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

nsteps = 2

function run_test(label, compile_fn, run_fn)
    print("  $label: ")
    try
        compiled = compile_fn()
        result = run_fn(compiled)
        println("OK (result=$result)")
        return true
    catch e
        if e isa InterruptException
            rethrow()
        end
        println("ERROR: $(sprint(showerror, e))")
        return false
    end
end

println("=" ^ 72)
println("SinkDUS MWE — Pure Reactant + Enzyme")
println("MLIR dumps → $mlir_dump_dir")
println("nsteps = $nsteps")
println("=" ^ 72)

# ── Test A: Inner pattern ────────────────────────────────────────────────────
println("\nTest A — Inner DUS+Slice (directly targets SinkDUS matching):")
let a = make_padded(), accum = make_interior(), fill_val = make_interior(0.5)
    da = make_padded()
    daccum = make_interior()
    dfill_val = make_interior()

    run_test("Forward", () -> begin
        Reactant.@compile raise_first=true raise=true sync=true loss_inner(
            a, accum, fill_val, nsteps)
    end, compiled -> compiled(a, accum, fill_val, nsteps))

    run_test("Gradient", () -> begin
        Reactant.@compile raise_first=true raise=true sync=true grad_loss_inner(
            a, da, accum, daccum, fill_val, dfill_val, nsteps)
    end, compiled -> compiled(a, da, accum, daccum, fill_val, dfill_val, nsteps))
end

# ── Test B: Outer pattern ────────────────────────────────────────────────────
println("\nTest B — Outer DUS→While→Slice (matches Breeze MLIR structure):")
let a = make_padded(), b = make_padded(), init = make_interior(300.0)
    da = make_padded()
    db = make_padded()
    dinit = make_interior()

    run_test("Forward", () -> begin
        Reactant.@compile raise_first=true raise=true sync=true loss_outer(
            a, b, init, nsteps)
    end, compiled -> compiled(a, b, init, nsteps))

    run_test("Gradient", () -> begin
        Reactant.@compile raise_first=true raise=true sync=true grad_loss_outer(
            a, da, b, db, init, dinit, nsteps)
    end, compiled -> compiled(a, da, b, db, init, dinit, nsteps))
end

# ── Test C: Combined pattern ─────────────────────────────────────────────────
println("\nTest C — Combined inner+outer (maximum pattern interaction):")
let a = make_padded(), b = make_padded(), c = make_padded()
    accum = make_interior()
    init = make_interior(300.0)
    fill_a = make_interior(0.5)
    fill_b = make_interior(0.3)
    da = make_padded()
    db = make_padded()
    dc = make_padded()
    daccum = make_interior()
    dinit = make_interior()
    dfill_a = make_interior()
    dfill_b = make_interior()

    run_test("Forward", () -> begin
        Reactant.@compile raise_first=true raise=true sync=true loss_combined(
            a, b, c, accum, init, fill_a, fill_b, nsteps)
    end, compiled -> compiled(a, b, c, accum, init, fill_a, fill_b, nsteps))

    run_test("Gradient", () -> begin
        Reactant.@compile raise_first=true raise=true sync=true grad_loss_combined(
            a, da, b, db, c, dc, accum, daccum,
            init, dinit, fill_a, dfill_a, fill_b, dfill_b, nsteps)
    end, compiled -> compiled(
            a, da, b, db, c, dc, accum, daccum,
            init, dinit, fill_a, dfill_a, fill_b, dfill_b, nsteps))
end

println("\n" * "=" ^ 72)
println("If all tests pass, the SinkDUS bug requires additional complexity")
println("(e.g. enzymexla.kernel_call lowering → DUS) not present in this MWE.")
println("Check mlir_dump_mwe/ for the MLIR produced by each test.")
println("=" ^ 72)
