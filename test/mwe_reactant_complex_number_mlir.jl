#=
MWE: Complex numbers in KernelAbstractions kernels fail Reactant MLIR lowering (issue #223)

This MWE targets Breeze issue #223 ("Reactant tracing for AnelasticFormulation").

## Key nuance (important for debugging)
- Complex numbers **can** work in *pure Julia* traced code with Reactant
  (see the control case below).
- Complex numbers currently fail inside **KernelAbstractions kernels** compiled by Reactant.
  This is the *direct* blocker for Breeze because Oceananigans' FFT-based Poisson solver uses
  a kernel that does:

    ϕ[i, j, k] = real(ϕc[i, j, k])   # where ϕc is complex-valued FFT data

The failure looks like:
- `'llvm.extractvalue' op operand #0 must be LLVM aggregate type, but got 'complex<f32>'`
  (or `complex<f64>`)
followed by:
- `"failed to run pass manager on module"`

How to run:
  julia --project=test test/mwe_reactant_complex_number_mlir.jl

Expected behavior (December 2025):
- Control (pure Julia): ✅ succeeds for ComplexF32/ComplexF64
- Trigger (KA kernel): ❌ fails for ComplexF32/ComplexF64
=#

using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Solvers: copy_real_component!
using Oceananigans.Utils: launch!
using CUDA

Reactant.set_default_backend("cpu")

function short_error(e, bt; max_chars=800)
    s = sprint(showerror, e, bt)
    return first(s, min(lastindex(s), max_chars))
end

println("="^80)
println("MWE: Complex numbers in KernelAbstractions kernels (Reactant MLIR)")
println("="^80)
println("Reactant:     v", pkgversion(Reactant))
println("Oceananigans: v", pkgversion(Oceananigans))
println("CUDA:         v", pkgversion(CUDA))
println()

# ----------------------------------------------------------------------
# Control: Complex numbers in pure Julia traced code can work
# ----------------------------------------------------------------------
println("-"^80)
println("Control: pure Julia complex ops compile (sum(real.(A)))")
println("-"^80)

extract_and_sum_real(A) = sum(real.(A))

for CT in (ComplexF32, ComplexF64)
    A = Reactant.to_rarray(rand(CT, 4, 4))
    try
        compiled = Reactant.@compile sync=true raise=true extract_and_sum_real(A)
        y = compiled(A)
        println("OK ", CT, " -> ", y)
    catch e
        bt = catch_backtrace()
        println("UNEXPECTED FAIL for ", CT)
        println(short_error(e, bt))
    end
end

println()

# ----------------------------------------------------------------------
# Trigger: Oceananigans' `copy_real_component!` KernelAbstractions kernel
# ----------------------------------------------------------------------
println("-"^80)
println("Trigger: Oceananigans.Solvers.copy_real_component! (real(Complex) inside KA kernel)")
println("-"^80)

arch = ReactantState()
grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))

function run_copy!(arch, out, inp, grid)
    launch!(arch, grid, :xyz, copy_real_component!, out, inp, axes(out))
    return nothing
end

for CT in (ComplexF32, ComplexF64)
    RT = real(CT)
    out = on_architecture(arch, zeros(RT, 4, 4, 4))
    inp = on_architecture(arch, rand(CT, 4, 4, 4))

    try
        compiled = Reactant.@compile sync=true raise=true run_copy!(arch, out, inp, grid)
        compiled(arch, out, inp, grid)
        println("UNEXPECTED OK for ", CT, " (this would mean the upstream blocker is fixed)")
    catch e
        bt = catch_backtrace()
        println("FAIL (expected) for ", CT)
        println(short_error(e, bt; max_chars=500))
    end

    println()
end

println("="^80)
println("Summary:")
println("  - Pure Julia Complex is OK under Reactant")
println("  - Complex inside KA kernels fails (copy_real_component!) → blocks FFT-based Poisson solvers and Breeze issue #223")
println("="^80)
