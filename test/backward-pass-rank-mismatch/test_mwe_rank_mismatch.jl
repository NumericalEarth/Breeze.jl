#=
Investigation: Backward Pass Rank Mismatch — Pure MWE
Status: IN PROGRESS (as of 2026-02-23)
Purpose: Reproduce the DUS rank mismatch without Oceananigans/Breeze
Related: cursor-toolchain/rules/domains/differentiability/investigations/backward-pass-rank-mismatch.md

This MWE attempts to reproduce the core pattern that triggers the bug:
  1. Two arrays with different sizes (symmetric 10x10x10 vs asymmetric 11x10x10)
  2. A KA kernel with StaticSize((10,10,10)) writing to both arrays
  3. The kernel writes all 10x10x10 of the symmetric array but only 10x10x10
     of the 11x10x10 asymmetric array (leaving one slice unwritten)
  4. A @trace loop with checkpointing → Enzyme differentiates
  5. Post-Enzyme while_dus pass may corrupt the checkpoint cache rank

This simulates the Face-on-Bounded field pattern where:
  - ρ (Center,Center,Center) has interior (10,10,10) and parent (16,16,16)
  - ρu (Face,Center,Center) has interior (11,10,10) and parent (17,16,16)
  - The kernel ndrange is StaticSize((10,10,10)) for both

If this MWE does NOT reproduce the bug, the trigger may require:
  - OffsetStaticSize (Oceananigans' extended ndrange type)
  - Multiple arrays with different asymmetries (ρu and ρw)
  - Halo fill kernels with different ndranges interleaved
  - Multiple DUS operations per loop iteration (SSP RK3 substeps)
=#

using KernelAbstractions
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Test
using Pkg

println("Package versions:")
for pkg in ["Reactant", "Enzyme"]
    v = Pkg.dependencies()[Base.UUID(Pkg.project().dependencies[pkg])].version
    println("  $pkg: v$v")
end

mlir_dump_dir = joinpath(@__DIR__, "mlir_dump")
mkpath(mlir_dump_dir)
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mlir_dump_dir
@info "MLIR dumps will be written to" mlir_dump_dir

Reactant.set_default_backend("cpu")

# ---------------------------------------------------------------------------
# Kernel: writes to two arrays with different sizes using the same ndrange.
# The asymmetric array has size N+1 in the first dimension, but the kernel
# only accesses indices 1:N — this mimics Face-on-Bounded in Oceananigans.
# ---------------------------------------------------------------------------
@kernel function asymmetric_update!(symmetric, asymmetric, α)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        symmetric[i, j, k] = symmetric[i, j, k] + α * symmetric[i, j, k]
        asymmetric[i, j, k] = asymmetric[i, j, k] + α * asymmetric[i, j, k]
    end
end

# ---------------------------------------------------------------------------
# Variant 2: Uses parent-sized arrays with halo offsets.
# The kernel writes at offset H (simulating halo) into a larger parent array.
# This is closer to how Oceananigans actually structures the writes.
# ---------------------------------------------------------------------------
@kernel function halo_offset_update!(sym_parent, asym_parent, α, H)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        sym_parent[i + H, j + H, k + H] = sym_parent[i + H, j + H, k + H] * (1 + α)
        asym_parent[i + H, j + H, k + H] = asym_parent[i + H, j + H, k + H] * (1 + α)
    end
end

@testset "DUS rank mismatch MWE" begin

    # -------------------------------------------------------------------
    # Test 1: Basic asymmetric array sizes, no halo
    # symmetric:  10×10×10 (all written by kernel)
    # asymmetric: 11×10×10 (only 10×10×10 written by kernel)
    # -------------------------------------------------------------------
    @testset "Test 1: Basic asymmetric arrays (no halo)" begin
        N = 10
        α = 0.01

        symmetric  = Reactant.to_rarray(ones(Float64, N, N, N))
        asymmetric = Reactant.to_rarray(ones(Float64, N+1, N, N))
        dsymmetric  = Reactant.to_rarray(zeros(Float64, N, N, N))
        dasymmetric = Reactant.to_rarray(zeros(Float64, N+1, N, N))

        backend = KernelAbstractions.get_backend(symmetric)

        function loss_v1(symmetric, asymmetric, α, nsteps)
            kern! = asymmetric_update!(backend, KernelAbstractions.StaticSize((N, N, N)))
            @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
                kern!(symmetric, asymmetric, α)
            end
            return mean(symmetric .^ 2) + mean(asymmetric .^ 2)
        end

        function grad_loss_v1(symmetric, dsymmetric, asymmetric, dasymmetric, α, nsteps)
            parent(dsymmetric) .= 0
            parent(dasymmetric) .= 0
            _, loss_value = Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                loss_v1, Enzyme.Active,
                Enzyme.Duplicated(symmetric, dsymmetric),
                Enzyme.Duplicated(asymmetric, dasymmetric),
                Enzyme.Const(α),
                Enzyme.Const(nsteps))
            return dsymmetric, dasymmetric, loss_value
        end

        nsteps = 4

        @testset "Forward pass" begin
            @time "Compiling forward" compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss_v1(
                symmetric, asymmetric, α, nsteps)
            result = compiled_fwd(symmetric, asymmetric, α, nsteps)
            @test result > 0
            @test isfinite(result)
        end

        @testset "Backward pass (Enzyme)" begin
            @test_broken begin
                @time "Compiling backward" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss_v1(
                    symmetric, dsymmetric, asymmetric, dasymmetric, α, nsteps)
                compiled_grad !== nothing
            end
        end
    end

    # -------------------------------------------------------------------
    # Test 2: Parent-sized arrays with halo offset
    # Simulates Oceananigans parent arrays where kernel writes at offset H
    # sym_parent:  16×16×16 (Center: N + 2H = 10 + 6 = 16)
    # asym_parent: 17×16×16 (Face+Bounded: N+1 + 2H = 11 + 6 = 17)
    # Kernel ndrange: (10, 10, 10), writes at offset H=3
    # -------------------------------------------------------------------
    @testset "Test 2: Halo-offset arrays (simulating Oceananigans parent)" begin
        N = 10
        H = 3
        α = 0.01

        sym_parent  = Reactant.to_rarray(ones(Float64, N + 2H, N + 2H, N + 2H))
        asym_parent = Reactant.to_rarray(ones(Float64, N + 1 + 2H, N + 2H, N + 2H))
        dsym_parent  = Reactant.to_rarray(zeros(Float64, N + 2H, N + 2H, N + 2H))
        dasym_parent = Reactant.to_rarray(zeros(Float64, N + 1 + 2H, N + 2H, N + 2H))

        backend = KernelAbstractions.get_backend(sym_parent)

        function loss_v2(sym_parent, asym_parent, α, nsteps)
            kern! = halo_offset_update!(backend, KernelAbstractions.StaticSize((N, N, N)))
            @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
                kern!(sym_parent, asym_parent, α, H)
            end
            return mean(sym_parent .^ 2) + mean(asym_parent .^ 2)
        end

        function grad_loss_v2(sym_parent, dsym_parent, asym_parent, dasym_parent, α, nsteps)
            parent(dsym_parent) .= 0
            parent(dasym_parent) .= 0
            _, loss_value = Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                loss_v2, Enzyme.Active,
                Enzyme.Duplicated(sym_parent, dsym_parent),
                Enzyme.Duplicated(asym_parent, dasym_parent),
                Enzyme.Const(α),
                Enzyme.Const(nsteps))
            return dsym_parent, dasym_parent, loss_value
        end

        nsteps = 4

        @testset "Forward pass" begin
            @time "Compiling forward" compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss_v2(
                sym_parent, asym_parent, α, nsteps)
            result = compiled_fwd(sym_parent, asym_parent, α, nsteps)
            @test result > 0
            @test isfinite(result)
        end

        @testset "Backward pass (Enzyme)" begin
            @test_broken begin
                @time "Compiling backward" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss_v2(
                    sym_parent, dsym_parent, asym_parent, dasym_parent, α, nsteps)
                compiled_grad !== nothing
            end
        end
    end

    # -------------------------------------------------------------------
    # Test 3: Symmetric control — all arrays same size
    # Should NOT trigger the bug (no asymmetry)
    # -------------------------------------------------------------------
    @testset "Test 3: Symmetric control (all same size)" begin
        N = 10
        α = 0.01

        arr_a = Reactant.to_rarray(ones(Float64, N, N, N))
        arr_b = Reactant.to_rarray(ones(Float64, N, N, N))
        darr_a = Reactant.to_rarray(zeros(Float64, N, N, N))
        darr_b = Reactant.to_rarray(zeros(Float64, N, N, N))

        backend = KernelAbstractions.get_backend(arr_a)

        function loss_v3(arr_a, arr_b, α, nsteps)
            kern! = asymmetric_update!(backend, KernelAbstractions.StaticSize((N, N, N)))
            @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
                kern!(arr_a, arr_b, α)
            end
            return mean(arr_a .^ 2) + mean(arr_b .^ 2)
        end

        function grad_loss_v3(arr_a, darr_a, arr_b, darr_b, α, nsteps)
            parent(darr_a) .= 0
            parent(darr_b) .= 0
            _, loss_value = Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                loss_v3, Enzyme.Active,
                Enzyme.Duplicated(arr_a, darr_a),
                Enzyme.Duplicated(arr_b, darr_b),
                Enzyme.Const(α),
                Enzyme.Const(nsteps))
            return darr_a, darr_b, loss_value
        end

        nsteps = 4

        @testset "Forward pass" begin
            @time "Compiling forward" compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss_v3(
                arr_a, arr_b, α, nsteps)
            result = compiled_fwd(arr_a, arr_b, α, nsteps)
            @test result > 0
            @test isfinite(result)
        end

        @testset "Backward pass (Enzyme)" begin
            @time "Compiling backward" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss_v3(
                arr_a, darr_a, arr_b, darr_b, α, nsteps)
            @test compiled_grad !== nothing

            dsym, dasym, loss_val = compiled_grad(arr_a, darr_a, arr_b, darr_b, α, nsteps)
            @test loss_val > 0
            @test isfinite(loss_val)
        end
    end

    # -------------------------------------------------------------------
    # Test 4: Multiple asymmetric arrays (closer to Breeze's 5-field model)
    # ρ:  (10, 10, 10) — Center, Center, Center
    # ρu: (11, 10, 10) — Face on Bounded x
    # ρw: (10, 10, 11) — Face on Bounded z
    # -------------------------------------------------------------------
    @testset "Test 4: Multi-field asymmetry (simulating ρ, ρu, ρw)" begin
        N = 10
        α = 0.01

        rho  = Reactant.to_rarray(ones(Float64, N, N, N))
        rhou = Reactant.to_rarray(ones(Float64, N+1, N, N))
        rhow = Reactant.to_rarray(ones(Float64, N, N, N+1))
        drho  = Reactant.to_rarray(zeros(Float64, N, N, N))
        drhou = Reactant.to_rarray(zeros(Float64, N+1, N, N))
        drhow = Reactant.to_rarray(zeros(Float64, N, N, N+1))

        backend = KernelAbstractions.get_backend(rho)

        @kernel function multi_field_kernel!(rho, rhou, rhow, α)
            i, j, k = @index(Global, NTuple)
            @inbounds begin
                rho[i, j, k]  = rho[i, j, k]  + α * rho[i, j, k]
                rhou[i, j, k] = rhou[i, j, k] + α * rhou[i, j, k]
                rhow[i, j, k] = rhow[i, j, k] + α * rhow[i, j, k]
            end
        end

        function loss_v4(rho, rhou, rhow, α, nsteps)
            kern! = multi_field_kernel!(backend, KernelAbstractions.StaticSize((N, N, N)))
            @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
                kern!(rho, rhou, rhow, α)
            end
            return mean(rho .^ 2) + mean(rhou .^ 2) + mean(rhow .^ 2)
        end

        function grad_loss_v4(rho, drho, rhou, drhou, rhow, drhow, α, nsteps)
            parent(drho) .= 0
            parent(drhou) .= 0
            parent(drhow) .= 0
            _, loss_value = Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                loss_v4, Enzyme.Active,
                Enzyme.Duplicated(rho, drho),
                Enzyme.Duplicated(rhou, drhou),
                Enzyme.Duplicated(rhow, drhow),
                Enzyme.Const(α),
                Enzyme.Const(nsteps))
            return drho, drhou, drhow, loss_value
        end

        nsteps = 4

        @testset "Forward pass" begin
            @time "Compiling forward" compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss_v4(
                rho, rhou, rhow, α, nsteps)
            result = compiled_fwd(rho, rhou, rhow, α, nsteps)
            @test result > 0
            @test isfinite(result)
        end

        @testset "Backward pass (Enzyme)" begin
            @test_broken begin
                @time "Compiling backward" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss_v4(
                    rho, drho, rhou, drhou, rhow, drhow, α, nsteps)
                compiled_grad !== nothing
            end
        end
    end
end
