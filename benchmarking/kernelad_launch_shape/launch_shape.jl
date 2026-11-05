# Runs convert-parallel-to-gpu1 on launch_shape.mlir as the kernel AD pipeline does
# (emit_single_alternative=true) and as the Benchmarks workaround does (no option, with
# POLYGEIST_GPU_KERNEL_BLOCK_SIZE=256), prints the launch each kernel gets, and tests them.
#
#   julia --project=<environment> launch_shape.jl launch_shape.mlir
#
# The environment needs Reactant.jl at the kernel AD entry's `reactant_rev`, with Reactant_jll's
# libReactantExtra_path preference at an Enzyme-JAX build of `enzyme_jax_branch`. The pass is
# target independent, so a CPU build reproduces it (see .github/workflows/KernelADLaunchShape.yml).
using Reactant
using Test

const MLIR = Reactant.MLIR

# The launch of each function: `nothing` if it was replaced by the error code 701, otherwise
# its grid and block sizes
function launches(mod)
    text = string(mod)
    result = Dict{String, Any}()
    for m in eachmatch(r"func\.func @(\w+)(.*?)(?=func\.func|\z)"s, text)
        name, body = m.captures
        launch = match(r"blocks\(.*?\) in \(%\w+ = %c(\d+), %\w+ = %c(\d+), %\w+ = %c(\d+)\) threads\(.*?\) in \(%\w+ = %c(\d+), %\w+ = %c(\d+), %\w+ = %c(\d+)\)", body)
        result[name] = if occursin("constant 701", body) || launch === nothing
            nothing
        else
            sizes = parse.(Int, launch.captures)
            (blocks = Tuple(sizes[1:3]), threads = Tuple(sizes[4:6]))
        end
    end
    return result
end

function run_pass(source, pipeline, block_size)
    if block_size === nothing
        delete!(ENV, "POLYGEIST_GPU_KERNEL_BLOCK_SIZE")
    else
        ENV["POLYGEIST_GPU_KERNEL_BLOCK_SIZE"] = block_size
    end
    println("=== ", pipeline, block_size === nothing ? "" : ", POLYGEIST_GPU_KERNEL_BLOCK_SIZE=$block_size")
    result = nothing
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        MLIR.IR.activate(ctx)
        try
            mod = parse(MLIR.IR.Module, source)
            Reactant.Compiler.run_pass_pipeline!(mod, pipeline)
            result = launches(mod)
        finally
            MLIR.IR.deactivate(ctx)
        end
    end
    for name in sort!(collect(keys(result)))
        println("  @", name, ": ", something(result[name], "error 701, the kernel is dropped"))
    end
    return result
end

threads(launch) = launch === nothing ? 0 : prod(launch.threads)

source = read(ARGS[1], String)

kernelad = run_pass(source, "convert-parallel-to-gpu1{emit_single_alternative=true}", nothing)
workaround = run_pass(source, "convert-parallel-to-gpu1", "256")

@testset "kernel AD launch shape" begin
    # Broken while lower-kernel-to-polygeist does not record which loop dimensions are threads.
    # An unexpected pass means Enzyme-JAX fixed it, and Benchmarks.yml's `kernelad_block_size`
    # workaround can go
    @testset "emit_single_alternative=true" begin
        @test_broken threads(kernelad["three_dims"]) > 1
        @test_broken kernelad["four_dims"] !== nothing
    end
    @testset "POLYGEIST_GPU_KERNEL_BLOCK_SIZE=256" begin
        @test threads(workaround["three_dims"]) == 256
        @test threads(workaround["four_dims"]) == 256
    end
end
