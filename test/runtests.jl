import Breeze
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests
using Test

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# Parse arguments
args = parse_args(ARGS)

const REACTANT_COMPAT = !((VERSION >= v"1.12") || (Base.JLOptions().check_bounds == 1))

if filter_tests!(testsuite, args)
    # Reactant compilation tests require --check-bounds=auto (Reactant/Enzyme
    # limitation) and Julia < 1.12 until upstream support is improved.
    if !REACTANT_COMPAT
        delete!(testsuite, "reactant_centered_compilation")
    end

    # WENO compilation is always removed from the parallel suite and run in
    # its own subprocess to avoid OOM under ParallelTestRunner.
    delete!(testsuite, "reactant_weno_compilation")
end

const init_code = quote
    import CUDA
    using Oceananigans.Architectures: CPU, GPU

    if get(ENV, "BREEZE_ENSURE_CUDA_FUNCTIONAL", "") == "true"
        CUDA.functional() || error("CUDA is not functional but we expect it to be, make sure it's set up correctly")
    end

    const default_arch = CUDA.functional() ? GPU() : CPU()

    # Float type helpers for tests
    # Default: Float64 only. Set BREEZE_TEST_FLOAT32=true to also test Float32.
    function test_float_types()
        if get(ENV, "BREEZE_TEST_FLOAT32", "false") == "true"
            return (Float32, Float64)
        else
            return (Float64,)
        end
    end

    # Returns both Float32 and Float64 for tests that need both precision levels
    all_float_types() = (Float32, Float64)
end

@testset "Breeze" begin
    runtests(Breeze, args; testsuite, init_code)

    # Run WENO compilation in an isolated subprocess to avoid OOM when
    # sharing memory with the parallel test runner (mirrors the pattern
    # used for MPI tests in Reactant's test suite).
    if REACTANT_COMPAT && (
        isempty(args.positionals) ||
        "reactant_weno_compilation" ∈ args.positionals
    )
        @testset "Reactant WENO compilation (subprocess)" begin
            test_file = joinpath(@__DIR__, "reactant_weno_compilation.jl")
            code = """
            import CUDA
            using Oceananigans.Architectures: CPU, GPU
            const default_arch = CUDA.functional() ? GPU() : CPU()
            include($(repr(test_file)))
            """
            run(`$(Base.julia_cmd()) --check-bounds=auto --project=$(Base.active_project()) -e $code`)
        end
    end
end
