import Breeze
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# Parse arguments
args = parse_args(ARGS)

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

runtests(Breeze, args; testsuite, init_code)