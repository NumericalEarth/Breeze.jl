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
end

runtests(Breeze, args; testsuite, init_code)
