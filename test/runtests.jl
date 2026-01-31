import Breeze
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests, PTRWorker

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# Create custom worker without --check-bounds=yes (causes Reactant crash)
function create_worker_without_checkbounds()
    exeflags = String[]
    for flag in Base.julia_cmd().exec[2:end]
        startswith(flag, "--check-bounds") && continue
        push!(exeflags, flag)
    end
    push!(exeflags, "--startup-file=no")
    push!(exeflags, "--depwarn=yes")
    push!(exeflags, "--project=$(Base.active_project())")
    push!(exeflags, "--color=yes")
    env = ["JULIA_NUM_THREADS" => "1", "OPENBLAS_NUM_THREADS" => "1"]
    return PTRWorker(; exeflags, env)
end

const custom_worker = create_worker_without_checkbounds()

# Parse arguments
args = parse_args(ARGS)

if filter_tests!(testsuite, args)
    # Skip Enzyme/Reactant tests in Julia v1.12+ until upstream
    # support is improved.
    if VERSION >= v"1.12"
        delete!(testsuite, "differentiation")
    end
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

runtests(Breeze, args; testsuite, init_code, test_worker = _ -> custom_worker)
