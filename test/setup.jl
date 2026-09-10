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

# Work around for <https://github.com/JuliaLang/julia/issues/54998>, often seen
# with Reactant code (and often only on the CI machine).
macro with_stack_size(stack_size, expr)
    return quote
        local _size = $(esc(stack_size))
        _size isa Integer || error("Stack size must be an integer")
        local task = Task(() -> $(esc(expr)), _size)
        schedule(task)
        wait(task)
        fetch(task)
    end
end

macro with_stack_size(expr)
    return :(@with_stack_size 16 << 20 $(esc(expr)))
end
