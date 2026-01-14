#####
##### Test utilities for Breeze.jl tests
#####

"""
    test_float_types()

Returns the float types to test. By default returns `(Float64,)` for faster tests.
Set environment variable `BREEZE_TEST_FLOAT32=true` to include Float32 testing.

This reduces test time by ~50% while still catching float-type-specific issues
in nightly/full CI runs.
"""
function test_float_types()
    if get(ENV, "BREEZE_TEST_FLOAT32", "false") == "true"
        return (Float32, Float64)
    else
        return (Float64,)
    end
end

"""
    all_float_types()

Returns both Float32 and Float64. Use this for tests that specifically need
to verify Float32 behavior (e.g., precision-sensitive calculations).
"""
all_float_types() = (Float32, Float64)
