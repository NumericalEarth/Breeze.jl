---
paths:
  - test/**/*.jl
---

# Testing Rules

## Running Tests

```julia
# All tests
Pkg.test("Breeze")

# Specific test file (ParallelTestRunner autodiscovery)
Pkg.test("Breeze"; test_args=`atmosphere_model_construction`)

# CPU-only (disable GPU)
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
Pkg.test("Breeze")
```

## Writing Tests

- Use `default_arch` for architecture, `Oceananigans.defaults.FloatType` for precision
- Include both unit tests and integration tests
- Test numerical accuracy where analytical solutions exist
- Use minimal grid sizes to reduce CI time

## Debugging

- GPU "dynamic invocation error": run on CPU first to isolate GPU-specific issues
- Julia version issues: delete `Manifest.toml`, then `Pkg.instantiate()`
- Ensure doctests pass; run `quality_assurance.jl` for explicit imports and Aqua.jl checks

## Quality

- Ensure all explicit imports are correct (tests check this automatically)
- Always add tests for new functionality
- Make sure test files are actually included / discoverable by `runtests.jl`
