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
- Every `test/*.jl` file is discovered automatically as a test. Helper files must be removed from
  the test suite in `test/runtests.jl`, as `setup` is
- `reactant/` tests run only with `--check-bounds=auto` on Julia older than 1.14

## Distributed (MPI) Tests

Multi-rank tests use a driver/worker split, following Oceananigans.jl's `test/distributed`:

- `test/distributed/<name>.jl` is a normal ParallelTestRunner test (a *driver*), run serially. It
  launches `mpiexec -n N julia ... test/mpi/<name>.jl <args>` and checks the exit status and a
  sentinel line.
- `test/mpi/<name>.jl` is the *worker*, run in lockstep on every rank. `test/mpi/` is excluded from
  autodiscovery in `test/runtests.jl`. Workers can be run stand-alone:
  `mpiexec -n 4 julia --project=test test/mpi/acoustic_wave.jl xy`.
