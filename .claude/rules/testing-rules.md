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
- Most CI time is compilation. Kernels are compiled for a grid's size and topology, so a second
  size or topology in a file compiles every kernel again: reuse one grid across testsets where the
  size is not what is being tested
- `MultirateInfinitesimalStep`, the default of split-explicit dynamics, takes two to four times
  longer to compile than `AcousticRungeKutta3`. A test that is not about the time stepper can pass
  `timestepper = :AcousticRungeKutta3`, as long as its file still steps the default somewhere
- Sweeps over configurations that each compile their own model (coefficient sets, terrain
  formulations, topologies) run one representative, and all members where
  `BREEZE_TEST_EXHAUSTIVE=true`: use `test_sweep` and `exhaustive_tests` from `test/setup.jl`

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
