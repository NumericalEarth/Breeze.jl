# Testing Guidelines

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

GPU "dynamic invocation error" → run on CPU. If it passes, the issue is GPU-specific (type inference).

## Available Test Files

Every `test/<name>.jl` is a test name for `test_args`; `ls test/` lists them all. A few anchors:

| Test file | What it covers |
|-----------|---------------|
| `unit_tests.jl` | Core unit tests |
| `atmosphere_model_construction.jl` | Model construction |
| `dynamics.jl` | Dynamical core |
| `saturation_adjustment.jl` | Thermodynamic saturation |
| `cloud_microphysics_1M.jl`, `cloud_microphysics_2M.jl` | 1- and 2-moment microphysics |
| `predicted_particle_properties_*.jl`, `p3_*.jl` | P3 microphysics |
| `acoustic_substepping_*.jl` | Compressible dynamics |
| `terrain_following_*.jl` | Terrain-following coordinates |
| `*_radiative_transfer.jl` | Radiation extensions |
| `quality_assurance.jl` | Explicit imports, Aqua.jl |
| `doctests.jl` | Doctest verification |
| `reactant/` | Reactant compilation |
| `distributed/*.jl` | Distributed architecture test driver |
| `mpi/*.jl` | Implementation of distributed test |

## Distributed (MPI) Tests

Multi-rank tests use a driver/worker split, following Oceananigans.jl's `test/distributed`:

- `test/distributed/<name>.jl` is a normal ParallelTestRunner test (a *driver*). It launches
  `mpiexec -n N julia ... test/mpi/<name>.jl <args>` and checks the exit status and a sentinel line.
- `test/mpi/<name>.jl` is the *worker*, run in lockstep on every rank. `test/mpi/` is excluded from
  autodiscovery in `runtests.jl`. Workers can be run stand-alone:
  `mpiexec -n 4 julia --project=test test/mpi/acoustic_wave.jl xy`.

## Writing Tests

- Use `default_arch` for architecture, `Oceananigans.defaults.FloatType` for precision
- Include unit and integration tests. Test numerical accuracy against analytical solutions.
- Use minimal grid sizes to reduce CI time

## Quality Assurance

- Ensure doctests pass. Run `quality_assurance.jl`. Use Aqua.jl for package checks.
- `quality_assurance.jl` checks explicit imports — run this for any change

## Fixing Bugs

- Missing method imports cause subtle bugs, especially in extensions
- Prefer exporting expected names over changing user scripts
- **Never extend `getproperty`** to fix undefined property bugs — fix the caller instead
- **"Type is not callable"**: Variable name conflicts with function name. Rename the variable or qualify the function.
- **Connecting dots**: If a test fails after a change, revisit that change. A fix that makes code _run_ may make it _incorrect_.

## Debugging Tips

- Version compatibility issues often resolve by deleting `Manifest.toml` and running `Pkg.instantiate()`
- GPU tests may fail with "dynamic invocation error". Run on CPU first to isolate GPU-specific issues.
