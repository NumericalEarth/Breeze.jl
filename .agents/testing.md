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

| Test file | What it covers |
|-----------|---------------|
| `unit_tests.jl` | Core unit tests |
| `atmosphere_model_construction.jl` | Model construction |
| `set_atmosphere_model.jl` | Setting model fields |
| `dynamics.jl` | Dynamical core |
| `tracer_dynamics.jl` | Tracer transport |
| `diagnostics.jl` | Diagnostic fields |
| `saturation_adjustment.jl` | Thermodynamic saturation |
| `reference_states.jl` | Reference state profiles |
| `cloud_microphysics_0M.jl` | 0-moment microphysics |
| `cloud_microphysics_1M.jl` | 1-moment microphysics |
| `cloud_microphysics_2M.jl` | 2-moment microphysics |
| `turbulence_closures.jl` | Turbulence closures |
| `vertical_diffusion.jl` | Vertical diffusion |
| `forcing_and_boundary_conditions.jl` | Forcing and BCs |
| `anelastic_pressure_solver_*.jl` | Pressure solver tests |
| `quality_assurance.jl` | Explicit imports, Aqua.jl |
| `doctests.jl` | Doctest verification |
| `reactant_*.jl` | Reactant compilation |

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
