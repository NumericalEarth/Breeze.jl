# Breeze.jl

Atmospheric fluid dynamics in Julia, built on Oceananigans.jl. See `AGENTS.md` for full guidelines.

## Quick Reference

- **Run tests**: `Pkg.test("Breeze")` or specific: `Pkg.test("Breeze"; test_args=\`dynamics\`)`
- **Build docs**: `julia --project=docs/ docs/make.jl`
- **CPU-only**: `ENV["CUDA_VISIBLE_DEVICES"] = "-1"` before testing

## Critical Rules

- **No `Any` types** -- all structs must be concretely typed
- **GPU kernels**: allocation-free, use `ifelse` not `if/else`, mark helpers `@inline`, no loops over grid points (use `launch!`)
- **Use literal zeros**: `max(0, a)` not `max(zero(FT), a)`
- **Explicit imports only** -- never use `import Module: func` to extend; use `Module.func(...) = ...`
- **Docstrings**: always `jldoctest` blocks (never plain `julia`), always include expected output, use `$(TYPEDSIGNATURES)`
- **Whitespace**: PRs fail CI without clean trailing whitespace. Remove trailing spaces, trailing blank lines; ensure single final newline
- **Examples run at full resolution on GPU** -- reduce resolution/switch to CPU only for local testing, always revert before committing
- **No unit conversions** in examples (no `* 1000` for kg to g)
- **Plot Fields directly** with Makie -- never use `interior(field, ...)` for plotting
- **Naming**: PascalCase types, snake_case functions/files, no abbreviations. See `docs/src/appendix/notation.md`
- **Minimize code duplication** -- consider upstream Oceananigans changes when Breeze code gets awkward

## Architecture

```
src/
├── Breeze.jl                    # Main module, 96 exports
├── Thermodynamics/              # Thermodynamic states & equations
├── AtmosphereModels/            # Core model logic, diagnostics, interfaces
├── AnelasticEquations/          # Anelastic dynamics (default)
├── CompressibleEquations/       # Fully compressible dynamics
├── PotentialTemperatureFormulations/  # ρθ conservation
├── StaticEnergyFormulations/    # ρe conservation
├── Microphysics/                # Cloud microphysics schemes
├── TurbulenceClosures/          # Subgrid models
├── TimeSteppers/                # SSP RK3, acoustic substepping
├── BoundaryConditions/          # Surface fluxes, drag
├── Forcings/                    # Geostrophic, subsidence
└── KinematicDriver/             # Prescribed velocity simulations
ext/
├── BreezeCloudMicrophysicsExt/  # CloudMicrophysics.jl integration
└── BreezeRRTMGPExt/             # RRTMGP.jl radiative transfer
```

## Key Patterns

- **Materialization**: user constructors create skeleton structs, `materialize_*` creates fully-typed versions
- **Interfaces**: dynamics, formulations, microphysics, forcings all define interface contracts
- **Convention**: `model` for AtmosphereModel instances, `simulation` for Simulation instances
- **`set!` once**: call `set!(model, ...)` ideally once; it calls `update_state!` internally

## Git & CI

- Follow ColPrac. Feature branches, descriptive commits
- Clean whitespace before committing (see AGENTS.md for scripts)
- Run `quality_assurance.jl` for Aqua.jl checks
