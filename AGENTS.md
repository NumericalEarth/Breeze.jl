# Breeze.jl — Agent Rules

## Project Overview

Breeze.jl is Julia software for simulating atmospheric flows.
It relies on [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) for grids, fields, solvers, and advection schemes,
with extensions to CloudMicrophysics for microphysical schemes, RRTMGP for radiative transfer solvers,
and interfaces with ClimaOcean for coupled atmosphere-ocean simulations.

## Language & Environment

- **Julia 1.10+** | CPU and GPU (CUDA)
- **Key packages**: Oceananigans.jl, CloudMicrophysics.jl, RRTMGP.jl, ClimaOcean.jl,
                    KernelAbstractions.jl, CUDA.jl, Enzyme.jl, Reactant.jl
- **Style**: ExplicitImports.jl for source code; `using Oceananigans` and `using Breeze` for examples
- **Testing**: ParallelTestRunner.jl for distributed testing

## Critical Rules

### Kernel Functions (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions.jl)
- Kernels must be **type-stable** and **allocation-free**
- Use `ifelse` — never short-circuiting `if`/`else` or ternary `?`/`:` in kernels
- No error messages, no Models inside kernels
- Mark functions called inside kernels with `@inline`
- **Never loop over grid points outside kernels** — use `launch!`
- **Use literal zeros**: `max(0, a)` not `max(zero(FT), a)`. Julia handles type promotion.

### Type Stability & Memory

- All structs must be concretely typed. **Never use `Any` as a type parameter or field type.**
- Use the **materialization pattern**: user-facing constructor creates a "skeleton" struct with
  placeholder types (like `Nothing`), then `materialize_*` creates the fully-typed version.
- For mutable state within an immutable struct, use a `mutable struct` as the field type.
- Type annotations are for **dispatch**, not documentation
- Minimize allocation; favor inline computation

### Imports

- Source code: explicit imports (checked by tests). Never use `import` to extend functions;
  always use `Module.function_name(...) = ...` or `function Module.function_name() ... end`
- Exports at the top of module files, before other code
- Import Oceananigans/Breeze names first, then external packages
- Internal Breeze imports use absolute paths, not relative
- Examples/docs: rely on `using Oceananigans` and `using Breeze`

### Docstrings

- Use `$(TYPEDSIGNATURES)` from DocStringExtensions.jl (never write explicit signatures)
- **ALWAYS `jldoctest` blocks, NEVER plain `julia` blocks** — doctests are tested; plain blocks rot
- Include expected output after `# output`; prefer `show` methods over boolean comparisons
- **Citations**: Use inline `[Author (year)](@cite Key)` syntax woven into prose
- Use unicode for math (`θ`, `ρ`, `Π`), not LaTeX

### Software Design

- Minimize code duplication (allow only for trivial one-liners)
- When something would be better in Oceananigans, add a detailed TODO note
- Almost always extend functions in source code, not in examples
- Coding style: consult `docs/src/appendix/notation.md` for variable names
- Use math or English consistently in expressions; don't mix
- Keyword arguments: no-space for inline `f(x=1)`, single-space for multiline `f(a = 1, b = 2)`

## Naming Conventions

- **Files**: snake_case — `atmosphere_model.jl`
- **Types/Constructors**: PascalCase — `AtmosphereModel`
- **Functions**: snake_case — `compute_pressure!`
- **Kernels**: may prefix with underscore — `_kernel_function`
- **Variables**: English long name or unicode from `notation.md`. Add new variables to that table.
- **Avoid abbreviations**: `latitude` not `lat`, `temperature` not `temp`

## Module Structure

```
src/
├── Breeze.jl                  # Main module, exports
├── Thermodynamics/            # Thermodynamic states & equations
├── AtmosphereModels/          # Core atmosphere model logic
├── Microphysics/              # Cloud microphysics
├── TurbulenceClosures/        # Including those ported from Oceananigans
├── Advection.jl               # Advection operators for anelastic models
├── CompressibleEquations/     # Compressible dynamics
├── AnelasticEquations/        # Anelastic dynamics
├── ParcelModels/              # Parcel model dynamics
└── MoistAirBuoyancies.jl      # Legacy buoyancy for Oceananigans.NonhydrostaticModel
```

## Breeze Formulations

Breeze uses "formulations" for different equation sets. Currently `AnelasticDynamics` in conservation
form (all prognostics are densities) with two thermodynamic formulations:
  - `LiquidIcePotentialTemperatureThermodynamics` — prognostic `ρθ`
  - `StaticEnergyThermodynamics` — prognostic `ρe`

Planned: fully compressible formulation, `EntropyThermodynamics` (prognostic `ρη`).

## Common Pitfalls

1. **Type instability** in kernels — ruins GPU performance
2. **Overconstraining types**: use annotations for dispatch, not documentation
3. **Missing imports**: tests will catch this — add explicit imports
4. **Plain `julia` blocks in docstrings**: always use `jldoctest`
5. **Subtle bugs from missing method imports**, especially in extensions
6. **Never extend `getproperty`** to fix undefined property bugs — fix the caller instead
7. **"Type is not callable"**: variable name shadows a function — rename or qualify
8. **Quick fixes that break correctness**: if a test fails after a change, revisit the original edit
9. **Scope creep in PRs**: keep changes focused on a single concern
10. **Modifying Project.toml dependencies**: never add, remove, or change `[deps]` or `[weakdeps]`
    in the root `Project.toml` unless the task absolutely requires it. Dependency changes have
    wide-reaching consequences — they affect CI, load time, and downstream compatibility.
    Only touch `[compat]` bounds when explicitly asked.

## Git Workflow & Whitespace

Follow [ColPrac](https://github.com/SciML/ColPrac). Feature branches, descriptive commits,
update tests and docs with code changes, check CI before merging.

**PRs fail CI with trailing whitespace or trailing blank lines.** Before committing:
remove trailing whitespace, remove trailing blank lines, ensure file ends with exactly one newline.

## Agent Behavior

- Prioritize type stability and GPU compatibility
- Follow established patterns in existing code
- Add tests for new functionality; update exports when adding public API
- Reference physics equations in comments when implementing dynamics
- When unsure: study working examples first (BOMEX, RICO, etc.), look at similar
  Oceananigans implementations, review tests for usage patterns

## Further Reading

Detailed reference docs are in `.agents/` — read on demand:

| Document | Content |
|----------|---------|
| `.agents/testing.md` | Running tests, writing tests, debugging, QA |
| `.agents/documentation.md` | Building docs, fast builds, Literate.jl examples, doctest details |
| `.agents/validation.md` | Reproducing paper results, common issues, TC genesis |
| `.agents/physics-debugging.md` | Thermodynamic variables, diagnose-before-fix, model architecture |

### Auto-loading Rules

Rules in `.claude/rules/` load automatically when you touch matching files:
- `kernel-rules.md` — GPU kernel requirements (src/)
- `docstring-rules.md` — docstring and jldoctest conventions (src/)
- `testing-rules.md` — test writing and running (test/)
- `docs-rules.md` — documentation building and style (docs/)
- `examples-rules.md` — Literate.jl example conventions (examples/)

### Skills (slash commands)

- `/run-tests` — run targeted tests, prioritized by what's likely to break
- `/build-docs` — build documentation locally
- `/add-feature` — checklist for adding new physics/features
- `/new-simulation` — set up, run, and visualize a new simulation
- `/babysit-ci` — monitor CI, auto-fix small issues, retrigger flaky runs
