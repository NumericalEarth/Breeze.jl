# Breeze.jl suggestions for agent-coders

## Project Overview

Breeze.jl is Julia software for simulating atmospheric flows.
Breeze relies on [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) for grids, fields, solvers, and advection schemes.
Breeze has extensions to CloudMicrophysics for microphysical schemes, RRTMGP for radiative transfer solvers.
Breeze interfaces with ClimaOcean for coupled atmosphere-ocean simulations.

## Language & Environment
- **Language**: Julia 1.10+
- **Architectures**: CPU and GPU
- **Key Packages**: Oceananigans.jl, CloudMicrophysics.jl, RRTMGP.jl, ClimaOcean.jl,
                    KernelAbstractions.jl, CUDA.jl, Enzyme.jl, Reactant.jl, Documenter.jl
- **Testing**: ParallelTestRunner.jl for distributed testing

## Code Style & Conventions

### Julia practices

1. **Explicit Imports**: Use `ExplicitImports.jl` style — explicitly import all used functions/types.
   Tests automatically check for proper imports. Never use `import` to extend functions;
   always use `Module.function_name(...) = ...` or `function Module.function_name() ... end`.

2. **Type Stability**: Prioritize type-stable code for performance.
   - All structs must be concretely typed. **Never use `Any` as a type parameter or field type.**
   - Use the **materialization pattern**: user-facing constructor creates a "skeleton" struct with
     placeholder types (like `Nothing`), then `materialize_*` creates the fully-typed version.
   - For mutable state within an immutable struct, use a `mutable struct` as the field type.

3. **Kernel Functions** (GPU compatibility):
   - Use KernelAbstractions.jl (`@kernel`, `@index`). Keep kernels type-stable and allocation-free.
   - Use `ifelse` instead of `if`/`else` or ternary `?`/`:`. No error messages inside kernels.
   - Models _never_ go inside kernels. Mark functions inside kernels with `@inline`.
   - **Never use loops outside kernels**: Replace `for` loops over grid points with `launch!` kernels.
   - **Use literal zeros**: `max(0, a)` not `max(zero(FT), a)`. Julia handles type promotion.

4. **Documentation**:
   - Use `$(TYPEDSIGNATURES)` from DocStringExtensions.jl (never write explicit signatures).
   - **Citations**: Use inline `[Author (year)](@cite Key)` syntax woven into prose.
     Avoid separate "References" sections with bare `[Key](@cite)`.

5. **Memory leanness**: Favor inline computations over temporary allocations.
   If an implementation is awkward, suggest an upstream Oceananigans feature instead.

6. **Debugging**: Version compatibility issues often resolve by deleting `Manifest.toml`
   and running `Pkg.instantiate()`.

7. **Software design**: Minimize code duplication (allow duplication only for trivial one-liners).
   When something would be better in Oceananigans, add a detailed TODO note.

8. **Extending functions**: Almost always extend in source code, not in examples.

### Oceananigans ecosystem best practices

1. **Coding style**
  - Consult `docs/src/appendix/notation.md` for variable names.
  - Use math or English consistently in expressions; don't mix.
  - Keyword arguments: no-space for inline `f(x=1)`, single-space for multiline `f(a = 1, b = 2)`.
  - Use `const` _only when necessary_.
  - `TitleCase` for types/constructors, `snake_case` for functions/variables.
  - Number variables start with `N` (`Nx`, `Ny`, `Nt`). Spatial indices `i, j, k`, time index `n`.

2. **Import/export style**
  - Exports at the top of module files, before other code.
  - Import Oceananigans/Breeze names first, then external packages.
  - Internal Breeze imports use absolute paths, not relative.
  - Source code: explicitly import all names. Scripts: `using Oceananigans` and `using Breeze`.

3. **Examples and integration tests**
  - **Testing examples**: Reduce resolution and switch to CPU for debugging. **Always revert** before committing.
  - Use Literate style — let code speak for itself. Lighthearted, engaging prose.
  - Invoke `set!` ideally once (it calls `update_state!` internally).
  - Follow existing example style, not source code style.
  - Initial condition functions act _pointwise_ — no broadcasting inside them.
  - **CRITICAL — Do not convert units**: Keep units consistent with source code. Only exception: spatial coordinates to km for axis labels.
  - Use concise names and unicode consistent with source code and `notation.md`.
  - Always add axis labels and colorbars.
  - New examples should add new physics/value, not copy existing ones.
  - Prefer exported names. If too many internal names needed, export them or create a new abstraction.
  - Use `xnode`/`ynode`/`znode` for `discrete_form=true` forcing/BCs. Never access grid metrics manually.
  - Use `Oceananigans.defaults.FloatType = FT` for precision, not manual `FT(1)` conversions.
  - Use integers for integer values. Rely on "autotupling": `tracers = :c` not `tracers = (:c,)`.
  - Call models `model` and simulations `simulation`.
  - Examples use `examples/Project.toml`. Add example-specific packages there, not main `Project.toml`.
  - **CRITICAL — Plotting Fields**: NEVER use `interior(field, ...)`. Makie plots `Field` objects directly.
    Use `view(field, i, j, k)` to window fields. Works with `@lift` for animations too:
    ```julia
    # WRONG:
    data = @lift interior(field_ts[$n], :, 1, :)
    heatmap!(ax, x, z, data, ...)
    # CORRECT:
    field_n = @lift field_ts[$n]
    heatmap!(ax, field_n, ...)
    ```
  - Use suffix `ts` for time series, `n` for time-indexed fields.
  - **Color palette**: `:dodgerblue` (vapor), `:lime` (cloud), `:orangered` (rain), `:magenta` (temperature).

4. **Documentation Style**
  - Use unicode in math (e.g., ``θᵉ`` not ``\theta^e``); Documenter converts to LaTeX.
  - Always add `@ref` cross-references for Breeze functions. Link to Oceananigans docs for external functions.

5. **Common misconceptions**
  - `update_state!` is called within `set!` — rarely invoke manually.
  - `compute!` is called in `Field(op)` constructor — don't call it again.

6. **Doctests**
  - Always use `jldoctest` blocks, never plain `` ```julia `` blocks.
  - All doctests must include expected output.
  - Prefer exercising `Base.show` over equality comparisons.
  - For run-only verification, end with `typeof(result)` or a simple field access.

### Naming Conventions
- **Files**: snake_case (e.g., `atmosphere_model.jl`)
- **Types**: PascalCase (e.g., `AtmosphereModel`)
- **Functions**: snake_case (e.g., `compute_pressure!`)
- **Kernels**: May be prefixed with underscore (e.g., `_kernel_function`)
- **Variables**: English long name or unicode from `notation.md`. Add new variables to that table.
- **Avoid abbreviations**: `latitude` not `lat`, `temperature` not `temp`.

### Breeze Module Structure
```
src/
├── Breeze.jl                  # Main module, exports
├── Thermodynamics/            # Thermodynamic states & equations
├── AtmosphereModels/          # Core atmosphere model logic
├── Microphysics/              # Cloud microphysics
├── TurbulenceClosures/        # Including those ported from Oceananigans
├── Advection.jl               # Advection operators for anelastic models
└── MoistAirBuoyancies.jl      # Legacy buoyancy for Oceananigans.NonhydrostaticModel
```

Planned: extension in `ext/` for RRTMGP.jl, LagrangianParticleTracking module.

### Breeze formulations

Breeze uses "formulations" for different equation sets. Currently `AnelasticDynamics` in conservation
form (all prognostics are densities) with two thermodynamic formulations:
  - `LiquidIcePotentialTemperatureThermodynamics` — prognostic `ρθ`
  - `StaticEnergyThermodynamics` — prognostic `ρe`

Planned: fully compressible formulation, `EntropyThermodynamics` (prognostic `ρη`).

### Microphysics implementation

Interface in `src/AtmosphereModels/microphysics_interface.jl`. Key functions:
- `maybe_adjust_thermodynamic_state`: Saturation adjustment for equilibrium schemes; trivial for non-equilibrium (prognostic condensate) schemes.
- `microphysical_tendency`: Tendencies for prognostic microphysical variables.
- `moisture_fractions`: Moisture mass fractions from prognostic fields.
- `update_microphysical_fields!`: Update diagnostic fields after state update.

## Testing Guidelines

### Running Tests
```julia
Pkg.test("Breeze")                                       # All tests
Pkg.test("Breeze"; test_args=`atmosphere_model_unit`)     # Specific test file
ENV["CUDA_VISIBLE_DEVICES"] = "-1"; Pkg.test("Breeze")   # CPU-only
```

GPU "dynamic invocation error" → run on CPU. If it passes, the issue is GPU-specific (type inference).

### Writing Tests
- Use `default_arch` for architecture, `Oceananigans.defaults.FloatType` for precision.
- Include unit and integration tests. Test numerical accuracy against analytical solutions.

### Quality Assurance
- Ensure doctests pass. Run `quality_assurance.jl`. Use Aqua.jl for package checks.

### Fixing bugs
- Missing method imports cause subtle bugs, especially in extensions.
- Prefer exporting expected names over changing user scripts.
- **Never extend `getproperty`** to fix undefined property bugs — fix the caller instead.
- **"Type is not callable"**: Variable name conflicts with function name. Rename the variable or qualify the function.
- **Connecting dots**: If a test fails after a change, revisit that change. A fix that makes code _run_ may make it _incorrect_.

## Common Development Tasks

### Adding New Physics
1. Create module in appropriate subdirectory with docstrings
2. Implement GPU-compatible kernel functions
3. Add unit tests
4. Update exports in `src/Breeze.jl` if user interface changes
5. Add validation script in `examples/` or `validation/`

### Modifying AtmosphereModel
- Core: `src/AtmosphereModels/atmosphere_model.jl`
- State updates: `update_atmosphere_model_state.jl`
- Pressure: `anelastic_pressure_solver.jl`

## Documentation

### Building and Viewing
```sh
julia --project=docs/ docs/make.jl        # Build
julia -e 'using LiveServer; serve(dir="docs/build")'  # View
```

### Tips
- Manually run `@example` blocks rather than full doc builds to find errors.
- Don't write `for` loops in docs blocks unless asked. Use built-in functions.
- **Debugging literated examples**: Comment out all other examples in `docs/make.jl` to isolate failures.
- **Testing doc pages**: Comment out ALL examples in `docs/make.jl` to skip literation; iterate on `@example` blocks.
- **Literate.jl**: Lines starting with `# ` at column 1 become markdown. Use `##` for in-function comments.

## Important Files

### Core
- `src/Breeze.jl` — Main module, all exports
- `src/AtmosphereModels/atmosphere_model.jl` — Central model definition
- `src/AtmosphereModels/anelastic_formulation.jl` — Anelastic equations
- `src/Thermodynamics/` — Thermodynamic relations
- `Project.toml` — Dependencies and compat bounds
- `test/runtests.jl` — Test configuration

### Examples
- `examples/thermal_bubble.jl`, `free_convection.jl`, `anelastic_bomex.jl`, `boussinesq_bomex.jl`
- `examples/mountain_wave.jl`, `cloudy_kelvin_helmholtz.jl`, `tropical_cyclone_world_v2.jl`
- Many more in `examples/`

## Physics Domain Knowledge

### Atmospheric Dynamics
- Anelastic approximation filters acoustic waves
- Moist thermodynamics via saturation adjustment
- Reference state defines background stratification
- Hydrostatic pressure computed diagnostically

### Numerical Methods
- Finite volume on Arakawa C-grid: velocities at faces, tracers at centers
- Take care of staggered grid locations in operators and diagnostics
- Favor WENO advection. Pressure Poisson solver for anelastic divergence.
- Time stepping: RungeKutta3 (default), Adams-Bashforth, Quasi-Adams-Bashforth

## Implementing Validation Cases

When reproducing paper results:

1. **Extract ALL parameters**: domain, resolution, constants, BCs, ICs, forcing, closure. Check tables and figure captions.
2. **Verify geometry first**: Visualize grid/domain before long runs. Compare to paper figures.
3. **Verify ICs**: Check `minimum`/`maximum` of fields. Visualize spatial distribution.
4. **Short test runs**: Few timesteps on CPU at low resolution. Check for NaNs, reasonable velocities, meaningful output. Then test GPU.
5. **Progressive validation**: Run ~1 hour sim time, visualize, compare to early-time paper figures.
6. **Match paper figures**: Same colormaps, axis ranges, time snapshots, diagnostics.

### Common Issues
- **NaN blowups**: Timestep too large, unstable ICs, or `if`/`else` on GPU (use `ifelse`)
- **Nothing happening**: Wrong sign on buoyancy anomaly, ICs not applied, forcing inactive
- **Wrong flow direction**: Check coordinate conventions
- **GPU issues**: Avoid branching, ensure type stability

## Common Pitfalls

1. **Type Instability**: Especially in kernels — ruins GPU performance
2. **Overconstraining types**: Use type annotations for _dispatch_, not documentation
3. **Forgetting Explicit Imports**: Tests will fail
4. **Plain `julia` blocks in docstrings**: Always use `jldoctest`

## Debugging Physics Simulations

### Thermodynamic Variable Discipline

| Variable | Meaning |
|----------|---------|
| `T` | Temperature (K) |
| `θ` | Potential temperature: `θ = T / Π` where `Π = (p/p₀)^κ` |
| `ρe` | Density × total energy (J/m³) |
| `ρθ` | Density × potential temperature (kg·K/m³) |

Before applying forcing: (1) check what variable the paper uses, (2) check working examples,
(3) check Breeze's prognostic variable, (4) verify units.

**Common mistakes**: Applying T tendency to θ, confusing `ρe` with `ρθ`, forgetting Exner function in T↔θ conversion.

### When a Stable Simulation Becomes Unstable

1. **STOP** — Don't add fixes. 2. Identify last working state via `git log`/`git diff`.
3. Revert. 4. Make ONE change at a time. 5. Find the breaking change.

The instability is NOT pre-existing if the code was stable before your changes.

### Diagnose-Before-Fix Protocol

1. **STOP** — Don't immediately try a fix.
2. **Characterize**: Where? What values? When did it start?
3. **Work backwards**: Extreme at high altitude → what's special there? NaN → division by small numbers?
4. **Compute analytically**: Expected tendency? Physically reasonable?
5. **Only then** propose a targeted fix.

**Anti-pattern**: "Blows up at high altitude → cap values." This treats symptoms.
**Correct**: "Why high altitude? Low Π → amplified forcing → fix: equilibrate initial condition."

### Model Architecture Awareness

When implementing from papers using different models (SAM, WRF, MPAS):
1. Identify the paper's prognostic variables and how forcing is applied.
2. Identify Breeze's prognostics (`ρθ` or `ρe`).
3. Derive the transformation (e.g., ∂θ/∂t = ∂T/∂t × 1/Π — can amplify 10× at high altitude!).
4. Check if the paper's model handles this implicitly (e.g., SAM uses static energy ∝ T).

### Checklist Before Modifying Physics Code

- [ ] Read relevant working examples (BOMEX, RICO, prescribed_SST)
- [ ] Identified which field applies similar physics
- [ ] Verified implementation matches paper specification
- [ ] Computed tendency magnitudes analytically at key locations
- [ ] Verified ICs are compatible with forcing
- [ ] Making ONE change only
- [ ] Committed or stashed current working state

## Git Workflow
- Follow ColPrac. Feature branches for new work. Descriptive commit messages.
- Update tests and docs with code changes. Check CI before merging.

## Code Formatting and Whitespace

**PRs fail CI with trailing whitespace or trailing blank lines.** Before committing:
1. Remove trailing whitespace from each line
2. Remove trailing blank lines at end of file
3. Ensure file ends with exactly one newline

```bash
for file in $(find /path/to/Breeze -type f \( -name "*.jl" -o -name "*.md" -o -name "*.sh" \) ! -path "*/.git/*"); do
  sed -i '' 's/[[:space:]]*$//' "$file"
  awk 'NF {p=1} p' "$file" | awk '{print}' > "$file.tmp" && mv "$file.tmp" "$file"
  [ -s "$file" ] && [ "$(tail -c1 "$file" | wc -l)" -eq 0 ] && echo >> "$file"
done
```

## Helpful Resources
- [Oceananigans docs](https://clima.github.io/OceananigansDocumentation/stable/)
- [Breeze docs](https://numericalearth.github.io/BreezeDocumentation/dev/)
- [Discussions](https://github.com/NumericalEarth/Breeze.jl/discussions)
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
- [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) / [docs](https://enzymead.github.io/Reactant.jl/stable/)
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) / [docs](https://enzyme.mit.edu/julia/dev)
- [YASGuide](https://github.com/jrevels/YASGuide), [ColPrac](https://github.com/SciML/ColPrac)

## When Unsure
1. Study working examples first (BOMEX, RICO, etc.)
2. Look at similar Oceananigans.jl implementations
3. Review tests for usage patterns
4. Check `docs/src/` or ask in GitHub discussions

## AI Assistant Behavior
- Prioritize type stability and GPU compatibility
- Follow established patterns in existing code
- Add tests for new functionality, update exports for new public API
- Reference physics equations in comments when implementing dynamics

## Interactive Julia REPL (MCPRepl.jl)

[MCPRepl.jl](https://github.com/kahliburke/MCPRepl.jl) exposes a Julia REPL via MCP for interactive development.

### Setup
```julia
# Install (one-time, global environment)
using Pkg; Pkg.activate(); Pkg.add(url="https://github.com/kahliburke/MCPRepl.jl")
using MCPRepl; MCPRepl.quick_setup(:lax)

# Start server
using Revise, MCPRepl, Oceananigans, Breeze
MCPRepl.start_proxy(port=3000)  # Dashboard at localhost:3000/dashboard
```

### Cursor Configuration
Create `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "julia-repl": {
      "url": "http://localhost:3000",
      "transport": "http",
      "headers": { "X-MCPRepl-Target": "Breeze.jl" }
    }
  }
}
```

### Available Tools
`julia_eval`, `lsp_goto_definition`, `lsp_find_references`, `lsp_rename`, `lsp_document_symbols`, `lsp_code_actions`

With Revise.jl, source edits are picked up automatically — no restart needed.

## Tropical Cyclone Genesis (Cronin & Chavas 2019)

| Case | Genesis | Requirements |
|------|---------|--------------|
| Moist (β=1) | Spontaneous | 8km resolution, forms in ~5 days |
| Dry (β=0) | Needs assistance | 2km resolution, seeding, or extreme forcing |

**Key insight**: Latent heat enables WISHE feedback for self-aggregation.

### Critical Parameters
- **Domain**: ≥1152 km for vortex merger cascade (576 km → lattice equilibrium)
- **Resolution**: 2km for dry TCs, 4-8km for moist TCs
- **Disequilibrium**: Tₛ - θ_surface ≈ 10-15 K typical for RCE

### Failure Modes
| Symptom | Cause | Fix |
|---------|-------|-----|
| No TC formation | Domain too small | Lx, Ly ≥ 1152 km |
| Simulation blows up | T far from equilibrium | Equilibrated θ profile |
| Flat intensity | Weak forcing (dry) | Moist physics or seed |

Monitor: max surface wind, max ζ/f, mean θ profile, spatial wind/vorticity plots.

## Roadmap

- Fully compressible formulation (explicit + acoustic substepping)
- Total energy with anelastic model
- Open boundaries (following Oceananigans.NonhydrostaticModel)
- CloudMicrophysics schemes, superdroplet microphysics
- MPI / distributed GPU
- More canonical LES validation cases
- Terrain-following coordinate, cut-cells ImmersedBoundaryGrid
