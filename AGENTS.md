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
- **Testing**: Breeze.jl uses ParallelTestRunner.jl for distributed testing

## Code Style & Conventions

### Julia practices and information

1. **Explicit Imports**: Use `ExplicitImports.jl` style - explicitly import all used functions/types
   - Import from Oceananigans explicitly (already done in `src/Breeze.jl`)
   - Tests automatically check for proper imports
   - Never use the `import` keyword for.
     To extend functions from other modules always use the syntax `Module.function_name(...) = ...` or `function Module.function_name() ... end`, not `import Module: function_name; function_name(...) = ...`

2. **Type Stability**: Prioritize type-stable code for performance
   - All structs must be concretely typed
   - **Never use `Any` as a type parameter or field type**. This destroys type inference and performance.
   - Use the **materialization pattern**: define a user-facing constructor that creates a "skeleton"
     struct with placeholder types (like `Nothing`), then `materialize_*` functions create the
     fully-typed version with concrete field types. This allows deferred type resolution while
     maintaining concrete types in the final object.
   - For mutable state within an immutable struct, use a `mutable struct` as the field type.
     The outer struct remains immutable with concrete types, while the inner mutable struct's
     fields can be updated. Example: `ParcelDynamics` contains a `ParcelState` (mutable) that
     gets its fields updated during time-stepping.

3. **Kernel Functions**: For GPU compatibility:
   - Use KernelAbstractions.jl syntax for kernels, eg `@kernel`, `@index`
   - Keep kernels type-stable and allocation-free
   - Short-circuiting if-statements should be avoided if possible. This includes
     `if`... `else`, as well as the ternary operator `?` ... `:`. The function `ifelse` should be used for logic instead.
   - Do not put error messages inside kernels.
   - Models _never_ go inside kernels
   - Mark functions inside kernels with `@inline`.
   - **Never use loops outside kernels**: Always replace `for` loops that iterate over grid points
     with kernels launched via `launch!`. This ensures code works on both CPU and GPU.
   - **Use literal zeros**: Write `max(0, a)` instead of `max(zero(FT), a)`. Julia handles type
     promotion automatically, and `0` is more readable. The same applies to `min`, `clamp`, etc.

4. **Documentation**:
   - Use DocStringExtensions.jl for consistent docstrings
   - Use `$(TYPEDSIGNATURES)` for automatic typed signature documentation (preferred over `$(SIGNATURES)`)
   - Never write explicit function signatures in docstrings; always use `$(TYPEDSIGNATURES)`
   - Add examples in docstrings when helpful
   - **Citations in docstrings**: Use inline citations with `[Author1 and Author2 (year)](@cite Key)` or `[Author1 et al. (Year)](@cite Key)` syntax.
     Avoid separate "References" sections with bare `[Key](@cite)` - these just show citation keys in the REPL
     without context, which is not helpful. Instead, weave citations naturally into the prose, e.g.:
     "Tetens' formula [Tetens1930](@citet) is an empirical formula..."

5. **Memory leanness**
   - Favor doing computations inline versus allocating temporary memory
   - Generally minimize memory allocation
   - If an implementation is awkward, don't hesitate to suggest an upstream feature (eg in Oceananigans)
     that will make something easier, rather than forcing in low quality code.

6. **Debugging**
   - Sometimes "julia version compatibility" issues are resolved by deleting the Manifest.toml,
     and then re-populating it with `using Pkg; Pkg.instantiate()`.

7. **Software design**
   - Try _very_ hard to minimize code duplication. Allow some code duplication for very small
     and simple functions, for example one-liners like `instantiate(X) = X()` that can be immediately
     understood. But for complicated infrastructure, re-use as much as possible.
   - Within Breeze, you will inevitably run into situations that would be better implemented by
     extending Oceananigans, rather than writing Breeze source code. When this happens, make a
     detailed and descriptive TODO note about what should be moved to Oceananigans.
8. **Extending functions**
   - Almost always extend functions in source code, not in examples

### Oceananigans ecosystem best practices

1. **General coding style**
  - Consult the Notation section in the docs (`docs/src/appendix/notation.md`) for variable names
  - Variables may take a "symbolic form" (often unicode symbols, useful when used in equations) or "English form" (more descriptive and self-explanatory). Use math and English consistently and try not to mix the two in expressions for clarity.
  - For keyword arguments, we like
    * "No space form" for inline functions: `f(x=1, y=2)`,
    * "Single space form for multiline representations:
    ```
    long_function(a = 1,
                  b = 2)
    ```
    * Variables should be declared `const` _only when necessary_, and not otherwise. This helps interpret the meaning and usage of variables. Do not overuse `const`.
  - `TitleCase` style is reserved for types, type aliases, and constructors.
  - `snake_case` style should be used for functions and variables (instances of types)
  - "Number variables" (`Nx`, `Ny`) should start with capital `N`. For number of time steps use `Nt`.
    Spatial indices are `i, j, k` and time index is `n`.

2. **Import/export style**
  - Write all exported names at the top of a module file, before any other code.
  - For explicit imports, import Oceananigans and Breeze names first. Then write imports for "external" packages.
  - For internal Breeze imports, use absolute paths, not relative paths.
  - Use different style for source code versus user scripts:
    * in source code, explicitly import all names into files
    * in scripts, follow the user interface by writing "using Oceananigans" and "using Breeze".
    * only use explicit import in scripts for names that are _not_ exported by the top-level files Oceananigans.jl, Breeze.jl etc.
    * sometimes we need to write `using Oceananigans.Units`

3. **Examples and integration tests**
  - **Testing examples**: When testing or debugging examples, reduce resolution and switch to CPU
    to speed up iteration. For example, change `Nx = Ny = 64` to `Nx = Ny = 16`, `Nz = 100` to
    `Nz = 20`, and `RectilinearGrid(GPU(); ...)` to `RectilinearGrid(CPU(); ...)`. You may also
    add `simulation.stop_iteration = 50` to limit runtime. **Always revert these changes** before
    committing - examples should run at full resolution on GPU for production.
  - Explain at the top of the file what a simulation is doing
  - Let code "speak for itself" as much as possible, to keep an explanation concise.
    In other words, use a Literate style.
  - Use a lighthearted, funny, engaging, style for example prose.
  - Use visualization interspersed with model setup or simulation running when needed.
    give an understanding of a complex grid, initial condition, or other model property.
  - Look at previous examples. New examples should add as much value as possible while remaining simple. This requires judiciously introducing new features and doing creative and surprising things with simulations that will spark readers' imagination.
  - For examples, or tests that invoke example-like code, invoke `set!` ideally once (or as few times as possible).
    There are two reasons: first, `set!` will determine the entire state of `AtmosphereModel`, and then call `update_state!` to fill halo regions and compute diagnostic variables. This only needs to be done once.
    The second reason is that it is easier to interpret a script by reading it when the initial condition is
    determined on one line rather than spread out over many lines.
  - Follow the style of existing examples, not the source code
  - Remember that initial condition functions act _pointwise_, there should be no broadcasting inside an initial condition function
  - **CRITICAL - Do not convert units**: Never multiply or divide by conversion factors (e.g., `* 1000` to convert
    kg/kg to g/kg). Always keep units consistent with the source code. If plotting requires different scales,
    consider plotting differences from initial conditions or using scientific notation in axis labels.
    The only exception is converting spatial coordinates to kilometers for axis labels.
  - If possible, avoid long underscore names. Use concise evocative names like `z = znodes(grid, Center())`.
  - Use unicode that is consistent with the source code. Do not be afraid of unicode for intermediate variables.
  - Make sure that all notation in examples is consistent with `docs/src/appendix/notation.md`
  - Always add axis labels and colorbars to simulations.
  - Check previous examples and strive to make new examples that add new physics and new value relative to old examples. Don't just copy old examples.
  - `@allowscalar` should very sparingly be used or never in an example. If you need to, make a suggestion to change the source code so that `@allowscalar` is not needed.
  - The examples should use exported names primarily. If an example needs an excessive amount of internal names, those names should be exported or a new abstraction needs to be developed.
  - For `discrete_form=true` forcing and boundary conditions, always use `xnode`, `ynode`, and `znode` from Oceananigans. _Never_ access grid metrics manually. Do not pass in
  - Use `Oceananigans.defaults.FloatType = FT` to change the precision; do not set precision within constructors manually.
  - Use integers when values are integers. Do not "eagerly convert" to Float64 by adding ".0" to integers.
  - Constructors should convert to `FT` under the hood, and it should be not be necessary to "manually convert" numbers to `FT`. In other words, we should not see `FT(1)` appearing very often,
  unless _absolutely_ necessary.
  - Keyword arguments that expect tuples (eg `tracers = (:a, :b)`) often "autotuple" single arguments. Always rely on this: i.e. use `tracers = :c` instead of `tracers = (:c,)` (the latter is more prone to mistakes and harder to read)
  - Instances of `AtmosphereModel` are almost always called `model`
  and instances of `Simulation` are called `simulation`.
  - The examples and docs have their own `Project.toml` environment. When your run examples you need to use `examples/Project.toml`.
    When you build new examples, please add example-specific packages to `examples/Project.toml`. Do not add example-specific
    packages to the main Breeze Project.toml. You may also need to add relevant packages to AtmosphereProfilesLibrary.
  - When making plots, do not use `interior(field, i, j, k)` to make a plot. Instead either pass `field`
    directly, as in `lines!(ax, field)` or `lines!(ax, z, field)` for a 1D plot with either automatic
    or custom vertical coordinate. You only need to provide the coordinate if it has different units, eg
    if you have converted z to kilometers. 1D fields work with `lines` and 2D fields work with 2D plots
    like `heatmap` or `contourf`. For 3D fields in a 2D plane, use `view(field, :, :, k)`
    (e.g. for a xy-slice).
  - **CRITICAL - Plotting Fields**: NEVER use `interior(field, ...)` for plotting. Makie/CairoMakie
    can plot `Field` objects directly, which automatically handles coordinates and is cleaner.
    Use `view(field, i, j, k)` to window fields if needed. This applies to both static plots and
    animations with `@lift`. For example:
    ```julia
    # WRONG - do not do this:
    data = @lift interior(field_ts[$n], :, 1, :)
    heatmap!(ax, x, z, data, ...)

    # CORRECT - pass Field directly:
    field_n = @lift field_ts[$n]
    heatmap!(ax, field_n, ...)
    ```
  - In examples, use the suffix `ts` (no underscore) for "time series" and the suffix `n` (no underscore)
    to refer to `FieldTimeSeries` indexed at time-index `n`.
  - **Preferred color palette**: Use bright, colorblind-friendly colors for plots:
    ```julia
    c_vapor = :dodgerblue      # Bright blue for vapor/moisture
    c_cloud = :lime            # Vivid green for cloud liquid
    c_rain = :orangered        # Bright orange-red for rain/precipitation
    c_temp = :magenta          # Vibrant magenta for temperature
    ```
    These colors are distinct, high-contrast, and accessible for colorblind viewers.

4. **Documentation Style**
  - Mathematical notation in `docs/src/appendix/notation.md`
  - Use Documenter.jl syntax for cross-references
  - Include code examples in documentation pages
  - Add references to papers from the literature by adding bibtex to `breeze.bib`, and then
    a corresponding citation
  - Make use of cross-references with equations
  - When writing math expressions, use unicode equivalents as much as possible and leverage the automatic conversion
    to latex that Documenter will do under the hood. This will make the source code more readable.
    For example, use ``θᵉ`` instead of ``\theta^e``.
  - Always add cross-references for Breeze functions, eg using `@ref`. For functions in Oceananigans,
    provide explicit links to Oceananigans documentation.
    Cross-references should be attached to every code object mentioned at least once in every documentation file,
    and often more than once for long files with multiple sections.

5. **Other tips and common misconceptions**
  - `update_state!` is called within `set!(model, ...)`. Scripts should never or rarely need to manually invoke `update_state!`
  - Fields and AbstractOperations can be used in `set!`.
  - `compute!` is called in the `Field(op)` constructor for `op::AbstractOperation`. It is redundant to call `compute!`
    immediately after building a `Field`.

6. **Doctests**
  - Always use `jldoctest` blocks, never plain code blocks (`` ```julia ``). Plain code blocks are not tested.
  - All doctests must include expected output. A doctest without output will fail.
  - Doctests should exercise `Base.show` rather than equality comparisons. The purpose is to verify
    that objects display correctly and that the code runs without error.
  - Developing a doctest typically involves ensuring that `show` for a newly defined object looks good
    and is human-readable. This can require work on nested structs to develop `summary`, `prettysummary`,
    and other display methods.
  - For doctests that only need to verify code runs, end with a statement that produces simple output,
    such as `typeof(result)` or accessing a field that returns a simple value.


### Naming Conventions
- **Files**: snake_case (e.g., `atmosphere_model.jl`, `update_atmosphere_model_state.jl`)
- **Types**: PascalCase (e.g., `AtmosphereModel`, `AnelasticDynamics`, `MoistAirBuoyancy`)
- **Functions**: snake_case (e.g., `update_atmosphere_model!`, `compute_pressure!`)
- **Kernels**: "Kernels" (functions prefixed with `@kernel`) may be prefixed with an underscore (e.g., `_kernel_function`)
- **Variables**: Use _either_ an English long name, or mathematical notation with readable unicode. Variable names should be taken from `docs/src/appendix/notation.md` in the docs. If a new variable is created (or if one doesn't exist), it should be added to the table in notation.md
- **Avoid abbreviations**: Use full words instead of abbreviations. For example, use `latitude` instead of `lat`, `longitude` instead of `lon`, `temperature` instead of `temp`. This improves code readability and self-documentation.

### Breeze Module Structure
```
src/
├── Breeze.jl                  # Main module, exports
├── Thermodynamics/            # Thermodynamic states & equations
├── AtmosphereModels/          # Core atmosphere model logic
├── Microphysics/              # Cloud microphysics
├── TurbulenceClosures/        # TurbulenceClosures, including those ported from Oceananigans
├── Advection.jl               # Advection operators for anelastic models
└── MoistAirBuoyancies.jl      # A legacy buoyancy formulation for usage with Oceananigans.NonhydrostaticModel
```

These are also planned:
- an extension in `ext/` for `RRTMGP.jl`
- modules that correspond to Oceananigans features:
    * LagrangianParticleTracking/

### Breeze formulations

Breeze uses "formulations" to express different equation sets that encode conservation of mass, momentum, and energy.
Currently Breeze always uses `AnelasticDynamics` in conservation form. In conservation form, all prognostic
variables are "densities". There are currently two anelastic thermodynamic formulations:
  - `LiquidIcePotentialTemperatureThermodynamics` with prognostic potential temperature density `ρθ`.
  - `StaticEnergyThermodynamics` with prognostic static energy density `ρe`.
Eventually there will also be a fully compressible formulation with prognostic total energy density.
We may also implement `EntropyThermodynamics` which prognostics entropy density `ρη`.

### Microphysics implementation guidelines

Breeze has a microphysics interface in `src/AtmosphereModels/microphysics_interface.jl` that defines
the functions that microphysics schemes must implement. Key functions include:

- `maybe_adjust_thermodynamic_state`: Adjusts the thermodynamic state based on the microphysics scheme.
  - For **saturation adjustment** schemes (equilibrium cloud formation): this function performs iterative
    saturation adjustment to partition moisture between vapor and condensate.
  - For **non-equilibrium** schemes (prognostic cloud condensate): this function should be **trivial**
    (just return the input state unchanged). Non-equilibrium schemes have fully prognostic cloud
    liquid/ice, so there is no adjustment to perform. The moisture partition is already determined
    by the prognostic fields.
- `microphysical_tendency`: Computes tendencies for prognostic microphysical variables.
- `moisture_fractions`: Computes moisture mass fractions from prognostic fields.
- `update_microphysical_fields!`: Updates diagnostic microphysical fields after state update.

## Testing Guidelines

### Running Tests
```julia
# All tests
Pkg.test("Breeze")

# Run a specific test file by passing in the first few characters of the file:
Pkg.test("Breeze"; test_args=`atmosphere_model_unit`)

# CPU-only (disable GPU)
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
Pkg.test("Breeze")
```

* GPU tests may fail with "dynamic invocation error". In that case, the tests should be run on CPU.
  If the error goes away, the problem is GPU-specific, and often a type-inference issue.

### Writing Tests
- Place tests in `test/` directory
- Use `default_arch` for architecture selection
- Toggle the floating point type using `Oceananigans.defaults.FloatType`.
- Name test files descriptively (snake_case)
- Include both unit tests and integration tests
- Test numerical accuracy where analytical solutions exist

### Quality Assurance
- Ensure doctests pass
- Run `quality_assurance.jl` to check code standards
- Use Aqua.jl for package quality checks

### Fixing bugs
- Subtle bugs often occur when a method is not imported, especially in an extension
- Sometimes user scripts are written expecting names to be exported, when they are not. In that case
  consider exporting the name automatically (ie implement the user interface that the user expects) rather
  than changing the user script
- **Extending getproperty:** never do this to fix a bug associated with accessing an undefined property.
  This bug should be fixed on the _caller_ side, so that an undefined name is not accessed.
  A common source of this bug is when a property name is changed (for example, to make it clearer).
  In this case the calling function merely needs to be updated.
- **"Type is not callable" errors**: Variable naming is hard. Sometimes, variable names conflict. A common issue is when the name of a _field_ (the result
  of a computation) overlaps with the name of a function in the same scope/context. This can lead to errors like "Fields cannot be called".
  The solution to this problem is to change the name of the field to be more verbose, or use a qualified name for the function
  that references the module it is defined in to disambiguate the names (if possible).
- **Connecting dots:** If a test fails immediately after a change was made, go back and re-examine whether that change
  made sense. Sometimes, a simple fix that gets code to _run_ (ie fixing a test _error_) will end up making it _incorrect_ (which hopefully will be caught as a test _failure_). In this case the original edit should be revisited: a more nuanced solution to the test error may be required.

## Common Development Tasks

### Adding New Physics
1. Create module in appropriate subdirectory
2. Define types/structs with docstrings
3. Implement kernel functions (GPU-compatible)
4. Add unit tests
5. If the user interface is changed, update main module exports in `src/Breeze.jl`
6. Add a script that can be used to validate the scientific content of the feature (e.g., in `examples/` or `validation/` when available).

### Modifying AtmosphereModel
- Core logic in `src/AtmosphereModels/atmosphere_model.jl`
- State updates in `update_atmosphere_model_state.jl`
- Pressure solver in `anelastic_pressure_solver.jl`
- Always consider anelastic formulation constraints

## Documentation

### Building Docs Locally
```sh
julia --project=docs/ docs/make.jl
```

### Viewing Docs
```julia
using LiveServer
serve(dir="docs/build")
```

### Testing docs
- Consider manually running `@example` blocks, rather than building the whole
  documentation to find errors.
- Unless explicitly asked, do not write `for` loops in docs blocks. Use built-in functions
  (which will launch kernels under the hood) instead.
- Be conservative about developing examples and tutorials. Do not write extensive example code unless asked.
  Instead, produce skeletons or outlines with minimum viable code.
- **Debugging literated examples**: When a specific example fails during doc builds, comment out
  all other examples in `docs/make.jl` except the failing one to isolate the error. This speeds up
  iteration dramatically since you only build one example at a time.
- **Testing documentation pages efficiently**: When testing changes to documentation pages (like
  `thermodynamics.md`), comment out ALL examples in the `examples` array in `docs/make.jl` to skip
  the slow literation step. This allows rapid iteration on `@example` blocks in the documentation
  markdown files.
- **Literate.jl comment syntax**: In literated examples, lines starting with `# ` (hash + space)
  at column 1 are converted to markdown. Comments inside functions that start with `#` at the
  beginning of a line will prematurely end code blocks. Either remove such comments or use `##`
  to keep them as code comments.

## Important Files to Know

### Core Implementation
- `src/Breeze.jl` - Main module, all exports
- `src/AtmosphereModels/atmosphere_model.jl` - Central model definition
- `src/AtmosphereModels/anelastic_formulation.jl` - Anelastic equations
- `src/Thermodynamics/` - Thermodynamic relations

### Configuration
- `Project.toml` - Package dependencies and compat bounds
- `test/runtests.jl` - Test configuration and architecture selection

### Examples
- `examples/thermal_bubble.jl` - Classic dry dynamics test
- `examples/free_convection.jl` - Moist convection
- `examples/anelastic_bomex.jl` - BOMEX intercomparison case (anelastic)
- `examples/boussinesq_bomex.jl` - BOMEX intercomparison case (Boussinesq)
- `examples/mountain_wave.jl` - Mountain wave simulation
- `examples/cloudy_kelvin_helmholtz.jl` - Cloudy Kelvin-Helmholtz instability
- Many more examples available in the `examples/` directory

## Physics Domain Knowledge

### Atmospheric Dynamics
- Anelastic approximation filters acoustic waves
- Moist thermodynamics via saturation adjustment
- Reference state defines background stratification
- Hydrostatic pressure computed diagnostically

### Numerical Methods
- Finite volume on structured grids (Arakawa C-grid)
- Staggered grid locations: velocities at cell faces, tracers at cell centers
- Take care of staggered grid location when writing operators or designing diagnostics.
- Favor WENO advection schemes.
- Pressure Poisson solver for anelastic divergence constraint
- Time stepping: RungeKutta3 (default), Adams-Bashforth, Quasi-Adams-Bashforth

## Implementing Validation Cases / Reproducing Paper Results

When implementing a simulation from a published paper:

### 1. Parameter Extraction
- **Read the paper carefully** and extract ALL parameters: domain size, resolution, physical constants,
  boundary conditions, initial conditions, forcing, closure parameters
- Look for parameter tables (often "Table 1" or similar)
- Check figure captions for additional details
- Note the coordinate system and conventions used

### 2. Geometry Verification (BEFORE running long simulations)
- **Always visualize the grid/domain geometry first**
- Check that:
  - Domain extents match the paper
  - Topography/immersed boundaries are correct
  - Coordinate orientations match (which direction is "downslope"?)
- Compare your geometry plot to figures in the paper

### 3. Initial Condition Verification
- After setting initial conditions, check:
  - `minimum(field)` and `maximum(field)` make physical sense
  - Spatial distribution looks correct (visualize if needed)
  - Dense water is where it should be, stratification is correct, etc.

### 4. Short Test Runs
Before running a long simulation:
- Run for a few timesteps on CPU at low resolution
- Verify:
  - No NaNs appear (check `maximum(abs, u)` etc.)
  - Flow is developing as expected (velocities increasing from zero)
  - Output files contain meaningful data
- Then test on GPU to catch GPU-specific issues

### 5. Progressive Validation
- Run a short simulation (e.g., 1 hour sim time) and visualize
- Check that the physics looks right:
  - Dense water flowing in the correct direction?
  - Velocities reasonable magnitude?
  - Mixing/entrainment happening where expected?
- Compare to early-time figures in the paper if available

### 6. Comparison to Paper Figures
- Create visualizations that match the paper's figure format
- Use the same colormaps, axis ranges, and time snapshots if possible
- Quantitative comparison: compute the same diagnostics as the paper

### 7. Common Issues
- **NaN blowups**: Usually from timestep too large, unstable initial conditions,
  or if-else statements on GPU (use `ifelse` instead)
- **Nothing happening**: Check that buoyancy anomaly has the right sign,
  that initial conditions are actually applied, that forcing is active
- **Wrong direction of flow**: Check coordinate conventions (is y increasing
  upslope or downslope?)
- **GPU issues**: Avoid branching, ensure type stability, use `randn()` carefully

## Common Pitfalls

1. **Type Instability**: Especially in kernel functions - ruins GPU performance
2. **Overconstraining types**: Julia compiler can infer types. Type annotations should be used primarily for _multiple dispatch_, not for documentation.
3. **Forgetting Explicit Imports**: Tests will fail - add to using statements
4. **Using plain `julia` blocks in docstrings**: NEVER do this. ALWAYS use `jldoctest` blocks so examples are tested and verified to work. Plain `julia` blocks are not tested and will become stale.

## Debugging Physics Simulations

### Thermodynamic Variable Discipline

Breeze.jl uses multiple thermodynamic variables that are related but NOT interchangeable:

| Variable | Meaning | Relationship |
|----------|---------|--------------|
| `T` | Temperature (K) | Absolute temperature |
| `θ` | Potential temperature (K) | `θ = T / Π` where `Π = (p/p₀)^κ` |
| `ρe` | Density × total energy (J/m³) | Includes kinetic + internal energy |
| `ρθ` | Density × potential temperature (kg·K/m³) | Prognostic in `LiquidIcePotentialTemperatureFormulation` |

**Before applying forcing or boundary conditions:**

1. **Check the paper**: Does it specify forcing in terms of T, θ, or energy?
2. **Check working examples**: Which field do BOMEX/RICO apply similar forcing to?
3. **Check the Breeze source**: Which variable is prognostic for your formulation?
4. **Verify units**: K/s vs K/day, W/m² vs K·m/s — get conversions right

**Common mistakes:**
- Applying a temperature tendency directly to potential temperature
- Confusing `ρe` (energy) with `ρθ` (potential temperature)
- Not accounting for the Exner function when converting T ↔ θ

### When a Stable Simulation Becomes Unstable

If the model was running stably and then becomes unstable after your changes:

1. **STOP** — Do not keep adding "fixes"
2. **Identify the last working state** — Use `git log` and `git diff`
3. **Revert to working state** — `git checkout` the stable version
4. **Make ONE change at a time** — Test after each change
5. **Find the breaking change** — The instability was introduced by something you changed

The instability is NOT a pre-existing bug if the code was stable before your changes.

### Mandatory Checks Before Modifying Physics Code

- [ ] Have I read the relevant working examples (BOMEX, RICO, prescribed_SST)?
- [ ] Have I identified which field the example applies similar physics to?
- [ ] Have I verified my implementation matches the paper's specification?
- [ ] Am I making ONE change only?
- [ ] Have I committed or stashed the current working state?

If any answer is "no", complete that step before proceeding.

## Git Workflow
- Follow ColPrac (Collaborative Practices for Community Packages)
- Create feature branches for new work
- Write descriptive commit messages
- Update tests and documentation with code changes
- Check CI passes before merging

## Code Formatting and Whitespace

**PRs will fail CI unless trailing whitespace and trailing blank lines are cleared.**
Before committing, clean up whitespace in all `.jl`, `.md`, and `.sh` files.

The cleanup must:
1. Remove trailing whitespace from each line
2. Remove trailing blank lines at end of file
3. Ensure file ends with exactly one newline

### Shell-based cleanup (recommended for agents)

```bash
# Combined cleanup: trailing whitespace, trailing blank lines, ensure final newline
for file in $(find /path/to/Breeze -type f \( -name "*.jl" -o -name "*.md" -o -name "*.sh" \) ! -path "*/.git/*"); do
  # Remove trailing whitespace from each line
  sed -i '' 's/[[:space:]]*$//' "$file"
  # Remove trailing blank lines and ensure exactly one final newline
  # This uses awk to skip trailing empty lines and adds one newline at end
  awk 'NF {p=1} p' "$file" | awk '{print}' > "$file.tmp" && mv "$file.tmp" "$file"
  # Ensure file ends with newline (in case awk produced empty output)
  [ -s "$file" ] && [ "$(tail -c1 "$file" | wc -l)" -eq 0 ] && echo >> "$file"
done
```

### Emacs Lisp cleanup

```elisp
(dolist (file (directory-files-recursively "/path/to/Breeze" "\\.\\(jl\\|md\\|sh\\)$"))
  (when (file-regular-p file)
    (with-temp-buffer
      (insert-file-contents file)
      ;; Force LF line ending
      (set-buffer-file-coding-system 'unix)
      ;; Add final newline in case it's missing (will be cleaned if extra)
      (goto-char (point-max))
      (insert "\n")
      ;; Replace non-breaking spaces with regular spaces
      (save-excursion
        (goto-char (point-min))
        (while (search-forward " " nil t)
          (replace-match " " nil t)))
      ;; Convert tabs to spaces
      (untabify (point-min) (point-max))
      ;; Remove trailing whitespace (includes trailing blank lines)
      (delete-trailing-whitespace)
      (write-region (point-min) (point-max) file))))
```

## Helpful Resources
- Oceananigans docs: https://clima.github.io/OceananigansDocumentation/stable/
- Breeze docs: https://numericalearth.github.io/BreezeDocumentation/dev/
- Discussions: https://github.com/NumericalEarth/Breeze.jl/discussions
- KernelAbstractions.jl: https://github.com/JuliaGPU/KernelAbstractions.jl
- Reactant.jl: https://github.com/EnzymeAD/Reactant.jl
- Reactant.jl docs: https://enzymead.github.io/Reactant.jl/stable/
- Enzyme.jl: https://github.com/EnzymeAD/Enzyme.jl
- Enzyme.jl docs: https://enzyme.mit.edu/julia/dev
- YASGuide: https://github.com/jrevels/YASGuide
- ColPrac: https://github.com/SciML/ColPrac
- MCPRepl.jl: https://github.com/kahliburke/MCPRepl.jl

## When Unsure
1. **Study working examples first** — BOMEX, RICO, and other examples in `examples/` are stable and correct. Compare your code to them before making changes.
2. Check existing examples in `examples/` directory
3. Look at similar implementations in Oceananigans.jl
4. Review tests for usage patterns
5. Ask in GitHub discussions
6. Check documentation in `docs/src/`

## AI Assistant Behavior
- Prioritize type stability and GPU compatibility
- Follow established patterns in existing code
- Add tests for new functionality
- Update exports in main module when adding public API
- Consider both CPU and GPU architectures
- Reference physics equations in comments when implementing dynamics
- Maintain consistency with Oceananigans.jl patterns

## Interactive Julia REPL for AI Agents (MCPRepl.jl)

[MCPRepl.jl](https://github.com/kahliburke/MCPRepl.jl) exposes a Julia REPL via the Model Context Protocol (MCP),
allowing AI agents to execute Julia code, run tests, and iterate quickly during development.

### Installation

If MCPRepl.jl is not already installed, add it to your global Julia environment:

```julia
using Pkg
Pkg.activate()  # Activate global environment
Pkg.add(url="https://github.com/kahliburke/MCPRepl.jl")
```

Then run the security setup (one-time):

```julia
using MCPRepl
MCPRepl.quick_setup(:lax)  # For local development (localhost only, no API key)
```

### Starting the MCP Server

Before the AI agent can use the REPL, start the server in Julia:

```julia
using MCPRepl
MCPRepl.start_proxy(port=3000)  # Recommended: persistent proxy with dashboard
# OR
MCPRepl.start!(port=3000)       # Direct REPL backend
```

The dashboard is available at `http://localhost:3000/dashboard` when using the proxy.

### Cursor Configuration

Create `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "julia-repl": {
      "url": "http://localhost:3000",
      "transport": "http",
      "headers": {
        "X-MCPRepl-Target": "Breeze.jl"
      }
    }
  }
}
```

After creating this file, reload Cursor (Cmd+Shift+P → "Reload Window").

### Speeding Up Development with Revise.jl

For rapid iteration, use Revise.jl alongside MCPRepl. This allows code changes to be
reflected immediately without restarting Julia:

```julia
using Revise
using MCPRepl
using Oceananigans
using Breeze

MCPRepl.start_proxy(port=3000)
```

With this setup:
1. The AI agent can execute code via the REPL
2. Source code edits are automatically picked up by Revise
3. No need to restart Julia or re-import packages after editing source files
4. Tests can be run interactively with immediate feedback

### Available MCP Tools

Once connected, the AI agent has access to:
- **`julia_eval`** — Execute Julia code in the REPL
- **`lsp_goto_definition`** — Navigate to symbol definitions
- **`lsp_find_references`** — Find all usages of a symbol
- **`lsp_rename`** — Rename symbols across the codebase
- **`lsp_document_symbols`** — Get file structure/outline
- **`lsp_code_actions`** — Get available quick fixes

### Workflow Example

A typical development workflow:

1. Start Julia with Revise and MCPRepl
2. AI agent makes code changes via file editing
3. Revise automatically loads the changes
4. AI agent tests changes via MCPRepl without restarting
5. Iterate rapidly until the feature/fix is complete

This eliminates the slow compile-restart cycle and enables interactive debugging.

## Roadmap

Features that are planned, which should be considered when implementing anything:

- Fully compressible formulation
    - explicit time-stepping
    - acoustic substepping
- A way to use total energy with the anelastic model
- open boundaries following the Oceananigans.NonhydrostaticModel implementation
- microphysics schemes from CloudMicrophysics
- superdroplet microphysics schemes (reference tbd, but will use LagrangianParticleTracking)
- MPI / distributed GPU support
- Many, many more canonical LES validation cases
- Terrain-following coordinate (may require upstream development in Oceananigans + using MutableVerticalCoordinate)
- Cut-cells ImmersedBoundaryGrid implementation
