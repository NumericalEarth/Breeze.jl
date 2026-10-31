# Breeze.jl

Atmosphere model built on Oceananigans (grids, fields, solvers, advection), with extensions for
CloudMicrophysics and RRTMGP. It runs on CPUs and GPUs through KernelAbstractions, so code that
passes on a CPU can still fail on a GPU; `.claude/rules/kernel-rules.md` covers why.

## Commands

```sh
# Run one test file on CPU (test names are file names under test/, without .jl)
CUDA_VISIBLE_DEVICES=-1 julia --project -e 'using Pkg; Pkg.test("Breeze"; test_args=`atmosphere_model_construction`)'

# Explicit imports and Aqua checks (run after any change to src/ or ext/)
CUDA_VISIBLE_DEVICES=-1 julia --project -e 'using Pkg; Pkg.test("Breeze"; test_args=`quality_assurance`)'

# Doctests
CUDA_VISIBLE_DEVICES=-1 julia --project -e 'using Pkg; Pkg.test("Breeze"; test_args=`doctests`)'

# Trailing whitespace and blank lines at end of file (CI also requires exactly one final newline)
git diff --check origin/main
```

The `reactant/` tests run only with `--check-bounds=auto` on Julia older than 1.14; `test/runtests.jl`
drops them otherwise. Examples have their own environment, `examples/Project.toml`.

## Formulations and how inputs are keyed

Dynamics (`dynamics` keyword of `AtmosphereModel`) and thermodynamic formulation (`formulation`
keyword) are chosen independently. All prognostics are densities.

- Dynamics: `AnelasticDynamics`, `CompressibleDynamics` (acoustic substepping), `PrescribedDynamics`
  (kinematic driver).
- Formulations: `LiquidIcePotentialTemperatureFormulation` (`:LiquidIcePotentialTemperature`, the
  default; prognostic `ρθ`) and `StaticEnergyFormulation` (`:StaticEnergy`; prognostic `ρs`).

Energy and water inputs in `boundary_conditions` and `forcing` are keyed by what they are, not by the
prognostic variable, because that variable depends on the formulation and the microphysics:

- `ρE` (specific alias `E` for forcings) is total energy. Breeze converts it for whichever
  thermodynamic variable the formulation evolves.
- `ρqᵗ` (specific alias `qᵗ`) is total water, re-keyed unconverted onto whichever moisture density
  the microphysics evolves.
- `ρs`, `ρqᵛ`, and `ρqᵉ` are keys only where they are prognostic. Supplying both an interface key
  and its target is an error, and unrecognized keys raise an `ArgumentError`.

A common physics bug is applying a temperature tendency from a paper directly to `ρθ`, which needs
the Exner function. Read `.agents/physics-debugging.md` before implementing forcing or microphysics
from a reference.

## Before you change these, ask

- **`[deps]` and `[weakdeps]` in `Project.toml`**. They change load time, CI, and every downstream
  package. Touch `[compat]` only when asked.
- **`Artifacts.toml`** (the P3 lookup tables) and **expected values or tolerances in tests**. A
  numerical test that starts failing is evidence of a behavior change; find the cause instead of
  updating the number.
- **Exported names and keyword arguments of public constructors**. NumericalEarth, the examples,
  and user scripts depend on them.

## Verifying your work

- Read the current definition of anything you call (`@which`, `methods`, or the source),
  including Breeze's and Oceananigans' own APIs. They change quickly and remembered signatures go
  stale.
- A test that fails on your branch is yours until you reproduce the same failure on `main`.
- Report results by quoting the test summary line. An exit code alone is not a pass.
- If a fix makes a failing test run but you cannot explain why it was failing, the fix is probably
  wrong. Revisit the change that broke it.
- A simulation that was stable before your change and is unstable after it was broken by your
  change. Revert and reapply one piece at a time rather than adding a fix on top.
- GPU "dynamic invocation error": rerun on CPU. If it passes there, the cause is almost always a
  type instability that the CPU tolerates.

## Design

- **Materialization pattern**: a user-facing constructor builds a skeleton struct with placeholder
  type parameters (such as `Nothing`); `materialize_*` builds the fully typed version once the grid
  and model are known.
- Structs are concretely typed; never use `Any` as a type parameter or field type. For mutable
  state inside an immutable struct, use a `mutable struct` as the field type.
- Extend functions in source code, not in examples. If an example needs internals, export them or
  add an abstraction.
- When something would be better in Oceananigans, add a detailed TODO note rather than a local
  workaround.
- Avoid duplicated code beyond trivial one-liners.

## Conventions that are not visible from the code

- Source code uses explicit imports, checked by `quality_assurance`. Extend functions with
  `Module.function_name(...) = ...`, not `import`. Exports go at the top of module files. Import
  Oceananigans/Breeze names first, then external packages; internal imports use absolute paths.
  Examples and docs use `using Oceananigans` and `using Breeze`.
- Docstrings use `$(TYPEDSIGNATURES)` (never a hand-written signature) and `jldoctest` examples;
  details are in `.claude/rules/docstring-rules.md`.
- Variable names are full English (`latitude`, not `lat`) or Unicode math from
  `docs/src/appendix/notation.md`, never a mix in one expression. Add new symbols to that table.
- A leading `_` is reserved for `@kernel` functions. Helpers, including "private" ones, get plain
  snake_case names; this codebase does not follow the Python convention.
- Keyword arguments: no spaces inline, `f(x=1)`; single spaces when split over lines,
  `f(a = 1, b = 2)`.
- Never extend `getproperty` to make an undefined-property error go away; fix the caller.
- A "type is not callable" error usually means a local variable shadows a function name.
- Keep a PR to one concern.

## Where to look

Rules in `.claude/rules/` load automatically in Claude Code when you edit matching files. Other
agents should read the one that matches the task:

| Task | Read |
|------|------|
| Writing or editing kernels, operators, or anything in `src/` or `ext/` | `.claude/rules/kernel-rules.md` |
| Docstrings | `.claude/rules/docstring-rules.md` |
| Tests | `.claude/rules/testing-rules.md` |
| Docs pages | `.claude/rules/docs-rules.md` |
| Examples | `.claude/rules/examples-rules.md` |
| Forcing, microphysics, or anything converting between thermodynamic variables | `.agents/physics-debugging.md` |
| Tropical cyclone genesis cases | `.agents/validation.md` |

Skills (`.claude/skills/`): `/run-tests`, `/build-docs`, `/new-simulation`, `/babysit-ci`.
