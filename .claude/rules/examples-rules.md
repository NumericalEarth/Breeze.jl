---
paths:
  - examples/**/*.jl
---

# Examples Rules

## Writing Examples

- Explain at the top what the simulation does
- Let code "speak for itself" — Literate style. Lighthearted, engaging prose.
- Follow existing example style, not source code style
- New examples should add new physics/value, not copy existing ones
- Don't "over import". Use names exported by `using Oceananigans` and `using Breeze`.
  If needed names aren't exported, consider exporting them.
- Examples use `examples/Project.toml`. Add example-specific packages there, not main `Project.toml`.

## Literate.jl Comment Conventions

- Single `#` comments at column 1 become markdown blocks in generated documentation
- Double `##` comments remain as code comments within code blocks
- Use `##` for inline code comments that should stay with the code
- Use single `#` only for narrative text that should render as markdown

## Plotting

**CRITICAL**: NEVER use `interior(field, ...)`. Makie plots `Field` objects directly.
Use `view(field, i, j, k)` to window fields. Works with `@lift` for animations too.

- Always add axis labels and colorbars
- **Color palette**: `:dodgerblue` (vapor), `:lime` (cloud), `:orangered` (rain), `:magenta` (temperature)
- **Do not convert units**: Keep consistent with source code. Exception: spatial coordinates to km for axis labels.

## Conventions

- Initial condition functions act _pointwise_ — no broadcasting inside them
- Use `Oceananigans.defaults.FloatType = FT` for precision, not manual `FT(1)` conversions
- Use integers for integer values. Rely on "autotupling": `tracers = :c` not `tracers = (:c,)`
- Invoke `set!` ideally once (it calls `update_state!` internally)
- Call models `model` and simulations `simulation`
- Use suffix `ts` for time series, `n` for time-indexed fields
- **Testing examples**: Reduce resolution and switch to CPU for debugging. **Always revert** before committing.
