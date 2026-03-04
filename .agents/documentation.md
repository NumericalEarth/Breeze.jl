# Documentation

## Building Docs Locally

```sh
julia --project=docs/ docs/make.jl        # Build
julia -e 'using LiveServer; serve(dir="docs/build")'  # View
```

## Fast Local Docs Builds

When testing documentation changes locally, temporarily modify `docs/make.jl`:

1. **Comment out Literate examples** — these take the longest to run
2. **Comment out GPU-requiring pages** if not on GPU
3. Optional speedups: `doctest = false`, `linkcheck = false`

**Important**: Remember to revert these changes before committing!

## Tips

- Manually run `@example` blocks rather than full doc builds to find errors
- Don't write `for` loops in docs blocks unless asked. Use built-in functions.
- **Debugging literated examples**: Comment out all other examples in `docs/make.jl` to isolate failures
- **Testing doc pages**: Comment out ALL examples to skip literation; iterate on `@example` blocks

## Documentation Style

- Use unicode in math (e.g., ``θᵉ`` not ``\theta^e``); Documenter converts to LaTeX
- Always add `@ref` cross-references for Breeze functions
- Link to Oceananigans docs for external functions
- **Citations**: Use inline `[Author (year)](@cite Key)` syntax woven into prose.
  Avoid separate "References" sections with bare `[Key](@cite)`.

## Docstring Examples (CRITICAL)

**NEVER use plain `julia` code blocks in docstrings. ALWAYS use `jldoctest` blocks.**

### Correct — use `jldoctest`:

~~~~
"""
$(TYPEDSIGNATURES)

Example:

```jldoctest
using Oceananigans, Breeze
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
typeof(grid)

# output
RectilinearGrid{Float64, Periodic, Periodic, Bounded, Nothing, Nothing, Nothing, Nothing}
```
"""
~~~~

### Key doctest requirements

- Always include expected output after `# output`
- Use simple, verifiable output (e.g., `typeof(result)`, accessing a field)
- Exercise `Base.show` to verify objects display correctly
- Keep doctests minimal but complete enough to verify the feature works
- **Do NOT use boolean comparisons** as the final line — invoke `show` instead

## Writing Examples

- Explain at the top what the simulation does
- Let code "speak for itself" — Literate style. Lighthearted, engaging prose.
- Follow existing example style, not source code style
- New examples should add new physics/value, not copy existing ones
- Prefer exported names. If too many internal names needed, export them or create a new abstraction.
- Examples use `examples/Project.toml`. Add example-specific packages there, not main `Project.toml`.
- Call models `model` and simulations `simulation`

### Literate.jl Conventions

- Single `#` comments at column 1 become markdown blocks in generated documentation
- Double `##` comments remain as code comments within code blocks
- Use `##` for inline code comments that should stay with the code
- Use single `#` only for narrative text that should render as markdown

### Plotting Fields

**CRITICAL**: NEVER use `interior(field, ...)`. Makie plots `Field` objects directly.
Use `view(field, i, j, k)` to window fields. Works with `@lift` for animations too:

```julia
# WRONG:
data = @lift interior(field_ts[$n], :, 1, :)
heatmap!(ax, x, z, data, ...)
# CORRECT:
field_n = @lift field_ts[$n]
heatmap!(ax, field_n, ...)
```

### Example Conventions

- Initial condition functions act _pointwise_ — no broadcasting inside them
- **Do not convert units**: Keep units consistent with source code. Exception: spatial coordinates to km for axis labels.
- Use concise names and unicode consistent with source code and `notation.md`
- Always add axis labels and colorbars
- Use `xnode`/`ynode`/`znode` for `discrete_form=true` forcing/BCs. Never access grid metrics manually.
- Use `Oceananigans.defaults.FloatType = FT` for precision, not manual `FT(1)` conversions
- Use integers for integer values. Rely on "autotupling": `tracers = :c` not `tracers = (:c,)`
- Invoke `set!` ideally once (it calls `update_state!` internally)
- Use suffix `ts` for time series, `n` for time-indexed fields
- **Color palette**: `:dodgerblue` (vapor), `:lime` (cloud), `:orangered` (rain), `:magenta` (temperature)
