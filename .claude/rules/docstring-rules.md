---
paths:
  - src/**/*.jl
---

# Docstring Rules

## Use DocStringExtensions.jl

- Use `$(TYPEDSIGNATURES)` — never write explicit signatures
- **Citations**: Use inline `[Author (year)](@cite Key)` syntax woven into prose

## CRITICAL: Always use `jldoctest`, NEVER plain `julia` blocks

Plain code blocks (`` ```julia ``) are NOT tested and can become stale or incorrect.
Doctests (`` ```jldoctest ``) are automatically tested and verified to work.

### Example:

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

## Doctest Best Practices

- Always include expected output after `# output`
- Use simple, verifiable output (e.g., `typeof(result)`, accessing a field)
- Doctests should exercise `Base.show` to verify objects display correctly
- Keep doctests minimal but complete enough to verify the feature works
- **Do NOT use boolean comparisons as the final line** — invoke a `show` method instead
- For run-only verification, end with `typeof(result)` or a simple field access
