---
paths:
  - src/**/*.jl
  - ext/**/*.jl
---

# Docstring Rules

- Use `$(TYPEDSIGNATURES_WITH_DEFAULTS)`, imported with `using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS`;
  never write the signature by hand. It is DocStringExtensions' `TYPEDSIGNATURES` with the default
  values of positional and keyword arguments shown, and it stays in sync with the method.
- Use `$(TYPEDSIGNATURES_WITH_DEFAULTS_NO_RETURN)` (also from `Breeze.Utils`) when the inferred
  return type is too long to be worth printing, such as a `Union` of deeply parametrized types.
- Code examples in docstrings are `jldoctest` blocks, not `julia` blocks. Doctests run in the
  `doctests` test; plain `julia` blocks are never executed and go stale silently.
- End a doctest with an expression whose `show` output is worth reading, and put that output after
  `# output`. This tests the feature and its `show` method at once. A final line such as
  `x ≈ 1.0` or `obj isa Type` prints `true` and tests almost nothing. For run-only checks, end
  with `typeof(result)` or a simple field access.
- Cite with inline `[Author (year)](@cite Key)` woven into the prose, not a separate "References"
  section of bare `[Key](@cite)` entries.
- Write math in Unicode (`θ`, `ρ`, `Π`), not LaTeX; docstrings are read in the REPL, where LaTeX
  does not render.

~~~~
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the liquid-ice potential temperature of `model`.

```jldoctest
using Oceananigans, Breeze

<a minimal call>

# output
<the printed result>
```
"""
~~~~
