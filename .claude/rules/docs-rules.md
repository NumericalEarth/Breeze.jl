---
paths:
  - docs/**/*
---

# Documentation Rules

## Building Docs

```sh
julia --project=docs/ docs/make.jl
```

## Fast Local Builds

For local testing, temporarily modify `docs/make.jl`:
1. Comment out Literate examples
2. Comment out GPU-requiring pages
3. Optional: `doctest = false`, `linkcheck = false`

**Remember to revert these changes before committing!**

## Viewing Docs

```julia
using LiveServer
serve(dir="docs/build")
```

## Style

- Use unicode in math (e.g., ``θᵉ`` not ``\theta^e``); Documenter converts to LaTeX
- Always add `@ref` cross-references for Breeze functions
- Link to Oceananigans docs for external functions
- In example code, NEVER explicitly import names already exported by `using Breeze`

## Docstrings

- ALWAYS use `jldoctest` blocks, NEVER plain `julia` blocks
- See `.claude/rules/docstring-rules.md` for full details
