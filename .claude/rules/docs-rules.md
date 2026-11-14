---
paths:
  - docs/**/*
---

# Documentation Rules

Building the docs, including a fast local build, is covered by the `/build-docs` skill
(`.claude/skills/build-docs/SKILL.md`).

## Style

- Use Unicode in math (``θᵉ``, not ``\theta^e``); Documenter converts it to LaTeX.
- Add `@ref` cross-references for Breeze functions and link to the Oceananigans docs for external
  ones.
- Cite with inline `[Author (year)](@cite Key)` woven into the prose.
- In example code, rely on `using Oceananigans` and `using Breeze`; explicitly importing an
  exported name hides what users actually need to type.
- Don't write `for` loops in docs blocks unless asked; use built-in functions.
- To find an error in a page, run its `@example` blocks directly instead of building the docs.

Docstring conventions are in `.claude/rules/docstring-rules.md`.
