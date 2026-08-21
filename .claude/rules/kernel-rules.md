---
paths:
  - src/**/*.jl
---

# Kernel Function Rules

GPU-compatible kernel functions are critical for Breeze performance.

## Requirements

- Use KernelAbstractions.jl syntax: `@kernel`, `@index`
- Keep kernels **type-stable** and **allocation-free**
- Use `ifelse` instead of short-circuiting `if`/`else` or ternary `?`/`:`
- No error messages inside kernels
- Models **never** go inside kernels
- Mark functions called inside kernels with `@inline`
- **Never use loops outside kernels**: Replace `for` loops over grid points with `launch!` kernels
- **Use literal zeros**: `max(0, a)` not `max(zero(FT), a)`. Julia handles type promotion.

## Type Stability

- All structs must be concretely typed. **Never use `Any` as a type parameter or field type.**
- Type instability in kernel functions ruins GPU performance
- Use type annotations for **multiple dispatch**, not documentation

## Memory Efficiency

- Favor inline computations over allocating temporary memory
- Minimize memory allocation overall
- If an implementation is awkward, suggest an upstream Oceananigans feature instead

## Staggered Grid & Indexing

- Velocities live at cell faces, tracers at cell centers (Arakawa C-grid)
- Take care of staggered grid location when writing operators or designing diagnostics

## Closure Captures

- **Never reassign a variable captured by a closure that reaches a GPU kernel** (masks,
  forcings, boundary conditions). Reassignment turns the capture into a `Core.Box`, the
  closure stops being `isbits`, and the kernel launch fails — often only at run time on GPU.
- Compute with single-assignment names instead: `λ₁ˡ, λ₂ˡ = x_domain(grid)` then
  `λ₁ = all_reduce(min, λ₁ˡ, arch)` — never `λ₁ = ...` followed by `λ₁ = f(λ₁)`.
- When in doubt, check `isbits(closure)` before launching.

## Dispatch Over Branching

- No value-level guards (`haskey`, `isnothing` chains) inside small GPU helper functions;
  make the decision once outside the kernel — fetch with `get(container, key, nothing)` —
  and provide a `::Nothing` method for the missing case.
- A `MethodError` on an unhandled combination is the intended "not implemented" signal;
  don't paper over it with runtime branches.
