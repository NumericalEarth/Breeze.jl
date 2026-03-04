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
