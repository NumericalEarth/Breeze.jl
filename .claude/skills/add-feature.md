---
name: add-feature
description: Checklist for adding new physics or features to Breeze
user_invocable: true
---

# Add Feature

Follow this checklist when adding new physics or features to Breeze.

## Checklist

1. **Create module** in the appropriate subdirectory under `src/`
2. **Define types/structs** with docstrings (use `jldoctest` blocks, never plain `julia`)
   - All structs concretely typed. Never use `Any` as a type parameter.
   - Use materialization pattern: skeleton struct → `materialize_*` for fully-typed version
3. **Implement kernel functions** — must be GPU-compatible:
   - Use `@kernel` and `@index` from KernelAbstractions.jl
   - Keep type-stable and allocation-free
   - Use `ifelse` instead of short-circuiting `if`/`else`
   - Mark inner functions with `@inline`
   - No loops over grid points outside kernels — use `launch!`
4. **Add unit tests** in `test/` (ParallelTestRunner autodiscovery will find them)
5. **Update exports** in `src/Breeze.jl` if the user interface changed
6. **Add validation example** in `examples/` when appropriate
7. **Verify on CPU**: Run tests with `ENV["CUDA_VISIBLE_DEVICES"] = "-1"`
8. **Check explicit imports**: Run `quality_assurance` test

## Key Conventions

- File names: snake_case
- Type names: PascalCase
- Function names: snake_case
- Kernel names: may be prefixed with underscore (e.g., `_compute_tendency_kernel`)
- Extend functions via `Module.function_name(...)`, never via `import`
