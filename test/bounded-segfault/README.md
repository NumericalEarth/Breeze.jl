# Bounded Topology Segfault Investigation (B.6.4)

**Investigation:** B.6.4 SinkDUS / Bounded Topology  
**Status:** Active - Root cause identified  
**Related:** `cursor-toolchain/rules/domains/differentiability/investigations/bounded-sinkdus-segfault.md`

## Summary

Reactant compilation fails with `"had set op which was not a direct descendant"` when using Breeze's AtmosphereModel with `(Bounded, Bounded, Flat)` topology and `nsteps > 1`.

**Key Finding:** This is NOT a general bounded topology issue. Oceananigans HydrostaticFreeSurfaceModel works with bounded topology, but Breeze AtmosphereModel fails due to:
1. SSPRK3 timestepper complexity (3 substeps per timestep)
2. Diagnostic velocity computation (`u = ρu / ρ`) requiring interpolation with halo access
3. Accumulation of many kernel launches with stencil operators accessing halo cells

## Test Files

| File | Status | Purpose |
|------|--------|---------|
| `test_breeze_bounded_minimal.jl` | FAILING | Minimal reproduction of the issue |

## Running Tests

```bash
# Run from Breeze.jl directory
julia --project=test test/bounded-segfault/test_breeze_bounded_minimal.jl
```

## Current Status (2026-02-11)

- ✅ Root cause identified: nested control flow from boundary-aware interpolation
- ✅ Confirmed Oceananigans bounded works (prognostic velocities)
- ✅ Confirmed Breeze bounded fails (diagnostic velocities with interpolation)
- ⏳ Test file created for verification
- ⏳ Upstream issue needs to be filed with Reactant

## Expected Behavior

| Configuration | nsteps=1 | nsteps=2 | nsteps=4 |
|---------------|----------|----------|----------|
| Oceananigans Bounded | ✓ | ✓ | ✓ |
| Breeze Bounded | ? | ✗ | ✗ |
| Breeze Periodic | ✓ | ✓ | ✓ |

## Workarounds

1. Use Periodic topology instead of Bounded
2. Limit to `nsteps = 1` (not practical)
3. Wait for upstream Reactant fix

## Related Issues

- B.6.3 (DelinearizeIndexingPass): Different issue - about loop-dependent indices in halo kernels
- nsteps-dominate-error: Child issue discovered during investigation - different error at nsteps≥6
