# Validation Cases

The general workflow for reproducing a paper (parameter extraction, checking geometry and initial
conditions, short test runs) is in the `/new-simulation` skill. This file holds case-specific notes.

## Tropical Cyclone Genesis (Cronin & Chavas 2019)

| Case | Genesis | Requirements |
|------|---------|--------------|
| Moist (β=1) | Spontaneous | 8km resolution, forms in ~5 days |
| Dry (β=0) | Needs assistance | 2km resolution, seeding, or extreme forcing |

**Key insight**: Latent heat enables WISHE feedback for self-aggregation.

### Critical Parameters

- **Domain**: ≥1152 km for vortex merger cascade (576 km → lattice equilibrium)
- **Resolution**: 2km for dry TCs, 4-8km for moist TCs
- **Disequilibrium**: Tₛ - θ_surface ≈ 10-15 K typical for RCE

### Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| No TC formation | Domain too small | Lx, Ly ≥ 1152 km |
| Simulation blows up | T far from equilibrium | Equilibrated θ profile |
| Flat intensity | Weak forcing (dry) | Moist physics or seed |

Monitor: max surface wind, max ζ/f, mean θ profile, spatial wind/vorticity plots.
