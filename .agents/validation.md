# Implementing Validation Cases

When reproducing paper results:

## 1. Parameter Extraction

- **Read the paper carefully** and extract ALL parameters: domain size, resolution, physical constants,
  boundary conditions, initial conditions, forcing, closure parameters
- Check parameter tables (often "Table 1") and figure captions
- Note the coordinate system and conventions used

## 2. Geometry Verification (BEFORE running long simulations)

- **Always visualize the grid/domain geometry first**
- Verify domain extents match the paper
- Compare your geometry plot to figures in the paper

## 3. Initial Condition Verification

- After setting initial conditions, check:
  - `minimum(field)` and `maximum(field)` make physical sense
  - Spatial distribution looks correct (visualize if needed)

## 4. Short Test Runs

Before running a long simulation:
- Run for a few timesteps on CPU at low resolution
- Check for NaNs (`maximum(abs, u)` etc.), reasonable velocities, meaningful output
- Then test on GPU to catch GPU-specific issues

## 5. Progressive Validation

- Run a short simulation (~1 hour sim time) and visualize
- Compare to early-time paper figures if available

## 6. Match Paper Figures

- Same colormaps, axis ranges, time snapshots, diagnostics

## Common Issues

- **NaN blowups**: Timestep too large, unstable ICs, or `if`/`else` on GPU (use `ifelse`)
- **Nothing happening**: Wrong sign on buoyancy anomaly, ICs not applied, forcing inactive
- **Wrong flow direction**: Check coordinate conventions
- **GPU issues**: Avoid branching, ensure type stability

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
