# Debugging Physics Simulations

## Thermodynamic Variable Discipline

| Variable | Meaning |
|----------|---------|
| `T` | Temperature (K) |
| `θ` | Potential temperature: `θ = T / Π` where `Π = (p/p₀)^κ` |
| `ρs` | Density × static energy (J/m³) — prognostic only under `StaticEnergyThermodynamics` |
| `ρθ` | Density × potential temperature (kg·K/m³) |
| `ρE` | Density × total energy (J/m³) — the *interface* key for an energy flux or forcing, applied to whichever thermodynamic variable the model evolves |
| `ρqᵛ`, `ρqᵉ` | Prognostic moisture density (kg/m³) — vapor under non-equilibrium cloud formation, equilibrium moisture under saturation adjustment |
| `ρqᵗ` | Total moisture density (kg/m³) — the *interface* key for a water flux or forcing, applied to whichever moisture variable the scheme evolves |

Before applying forcing: (1) check what variable the paper uses, (2) check working examples,
(3) check Breeze's prognostic variable, (4) verify units.

An energy input (W/m² at a boundary, W/m³ in the interior) goes under `ρE` — or the specific
key `E` for forcings — and Breeze converts it for the prognostic variable: divided by `cᵖᵐ`
(fluxes) or `cᵖᵐ Π` (forcings) for `ρθ`, unconverted for `ρs`.

A water input (kg/m²/s at a boundary, kg/m³/s in the interior) goes under `ρqᵗ` — or the
specific key `qᵗ` — and is applied to the prognostic moisture unconverted, since water added
there is water added to `qᵗ` under every scheme.

The specific names are keys only where they are actually prognostic: `ρs`/`s` when static
energy is the thermodynamic variable, `ρqᵛ` or `ρqᵉ` depending on the microphysics. Supplying
both an interface key and its target is an error, and both `boundary_conditions` and `forcing`
reject any unrecognized key with an `ArgumentError` rather than dropping it.

**Common mistakes**: Applying T tendency to θ, confusing `ρs` with `ρθ`, forgetting Exner function in T↔θ conversion.

## When a Stable Simulation Becomes Unstable

1. **STOP** — Don't add fixes. 2. Identify last working state via `git log`/`git diff`.
3. Revert. 4. Make ONE change at a time. 5. Find the breaking change.

The instability is NOT pre-existing if the code was stable before your changes.

## Diagnose-Before-Fix Protocol

1. **STOP** — Don't immediately try a fix.
2. **Characterize**: Where? What values? When did it start?
3. **Work backwards**: Extreme at high altitude → what's special there? NaN → division by small numbers?
4. **Compute analytically**: Expected tendency? Physically reasonable?
5. **Only then** propose a targeted fix.

**Anti-pattern**: "Blows up at high altitude → cap values." This treats symptoms.
**Correct**: "Why high altitude? Low Π → amplified forcing → fix: equilibrate initial condition."

## Model Architecture Awareness

When implementing from papers using different models (SAM, WRF, MPAS):
1. Identify the paper's prognostic variables and how forcing is applied.
2. Identify Breeze's prognostics (`ρθ` or `ρs`).
3. Derive the transformation (e.g., ∂θ/∂t = ∂T/∂t × 1/Π — can amplify 10× at high altitude!).
4. Check if the paper's model handles this implicitly (e.g., SAM uses static energy ∝ T).

## Microphysics Implementation

Interface in `src/AtmosphereModels/microphysics_interface.jl`. Key functions:
- `maybe_adjust_thermodynamic_state`: Saturation adjustment for equilibrium schemes; trivial for non-equilibrium schemes.
- `microphysical_tendency`: Tendencies for prognostic microphysical variables.
- `moisture_fractions`: Moisture mass fractions from prognostic fields.
- `update_microphysical_fields!`: Update diagnostic fields after state update.

## Checklist Before Modifying Physics Code

- [ ] Read relevant working examples (BOMEX, RICO, prescribed_SST)
- [ ] Identified which field applies similar physics
- [ ] Verified implementation matches paper specification
- [ ] Computed tendency magnitudes analytically at key locations
- [ ] Verified ICs are compatible with forcing
- [ ] Making ONE change only
- [ ] Committed or stashed current working state
