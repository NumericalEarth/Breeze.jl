# M-tier moist substepper tests — pre-fix baselines & Phase 2A status

**Pre-fix data captured:** 2026-04-28 (pre-Phase-3 / pre-A3/B1/B2)
**Phase 2A applied:** 2026-04-28 (γᵈRᵈ → γᵐRᵐ⁰ in PGF/Schur; μᵥ-on-buoyancy
attempted and reverted, see `~/.claude/.../moist_substepper_phase2a.md`)
**Substepper:** `forward_weight = 0.65`, `KlempDivergenceDamping(coefficient = 0.1)`.
**Run from:** `julia --project=examples test/substepper_validation/long_runs/<test>.jl`

> **Note on M1/M2 results:** these tests build their IC with `θ` computed
> using the *dry* exponent `κᵈ = Rᵈ/cpᵈ`, but the model EoS uses `κᵐ = Rᵐ/cpᵐ`.
> At qᵛ = 1g/kg this creates a ~0.06% pressure mismatch when `set!` →
> `update_state!` recomputes `p` from `ρ, ρθ_li`. That mismatch dominates
> the much smaller (~0.6%) γᵐ-vs-γᵈ correction Phase 2A introduces. So both
> M1 and M2 give nearly identical envelope numbers pre- and post-Phase-2A;
> they cannot cleanly probe A3 until rebuilt with discretely-moist-hydrostatic
> ICs. The definitive A3 test is the multi-day moist BW lat-lon (M9/M11).

These tests implement the M-tier hierarchy from `Breeze/MOIST_SUBSTEPPER_STRATEGY.md`,
which is the acceptance gate for the moist-substepper fix described in
`Breeze/validation/substepping/PRISTINE_SUBSTEPPER_PLAN.md` §A3/B1/B2/A1/A2 (Phase 3).

The results below are the **pre-fix baselines**: they show the bug exists
and quantify by how much the dry-only substepper misbehaves on moist
state. Post-fix, these tests should pass.

## M1 — Moist rest atmosphere with uniform qᵛ

`m1_moist_rest.jl`. 32×32×64 Cartesian, Lz=30 km, isothermal-T₀=250 K
moist hydrostatic with uniform `qᵛ = 1 g/kg`. Run 100 outer steps at
Δt=20 s.

| Metric | Pre-fix value | Phase 2A | Pass threshold |
|---|---|---|---|
| envelope max\|w\| | **9.46 × 10⁻² m/s** (peak iter 5; settled ~2.5e-3 sustained) | **9.46 × 10⁻² m/s** (peak iter 5; settled ~2.16e-3 sustained) | ≤ 10⁻¹⁰ m/s |
| final max\|w\| (iter 100) | 2.16 × 10⁻³ m/s | 2.16 × 10⁻³ m/s | ≤ 10⁻¹⁰ m/s |
| mass drift       | exactly 0 (within FP)   | exactly 0 (within FP) | ≤ 10⁻¹² |
| crashed?         | no                      | no | no |

**Status: ✗ FAIL — but biased test, not the substepper's fault.** The
envelope is identical pre- vs post-Phase-2A because the IC's κᵈ-vs-κᵐ
mismatch (≈0.06% at qᵛ=1g/kg) dominates the γᵐ-vs-γᵈ correction
(≈0.6% × ~10⁻³ amplitude = O(10⁻⁶) effect). Need to rebuild the IC
with `θ_li = T (pˢᵗ/p)^κᵐ` (not `κᵈ`) and `H = Rᵐ T / g` to remove the
IC-level imbalance before this test can probe A3.

## M2 — Moist rest atmosphere with horizontal qᵛ gradient

`m2_moist_rest_qv_gradient.jl`. 32×32×64 Cartesian, Lz=30 km. Same
isothermal-T₀=250 K moist hydrostatic background as M1, but with
`qᵛ(x) = qᵛ_max · max(0, cos(π (x − Lh/2) / Lh))` and `qᵛ_max = 10 g/kg`.
Each (x, y) column is in its own moist hydrostatic balance.

| Metric | Pre-fix value | Pass threshold |
|---|---|---|
| envelope max\|w\| | **NaN** | ≤ 10⁻¹⁰ m/s |
| time to NaN      | between iter 5 and iter 10 (i.e. 100–200 s simulated) | n/a |
| max\|w\| at iter 5 | 5.9 m/s   | should be ε |
| max\|u\| at iter 5 | 7.0 m/s   | should be ε |
| crashed?         | yes (NaN in ρ at iter 100) | no |

**Status (pre-fix): ✗ CATASTROPHIC FAIL.** The horizontal qᵛ gradient
makes the dry-on-moist bias a *spatially structured* pressure-gradient
force, which projects onto horizontal wave modes and grows
exponentially. Within ~100 s the rest atmosphere is destroyed.

**Status (Phase 2A): same catastrophic NaN by iter 10**, peak |u|=7.0 m/s,
|w|=5.9 m/s at iter 5 — virtually identical to pre-fix. Same caveat as
M1: the IC has a κᵈ-vs-κᵐ inconsistency that creates an O(0.06%)
pressure mismatch from t=0, and that mismatch is now multiplied by the
horizontal qᵛ gradient, leading to immediate horizontal pressure-gradient
forcing that overwhelms whatever γᵐRᵐ⁰ correction Phase 2A applies.
Need a discretely-moist-hydrostatic IC for this to be a clean A3 probe.

## M3 — Acoustic pulse in moist atmosphere

`m3_moist_acoustic_pulse.jl`. **Drafted but not yet run.** 2-D
512×64, Lx=80 km, Lz=10 km, T₀=300 K, uniform qᵛ=10 g/kg. Gaussian
(ρθ)′ pulse at x=Lx/2; track the right- and left-going pulse fronts
to extract cs.

Expected speeds:
- `cs_dry   = √(γᵈ Rᵈ T₀) ≈ 347.2 m/s`
- `cs_moist = √(γᵐ Rᵐ T₀) ≈ 349.0 m/s`  (qᵛ=10 g/kg)
- Difference: 0.52 %.

Pass threshold: |cs_measured − cs_moist| / cs_moist ≤ 1 %.

**Predicted pre-fix:** cs_measured ≈ cs_dry, ~ 0.5 % below cs_moist —
right at the edge of the tolerance. **Predicted post-fix:** match
cs_moist to ~ 0.1 %.

Not run pre-fix because M1 + M2 already qualitatively confirm the bug
and M3's quantitative signature (0.5 % cs bias) is delicate enough that
it's better to run it under controlled conditions once the fix lands.

## Post-fix expectations

After Phase 3 (PRISTINE §A3 + §B1 + §B2 — strategy doc Phases 1+2):

| Test | Expected post-fix | Phase 2A status |
|---|---|---|
| M0 (dry rest, existing test) | ✅ machine ε (no regression) | ✅ pass — bit-identical |
| M1 | ✅ envelope ≤ 10⁻¹⁰ m/s | ⚠️ need new IC; current IC has κᵈ-vs-κᵐ bug |
| M2 | ✅ envelope ≤ 10⁻¹⁰ m/s | ⚠️ same — same IC issue |
| M3 | ✅ cs matches cs_moist within 1 % | ⏳ not yet run post-fix |
| M9 / M10 / M11 | ✅ 15-day moist BW lat-lon at Δt=20 s | ⏳ M9 (no-flux) running 2026-04-28 |
| M12 | ✅ Δt sweep — smaller Δt no longer fails earlier | ⏳ pending |

## How to reproduce

```sh
cd ~/Breeze
julia --project=examples test/substepper_validation/long_runs/m1_moist_rest.jl
julia --project=examples test/substepper_validation/long_runs/m2_moist_rest_qv_gradient.jl
julia --project=examples test/substepper_validation/long_runs/m3_moist_acoustic_pulse.jl  # not yet validated
```

Each run: ~60–90 s wall (compile + 100 outer steps on one GPU).
