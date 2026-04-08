# Acoustic Substepping Cleanup and Damping Plan

This note proposes a focused cleanup of Breeze's split-explicit acoustic
substepping in three concurrent directions:

1. **Comments and notation** — make the active substep loop read as a
   faithful, line-by-line implementation of the MPAS conservative-perturbation
   acoustic substep algorithm. Remove historical comments that contradict the
   active code.
2. **Strategy types and a real bug fix** — replace the silently-ignored
   `divergence_damping_coefficient` parameter with a typed
   `damping :: AcousticDampingStrategy` API exposing four damping strategies
   (no damping, MPAS Klemp 2018, ERF-style pressure projection, and a
   conservative-variable projection variant).
3. **File reorder** — reorganize `acoustic_substepping.jl` so the active
   path reads top-down: prognostic state → coefficient helpers → horizontal
   forward → vertical solve → recovery → damping dispatch → stage driver.

The immediate motivation is correctness (a real bug — the user-facing
divergence-damping coefficient does nothing) and clarity (so the active code
can be cross-checked against the MPAS source one block at a time).
Performance-oriented cleanup is out of scope.

## Identified bug — `divergence_damping_coefficient` is silently dropped

While drafting this plan we found that the user-facing
`divergence_damping_coefficient` parameter has no effect on the runtime.

Locations:

- Constructor stores it: `src/CompressibleEquations/time_discretizations.jl:39, 45-50`.
- Substepper reads it from the time discretization:
  `src/CompressibleEquations/acoustic_substepping.jl:177, 239, 297`.
- Substep loop binds it to a local: `src/CompressibleEquations/acoustic_substepping.jl:1186`
  (`ϰᵈⁱ = substepper.divergence_damping_coefficient`).
- That local is **never referenced again**.
- The active divergence-damping kernel uses a hardcoded constant instead:
  `src/CompressibleEquations/acoustic_substepping.jl:1279` (`smdiv = FT(0.1)`).

The hardcoded `0.1` happens to equal both the MPAS default `config_smdiv = 0.1`
and the existing default `divergence_damping_coefficient = 0.10`, so user code
that takes the default produces correct results. But a user passing
`divergence_damping_coefficient = 0.5` (or any other value) silently gets the
hardcoded `0.1` with no warning.

This bug is fixed in **Phase 2.4** below as part of replacing the parameter
with the new strategy API.

## Background

The active substepper combines three independent algorithmic decisions:

1. **Fast prognostic variables.** The active substepper uses the MPAS
   conservative-perturbation form: ``(\rho u)''``, ``(\rho v)''``,
   ``(\rho w)''``, ``\rho''``, ``(\rho\theta)''``. This is the same family
   used by ERF (``U''``, ``V''``, ``W''``, ``\rho''``, ``\Theta''``) and is
   the natural form for a split-explicit operator that wants conservation.
   CM1 uses a different family (a pressure-like fast prognostic ``pp3d``
   plus momentum). CM1's family is **not** what runs in Breeze.

2. **RK stage substep distribution.** The active path is Wicker–Skamarock
   RK3 with ``β = (1/3, 1/2, 1)``. The substep counts per stage and the
   substep size are configurable via the `AcousticSubstepDistribution` type
   (`ProportionalSubsteps` for ``(N/3, N/2, N)`` substeps with constant
   ``Δτ = Δt/N``, or `MonolithicFirstStage` for the MPAS-3 form with a
   single ``Δt/3`` substep in stage 1). This was already cleaned up in a
   prior commit.

3. **Divergence-damping strategy.** This is what this plan cleans up.
   The active substepper currently runs the MPAS Klemp-Skamarock-Ha 2018
   form (a momentum correction proportional to
   ``\partial_x \delta_\tau (\rho\theta)''``) but exposes a parameter
   (`divergence_damping_coefficient`) whose docstring describes a different
   form (the CM1/WRF forward extrapolation
   ``\tilde{\pi}' = \pi' + \varkappa^{di}(\pi' - \pi'_\mathrm{old})``)
   and whose value is not actually used. The cleanup replaces the parameter
   with a typed strategy API supporting the four major forms in the
   literature.

The current state of the source mixes language and ideas from all three
decisions inconsistently. After the cleanup, the source should make each of
the three decisions visible and selectable independently.

## Cleanup objectives

After the cleanup, the source should make the following obvious from a
casual read:

- The active substep loop is the MPAS conservative-perturbation algorithm.
- The RK stage substep distribution is a separate, type-dispatched choice.
- The divergence-damping strategy is a third, separately-typed choice.
- Comments describe what the active code does, not what an older version did.
- Historical Exner-prognostic / CM1-style language lives in a clearly-labeled
  "algorithm lineage" appendix block, not interleaved with the active code.

### Recommended notation

Use Breeze mathematical notation everywhere in comments and docs for the
active MPAS-style path:

- ``(\rho u)''``, ``(\rho v)''``, ``(\rho w)''``
- ``\rho''``
- ``(\rho\theta_v)''`` or ``(\rho\theta)''`` depending on whether ``\theta``
  is the dry or virtual potential temperature being treated as the frozen
  thermodynamic variable
- ``\Delta t`` for the outer RK step
- ``\Delta \tau`` for the acoustic substep
- ``\varepsilon`` for off-centering
- ``\Delta \tau^\varepsilon = \tfrac12(1+\varepsilon)\Delta\tau``

When the code must be compared line-by-line with MPAS, give the translation
once, not in every comment:

- ``(\rho u)''`` ↔ `ru_p`
- ``(\rho w)''`` ↔ `rw_p`
- ``\rho''`` ↔ `rho_pp`
- ``(\rho\theta)''`` ↔ `rtheta_pp`

### Comment cleanup rules

1. Comments on the active path must describe the active variables, not an
   old Exner-pressure-prognostic algorithm.
2. Historical comments should be moved to a short "Algorithm lineage" block
   at the bottom of the file (or to the appendix docs).
3. Any comment that says "CM1-style" must be explicit about whether it
   refers to (a) stage scheduling, (b) divergence damping, or (c) the actual
   fast prognostic variables.
4. Any comment that says "MPAS-style" must identify which of (a)
   perturbation variables, (b) the vertical Schur complement, (c) divergence
   damping, or (d) recovery, it refers to.

## Strategy types design

### Type hierarchy

```julia
abstract type AcousticDampingStrategy end

# 1. No damping — useful as a baseline for "is divergence damping the
#    bottleneck?" experiments.
struct NoDivergenceDamping <: AcousticDampingStrategy end

# 2. MPAS Klemp-Skamarock-Ha 2018 form — momentum correction post-substep
#    based on the discrete thermodynamic-divergence proxy. Active form
#    today.
struct ThermodynamicDivergenceDamping{FT} <: AcousticDampingStrategy
    coefficient  :: FT                  # MPAS config_smdiv
    length_scale :: Union{FT, Nothing}  # nothing → use min(Δx, Δy)
end

# 3. ERF / CM1 / WRF literal pressure-projection form — forward
#    extrapolation of the diagnosed Exner perturbation, applied via an
#    EOS-linearized correction to the (ρθ)″ field that the next substep's
#    horizontal PGF reads.
struct PressureProjectionDamping{FT} <: AcousticDampingStrategy
    coefficient :: FT                   # β_d, the forward-projection weight
end

# 4. Smaller-code-change variant — forward extrapolation of the prognostic
#    (ρθ)″ directly, used in the next substep's horizontal PGF. Equivalent
#    to (3) at the linearized level but does not require a separate EOS
#    evaluation.
struct ConservativeProjectionDamping{FT} <: AcousticDampingStrategy
    coefficient :: FT                   # β_d
end
```

### Why two projection variants

The `PressureProjectionDamping` and `ConservativeProjectionDamping`
strategies implement the same underlying physical idea (forward-extrapolate
the pressure-like fast variable that drives the next substep's PGF) on two
different fields:

| | What gets projected | Where the projection is applied | Extra code cost |
|---|---|---|---|
| `PressureProjectionDamping` | diagnosed Exner perturbation ``\pi''`` | next substep's PGF (via EOS-linearized conversion to ``(\rho\theta)''_\mathrm{for\ pgf}``) | one extra CenterField (``\pi''_\mathrm{old}``); per-substep EOS evaluation |
| `ConservativeProjectionDamping` | conservative prognostic ``(\rho\theta)''`` | next substep's PGF (direct) | none — reuses `previous_rtheta_pp` |

At the strict linearized level (perturbations small relative to the
reference state, EOS map approximated by its tangent at the reference) the
two strategies are mathematically equivalent — both add the same correction
to the next PGF up to discretization details. They differ at second order
in the perturbations because the EOS map ``\pi = f((\rho\theta))`` is
nonlinear, and they may differ at the discrete level depending on which
finite-difference operators are applied where.

We implement **both** because:

1. The only way to determine empirically whether the choice matters for
   Breeze's tests (BW noise, IGW, MPAS comparisons) is to run both forms
   on the same configurations.
2. Bit-comparing against ERF (which uses `PressureProjectionDamping` in
   spirit) and against MPAS (which uses `ThermodynamicDivergenceDamping`)
   requires having both forms available.
3. `ConservativeProjectionDamping` is a much smaller code change, so we get
   most of the empirical-comparison value for less implementation work.
   `PressureProjectionDamping` is the literal-fidelity option for users
   who care about exact ERF/CM1 parity.

The default strategy across the cleanup is
`ThermodynamicDivergenceDamping(coefficient = 0.1)`, which preserves
today's runtime behavior bit-for-bit.

### Mathematical definitions

**`NoDivergenceDamping`**: no operation. The substep advances
``(\rho u)''``, ``(\rho v)''``, ``(\rho w)''``, ``\rho''``,
``(\rho\theta)''`` with no damping kernel.

**`ThermodynamicDivergenceDamping`** (active form today): post-substep
horizontal momentum correction (Klemp, Skamarock & Ha 2018):

```
Δ(ρu)″ = coef · ∂ₓ(δ_τ (ρθ)″) / (2 θ_m,edge)
Δ(ρv)″ = coef · ∂ᵧ(δ_τ (ρθ)″) / (2 θ_m,edge)
coef   = 2 · smdiv · ℓ_disp / Δτ
δ_τ (ρθ)″ = (ρθ)″_new − (ρθ)″_old
```

The divergence proxy ``\delta_\tau (\rho\theta)''`` comes from the discrete
pressure tendency equation, which is what gives this form its
gravity-wave-preserving property (Klemp et al. 2018).

**`PressureProjectionDamping`** (literal ERF / CM1 / WRF form): forward
extrapolation of the diagnosed Exner perturbation:

```
π″       = EOS((ρθ)″, frozen state)            ← diagnose from current (ρθ)″
π̃″      = π″ + β_d · (π″ − π″_old)             ← forward project
π″_old   ← π″                                   ← save for next substep
```

The forward-projected ``\tilde{\pi}''`` is meant to enter the next substep's
horizontal PGF in place of ``\pi''``. In Breeze's MPAS conservative
formulation the existing horizontal forward kernel reads ``(\rho\theta)''``
and computes the PGF as
``c^2 \cdot \Pi_\mathrm{face} \cdot \partial(\rho\theta)''/\partial x``.
To apply the literal ERF form *without* rewriting the kernel, the strategy
converts the projected ``\tilde{\pi}''`` back into a projected
``(\rho\widetilde{\theta})''_\mathrm{for\ pgf}`` via the linearized EOS:

```
(ρθ̃)″_for_pgf = (ρθ)″ + (∂(ρθ)/∂π) · (π̃″ − π″)
              = (ρθ)″ + (cₚ · ρ · θ_v / c²) · β_d · (π″ − π″_old)
```

Requirements:

- A ``\pi''`` scratch field (CenterField) for the diagnosed value.
- A ``\pi''_\mathrm{old}`` scratch field (CenterField) for the previous
  substep.
- A per-substep EOS evaluation: ``\pi'' = (R \cdot \rho\theta / p^{st})^\kappa - \pi_\mathrm{ref}``.
- The linearized EOS conversion factor ``c_p \cdot \rho \cdot \theta_v / c^2``,
  which must be evaluated from the frozen reference state at the cell center.

**`ConservativeProjectionDamping`**: forward extrapolation of the
conservative perturbation directly:

```
(ρθ̃)″_for_pgf = (ρθ)″ + β_d · ((ρθ)″ - previous_rtheta_pp)
```

where `previous_rtheta_pp` is the field already maintained by the substepper
for the MPAS damping path.

At the strict linearized level (using ``(\rho\theta)'' \approx (c_p \rho \theta_v / c^2) \cdot \pi''``)
this is equivalent to `PressureProjectionDamping`. The two strategies
diverge at second order in the perturbations and at the discretization
level — Phase 3 includes an empirical comparison.

### Pre-substep filter and post-substep correction hooks

The strategy interface uses two dispatched methods, each with one default
no-op behavior:

```julia
# Pre-substep filter — modifies the field that the next horizontal forward
# kernel reads as the PGF source. Default for all strategies except the two
# projection variants is an identity copy (ρθ″_for_pgf ← ρθ″).
apply_pgf_filter!(strategy, substepper, grid)

# Post-substep momentum correction. Default for all strategies except
# ThermodynamicDivergenceDamping is a no-op.
apply_divergence_damping!(strategy, substepper, grid, Δτ)
```

| strategy | `apply_pgf_filter!` | `apply_divergence_damping!` |
|---|---|---|
| `NoDivergenceDamping` | identity copy | no-op |
| `ThermodynamicDivergenceDamping` | identity copy | MPAS Klemp-2018 momentum correction |
| `PressureProjectionDamping` | EOS + forward project ``\pi'' \to \tilde{\pi}'' \to (\rho\widetilde{\theta})''_\mathrm{for\ pgf}`` via linearized EOS | no-op |
| `ConservativeProjectionDamping` | forward project ``(\rho\theta)'' \to (\rho\widetilde{\theta})''_\mathrm{for\ pgf}`` directly | no-op |

The horizontal forward kernel reads `substepper.ρθ″_for_pgf` (a new
`CenterField` introduced in Phase 3) as the PGF source. Strategy methods
are responsible for writing this field appropriately at the start of each
substep.

This design keeps the substep loop body identical across all four
strategies — only the per-substep `apply_pgf_filter!` and
`apply_divergence_damping!` calls differ. The kernel signatures themselves
do not change; only the field passed in changes.

## Implementation phases

Each phase is small enough to be one PR. Behavior is preserved by default
across all phases (the bit-equivalent regression in 2.6 must pass after
each phase).

### Phase 1 — comment and docstring cleanup *(no behavior change)*

| Task | File | Description |
|---|---|---|
| 1.1 | `time_discretizations.jl:9-35` | Drop the "Exner pressure formulation following CM1 (Bryan 2002)" framing from the `SplitExplicitTimeDiscretization` top docstring. Replace with: "Split-explicit acoustic substepping using the MPAS conservative-perturbation formulation (`ru_p, rv_p, rw_p, rho_pp, rtheta_pp`) with a Wicker–Skamarock RK3 outer loop and a configurable divergence-damping strategy." |
| 1.2 | `time_discretizations.jl:31` | Remove or rewrite the `divergence_damping_coefficient` field docstring. Currently describes the CM1 forward-extrapolation form, which is *not* active. Note that the field will be replaced by `damping :: AcousticDampingStrategy` in Phase 2. |
| 1.3 | `acoustic_substepping.jl` (top of file) | Add a 10-15 line "active path" header block after the module-level docstring, listing exactly: which prognostics are active (``\rho''``, ``(\rho\theta)''``, ``(\rho u)''``, ``(\rho v)''``, ``(\rho w)''``), which RK stage schedule is used (`AcousticSubstepDistribution` types), and which damping path is active. Note that `smdiv` is currently hardcoded — to be fixed in Phase 2.4. |
| 1.4 | `acoustic_substepping.jl` (throughout) | Cull historical "Exner-prognostic" / "CM1-style" comments that contradict the active code. Move them to a single "Algorithm lineage" appendix block at the bottom of the file. |
| 1.5 | `docs/src/appendix/substepping_cleanup_and_damping_plan.md` | (Done by the rewrite this plan now lives in.) The bug callout and the consolidated phased plan live in this doc. |

### Phase 2 — strategy types and bug fix *(behavior-preserving by default)*

| Task | File | Description |
|---|---|---|
| 2.1 | `time_discretizations.jl` | Add `AcousticDampingStrategy` abstract type and `NoDivergenceDamping`, `ThermodynamicDivergenceDamping{FT}` concrete types. Constructor for `ThermodynamicDivergenceDamping` defaults `coefficient = 0.1, length_scale = nothing`. |
| 2.2 | `time_discretizations.jl` | Add `damping :: AcousticDampingStrategy` field (type-parameterized) to `SplitExplicitTimeDiscretization`. Constructor defaults to `ThermodynamicDivergenceDamping()`. Backwards-compat: keep `divergence_damping_coefficient` as a kwarg for one release with a `@warn`, mapping to `damping = ThermodynamicDivergenceDamping(coefficient = ...)` if `damping` is left at default. |
| 2.3 | `acoustic_substepping.jl:173, 200, 297` | Add `damping` field to `AcousticSubstepper` (with concrete type parameter, alongside `substep_distribution`). Update constructor and `Adapt.adapt_structure`. |
| 2.4 | `acoustic_substepping.jl:1186, 1275-1297` | **Fix the bug.** Replace the hardcoded `smdiv = FT(0.1)` block with `apply_divergence_damping!(substepper.damping, substepper, grid, FT(Δτ))`. Define methods: `NoDivergenceDamping → no-op`; `ThermodynamicDivergenceDamping → wraps the existing _mpas_divergence_damping! kernel with smdiv = damping.coefficient and len_disp = damping.length_scale ?? min(Δx, Δy)`. |
| 2.5 | `CompressibleEquations.jl, Breeze.jl` | Export `AcousticDampingStrategy`, `NoDivergenceDamping`, `ThermodynamicDivergenceDamping`. |
| 2.6 | `test/` or scratch | **Verify**: (a) bit-equivalence regression — `damping = ThermodynamicDivergenceDamping(coefficient = 0.1)` (the new default) gives bit-identical results to today's hardcoded path on the canonical IGW (Δt = 6.86 s, N = 12, 100 outer steps). (b) `NoDivergenceDamping` runs and produces slightly more noise than (a). (c) Damping experiment now actually possible — 14-day BW at Δt = 200 s with `coefficient ∈ {0.1, 0.25, 0.5, 1.0}`. Reports crash time + max\|w\| trajectory. **This finally answers "is divergence-damping strength the BW bottleneck?"** |

### Phase 3 — `PressureProjectionDamping` and `ConservativeProjectionDamping`

Both pressure-projection variants land in this phase. They share the
`apply_pgf_filter!` infrastructure and the `ρθ″_for_pgf` scratch field.
Phase 3 depends on Phase 2 (the strategy types and dispatch hook must
exist first) but is independent of Phase 4 (file reorder).

| Task | File | Description |
|---|---|---|
| 3.1 | `time_discretizations.jl` | Add `PressureProjectionDamping{FT}` and `ConservativeProjectionDamping{FT}` next to the other strategies. Each has a single `coefficient :: FT` field (the forward-projection weight ``\beta_d``, default `0.1`). |
| 3.2 | `acoustic_substepping.jl` | Add `ρθ″_for_pgf :: CenterField` to `AcousticSubstepper`. This is the field the horizontal forward kernel reads as the PGF source. Update constructor + `Adapt.adapt_structure`. Initialize as a copy of ``(\rho\theta)''``. |
| 3.3 | `acoustic_substepping.jl` | For `PressureProjectionDamping` only, also add `π″ :: CenterField` and `π″_old :: CenterField` to `AcousticSubstepper`. These store the diagnosed Exner perturbation for the current and previous substeps. Use the existing `_default_len_disp` / EOS conventions for the diagnosis. |
| 3.4 | `acoustic_substepping.jl` | Add the `apply_pgf_filter!(::AcousticDampingStrategy, substepper, grid)` strategy method family. Default for `NoDivergenceDamping` and `ThermodynamicDivergenceDamping` is identity copy `(\rho\theta)'' \to (\rho\theta)''_\mathrm{for\ pgf}`. For `ConservativeProjectionDamping`: launch a kernel computing `(\rho\theta)''_\mathrm{for\ pgf} = (\rho\theta)'' + \beta_d ((\rho\theta)'' - \texttt{previous\_rtheta\_pp})`. For `PressureProjectionDamping`: launch a two-pass kernel that (a) diagnoses ``\pi'' = (R (\rho\theta)'' / p^{st})^\kappa - \pi_\mathrm{ref}``; (b) forward-projects ``\tilde{\pi}'' = \pi'' + \beta_d (\pi'' - \pi''_\mathrm{old})``; (c) computes the EOS-linearized conversion ``\Delta(\rho\theta)'' = (c_p \rho \theta_v / c^2) (\tilde{\pi}'' - \pi'')`` and writes ``(\rho\theta)''_\mathrm{for\ pgf} = (\rho\theta)'' + \Delta(\rho\theta)''``; (d) updates ``\pi''_\mathrm{old} \leftarrow \pi''`` for the next substep. |
| 3.5 | `acoustic_substepping.jl:1216-1222` | Modify the substep loop to call `apply_pgf_filter!(substepper.damping, substepper, grid)` *before* `_mpas_horizontal_forward!`, and pass `substepper.ρθ″_for_pgf` (instead of `substepper.ρθ″`) as the `rtheta_pp` argument to the kernel. The kernel signature itself stays unchanged. |
| 3.6 | `CompressibleEquations.jl, Breeze.jl` | Export `PressureProjectionDamping`, `ConservativeProjectionDamping`. |
| 3.7 | `test/` or scratch | **Bit-equivalence sanity check**: with all four strategies set to "no damping" (`NoDivergenceDamping()`, `ThermodynamicDivergenceDamping(coefficient = 0.0)`, `PressureProjectionDamping(coefficient = 0.0)`, `ConservativeProjectionDamping(coefficient = 0.0)`), all four must give bit-identical results on the canonical IGW. This validates that the `apply_pgf_filter!` and `apply_divergence_damping!` dispatch boundaries are clean (no spurious side effects when both hooks are no-ops). |
| 3.8 | `test/` or scratch | **Comparison experiment**: 14-day BW at Δt = 200 s with each of the four strategies at typical defaults (`NoDivergenceDamping()`, `ThermodynamicDivergenceDamping(coefficient = 0.1)`, `PressureProjectionDamping(coefficient = 0.1)`, `ConservativeProjectionDamping(coefficient = 0.1)`). Report crash time + max\|w\| trajectory for all four. **This finally answers: does the *form* of damping (MPAS Klemp 2018 momentum correction vs literal ERF pressure projection vs the conservative-projection variant) matter for Breeze's BW noise problem, or only the strength?** |

### Phase 4 — file structure reorder *(no behavior change)*

| Task | File | Description |
|---|---|---|
| 4.1 | `acoustic_substepping.jl` | Reorder the file so the active path reads top-down: (1) perturbation variable definitions and notation block; (2) frozen stage state block; (3) coefficient helpers; (4) horizontal forward step; (5) column RHS build; (6) vertical solve; (7) post-solve back-substitution; (8) damping strategy dispatch (`apply_pgf_filter!`, `apply_divergence_damping!`); (9) recovery; (10) stage driver. Pure rearrangement — the active code blocks stay byte-identical, only their order changes. Single commit. Done last because it touches every line; doing it earlier would make the diffs for Phases 1–3 harder to review. |

## Out of scope: Rayleigh sponge and horizontal hyperdiffusion

Two damping mechanisms are present in MPAS-A but absent from Breeze. They
contribute to MPAS-A's stability at large Δt and are likely contributors to
the gap between MPAS-A's "Δt = 1800 s on a 1° mesh works" and Breeze's
"Δt > 200 s crashes by day 2 on the 1° lat–lon grid". They are **not**
divergence damping and **not** part of the substepper, so they do not
belong in this cleanup plan.

A grep across `src/` for `rayleigh|sponge|absorbing|hyperdiff|biharmonic|del4|eddy_visc4`
returns zero matches.

- **Top Rayleigh / sponge layer.** MPAS uses `config_rayleigh_damp_w`,
  `config_zd`, etc. to absorb upward-propagating gravity waves at the
  model top. Without a sponge, vertically propagating gravity waves
  reflect off the rigid lid and re-energize the troposphere.
- **4th-order horizontal hyperdiffusion.** MPAS uses
  `config_h_mom_eddy_visc4` and `config_h_theta_eddy_visc4`
  (Smagorinsky-style biharmonic) to damp grid-scale and ``2\Delta x``
  noise on momentum and ``\theta``. This is the dominant noise filter in
  MPAS-A's BW runs and is *not* the same as divergence damping.

These should be tracked in a separate plan, ideally tagged "Top sponge
layer and horizontal hyperdiffusion for large-Δt stability". They become
especially relevant if Phase 3's empirical comparison reveals that no
choice of divergence-damping strategy is sufficient to make BW
``\Delta t > 60`` s stable on the 1° lat–lon grid.

## Recommended commit boundaries

| commit | content |
|---|---|
| 1 | Phase 1 (docstrings + comments + this doc rewrite) |
| 2 | Phase 2 (strategy types + bug fix) |
| 3 | Phase 3 (`PressureProjectionDamping` + `ConservativeProjectionDamping`) |
| 4 | Phase 4 (file structure reorder) |

Each commit independently passes the bit-equivalent regression in step 2.6
(or its Phase 3 expansion) — i.e., the canonical IGW max\|w\| with the new
default strategy must match today's hardcoded path to the floating-point
roundoff level.

## Bottom line

The active Breeze substepper is a combination of:

- an MPAS-style conservative-perturbation acoustic substep,
- a chosen RK stage substep distribution (`AcousticSubstepDistribution`),
- and a divergence-damping strategy (`AcousticDampingStrategy`).

Those three decisions should appear separately in the source and in the
public configuration API.

After this cleanup:

- The bug is fixed: `divergence_damping_coefficient` is replaced by a typed
  `damping` field that actually flows into the kernel.
- Four damping strategies are available (no damping, MPAS Klemp 2018,
  literal ERF pressure projection, conservative-projection variant) — three
  of which are new capabilities, not just renames.
- The active substepper code matches the MPAS conservative-perturbation
  algorithm line-by-line, with historical comments isolated in a clearly
  labeled appendix block.
- Phase 3 produces an empirical comparison answering the open question:
  "does the form of divergence damping matter for Breeze's BW noise, or
  is it just the strength?"
