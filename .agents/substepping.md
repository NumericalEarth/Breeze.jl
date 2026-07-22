# Split-Explicit Acoustic Substepping

Breeze's compressible dynamics integrate the fast (acoustic / gravity-wave) terms with a
split-explicit acoustic substep loop nested inside a Wicker–Skamarock RK3 (WS-RK3) outer
scheme — the same design as WRF-ARW and MPAS-Atmosphere. This doc records the WRF/MPAS
correspondence and the non-obvious design decisions behind the implementation, so they live
in one place rather than scattered across comments.

- Source: `src/CompressibleEquations/acoustic_substepping.jl`, `substep_boundary_update.jl`,
  `terrain_compressible_physics.jl`.
- User-facing math (equation numbers, derivations): `docs/src/compressible_dynamics.md`.

## What "split-explicit" means here

Each outer step is a 3-stage WS-RK3. Within stage `k` the slow tendencies are frozen and the
fast system is advanced by `nₖ` acoustic substeps of size `Δτ`, using forward–backward (MPAS
"first small step") sequencing: horizontal momentum forward-Euler, then the mass / `ρθ`
predictors and the vertically-implicit `(ρw)′` solve. Prognostics are carried as perturbations
from the stage-entry large-step state `Uᴸ` (`ρ′ = ρ − ρᴸ`, `(ρθ)′ = ρθ − ρθᴸ`, `(ρu)′ = ρu − ρuᴸ`,
…), so the WS-RK3 stage update `U^(k) = U⁰ + βₖ Δt R` falls out of the substep loop.

WS-RK3 stage fractions: `β = (1/3, 1/2, 1)` — note `β₁ + β₂ + β₃ = 11/6`, which is where the
number in the boundary-update argument below comes from.

## WRF/MPAS correspondence

| Breeze construct | Source | WRF / MPAS analog |
|---|---|---|
| WS-RK3 outer + acoustic substeps | `acoustic_substepping.jl` | Wicker & Skamarock (2002); Skamarock & Klemp (2008) split-explicit |
| Forward–backward "first small step" sequencing | `_explicit_horizontal_step!`, stage loop | MPAS first-small-step acoustic sequence |
| Perturbation init by rewind `(ρu)′ = U⁰ − Uᴸ_stage` | `_initialize_perturbation_with_rewind!` | WRF/MPAS: integrate each stage from `U(t) ≡ Uᴸ_outer` |
| Frozen large-step PGF reinstated in the explicit step | `_explicit_horizontal_step!` | MPAS `tend_u_euler` |
| Acoustic-mean velocity for non-acoustic scalar transport | `AcousticSubstepper` time-averaged velocity | WRF/MPAS dynamics–transport split |
| Per-substep divergence damping | `_*_divergence_damping!` | Klemp, Skamarock & Ha (2018); coefficient = MPAS `config_smdiv` |
| Per-substep open-boundary relaxation of `ρ′`, `(ρθ)′` (schemeless open sides) | `_relax_open_boundary_x!` / `_y!` | mirrors WRF/ERF/MPAS enforcing the specified lateral boundary every substep (a schemeless open wall otherwise reflects the acoustic pressure perturbation) |
| **Specified-zone boundary update** (`SubstepBoundaryUpdate`) | `substep_boundary_update.jl` | MPAS specified zone |
| — specified-cell / -face masks | `specified_zone_cell` / `specified_zone_faces` | MPAS `specZoneMaskCell` / `specZoneMaskEdge` |
| — per-substep tendency increment | `specified_zone_increment` | MPAS `ru_p += dts·lbc_tend_ru` |
| — `(ρw)′` zero-gradient column closure | `replace_specified_column_vertical_momentum!` | WRF `zero_grad_bdy` |
| — post-loop re-imposition of the zone | `reimpose_specified_zone!` | WRF specified-zone contract (interior physics never acts on the zone) |
| — `α`-relaxation superseded on specified sides | `_relax_open_boundary_*!` gated by `!specified_sides` | supersedes the Davies-style slow relaxation on driven sides |

Source comments describe Breeze's own logic; the WRF/MPAS correspondence lives here rather than
as inline `# MPAS …` / `# WRF …` tags.

## The increment-vs-overwrite decision (the "11/6 note")

The specified zone's momentum and scalar perturbations are updated **each acoustic substep by
an increment**, never by an absolute overwrite:

```
(ρu)′ ← (ρu)′ + Δτ · ∂ₜ(ρu)_boundary        # MPAS  ru_p += dts·lbc_tend_ru
```

Why the increment is load-bearing: each WS-RK3 stage re-initializes the perturbation with the
rewind `(ρu)′ = U⁰ − Uᴸ_stage` (so the specified cell's full momentum is reset to the
outer-step value `U⁰` at every stage start). Composed with the increment accumulated over
stage `k`'s substeps (total substep time `βₖ Δt`), the specified cell recovers

```
ρu = U⁰ + βₖ Δt · ∂ₜ      at the end of stage k,
```

i.e. the boundary value linearly extrapolated to that stage's time — exactly the WS-RK3 stage
target.

If the update instead **overwrote** the perturbation with an absolute `τ · ∂ₜ` (`τ` = time
since stage start), the per-stage rewind would no longer be cancelled by a matching relative
increment, and each stage's contribution would survive additively. The three stages then
compose to

```
(β₁ + β₂ + β₃) Δt · ∂ₜ = (1/3 + 1/2 + 1) Δt · ∂ₜ = 11/6 Δt · ∂ₜ   per outer step,
```

a secular over-advance of the boundary state at `11/6 ×` the intended rate. It is **invisible
to steady-state / zero-tendency tests** (`∂ₜ = 0 ⇒ no drift`), which is precisely why the
regression uses a nonzero constant `∂ₜ` and asserts an exact `Δt·∂ₜ` advance per outer step on
two consecutive steps:

- `test/substep_boundary_update.jl` → `"Specified-zone composition recovers U⁰ + Δt·∂ₜ over a
  full RK3 step"` (and its vertically-implicit and moisture twins).

The same rewind-plus-increment reasoning governs both the momentum kick
(`_explicit_horizontal_step!`) and the mass / `ρθ` predictors (`_build_predictors!`).

## References

- Wicker, L. J. & Skamarock, W. C. (2002). Time-splitting methods for elastic models using
  forward time schemes. *Mon. Wea. Rev.* **130**, 2088–2097.
- Skamarock, W. C. & Klemp, J. B. (2008). A time-split nonhydrostatic atmospheric model for
  weather research and forecasting applications. *J. Comput. Phys.* **227**, 3465–3485.
- Klemp, J. B., Skamarock, W. C. & Ha, S.-Y. (2018). Damping acoustic modes in compressible
  time-split integration schemes. *Mon. Wea. Rev.* **146**, 1911–1923.
