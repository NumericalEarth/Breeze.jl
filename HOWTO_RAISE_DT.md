# How to raise Δt on the substepping path

This branch (`glw/bw-stable-base`, and the near-twin `glw/bw-stable-no-patch`)
caps at small outer Δt for the dry baroclinic wave on the substepper
(`examples/baroclinic_wave_acoustic.jl` uses `Δt = 10` s) and the moist BW is
limited even more by microphysics stiffness. The branch `glw/hevi-imex-docs`
contains two follow-on fixes that lift both limits and let the dry **and**
moist BW run at the dry advective-CFL `Δt ≈ 225 s` on the 1° grid.

This file summarizes what those fixes do and how to backport them. Both fixes
are isolated to the substepper / time-stepper path and don't touch
microphysics, dynamics tendencies, or the EOS.

## Fix 1 — substepper-side stability at large Δt

**Commit on `glw/hevi-imex-docs`:** `46df021c` *Fix moist BW substepper:
virtual θ, freeze policy, and slow PGF*

This commit contains three independent substepper bug fixes. Two of them
matter for the **dry** Δt cap on this branch:

### 1a. Drop `Gⁿ.ρθ` from the WS-RK3 snapshot/restore

`acoustic_rk3_substep!` in `src/TimeSteppers/acoustic_runge_kutta_3.jl`
currently snapshots `Gⁿ.ρ` AND `Gⁿ.ρθ` at WS-RK3 stage 1 and restores both at
stages 2 and 3 (mimicking MPAS's "freeze tend_rho at rk_step==1"). For dry
runs the snapshot of `Gⁿ.ρθ` is mostly fine — but the per-stage recomputed
value is more accurate at large Δt because it picks up the per-stage
divergence and advection updates. Freezing it locks in stage-1 truncation
error that grows over the outer step.

**Action:** in
`src/TimeSteppers/acoustic_runge_kutta_3.jl::acoustic_rk3_substep!` and the
`AcousticRungeKutta3` constructor, drop `ρθ` from the
`slow_tendency_snapshot` named tuple and from the snapshot/restore branches.
Keep `Gⁿ.ρ` in the snapshot (density has no physics tendency) and keep the
horizontal-PGF snapshot. Move the snapshot/restore logic out of
`acoustic_rk3_substep!` into helpers
(`snapshot_slow_tendencies!` / `restore_slow_tendencies!`) in
`src/TimeSteppers/acoustic_substep_helpers.jl` if you want it factored.

### 1b. Replace the dry-air `linearized_pp` with `pp = p_frozen − p_base`

This is the **bigger** stability win at large Δt.

In `src/CompressibleEquations/acoustic_substepping.jl::_convert_slow_tendencies!`,
the perturbation pressure used to build `tend_w_euler` is currently computed
as

```julia
@inline function linearized_pp(i, j, k, ρθ⁰, πᵣ, pᵣ, Rᵈ, rcv, pˢᵗ)
    Π_base  = πᵣ[i, j, k]
    ρθ_base = pᵣ[i, j, k] / (Rᵈ * Π_base)
    ρθ_p    = ρθ⁰[i, j, k] - ρθ_base
    Π       = (Rᵈ * ρθ⁰[i, j, k] / pˢᵗ)^rcv      # ← dry-air EOS recovery
    return Rᵈ * (Π * ρθ_p + ρθ_base * (Π - Π_base))
end
```

That recovery `Π = (Rᵈ ρθ⁰ / pˢᵗ)^(Rᵈ/cᵥ)` is mathematically equivalent to
`(p / pˢᵗ)^(Rᵈ/cᵖ)` for **dry** air at exact arithmetic, but in Float32 it
introduces a substantial truncation pattern that grows with Δt. At Δt = 60 s
it shows up as a slow startup transient; at Δt = 200 s it kills the
integration within ~25 outer steps even for the dry IC.

The fix is to use the actual cached pressure perturbation directly:

```julia
@inline perturbation_pressure(i, j, k, p_frozen, pᵣ) =
    @inbounds p_frozen[i, j, k] - pᵣ[i, j, k]
```

Both branches of the difference are exact: in any hydrostatic balance,
`∂p_frozen/∂z = -ρ⁰ g` and `∂p_base/∂z = -ρ_base g`, so the
`-∂(pp)/∂z + (-g (ρ⁰ − ρ_base))` cancellation that builds `tend_w_euler`
holds for **any** EOS — dry, moist, total density, dry density, doesn't
matter.

**Action:** thread `substepper.frozen_pressure` into the
`_convert_slow_tendencies!` kernel launch and replace the `linearized_pp`
call with `perturbation_pressure(i, j, k, p_frozen, pᵣ)`. The full
before/after diff is in `46df021c`.

### 1c (moist only) — fix `_prepare_virtual_theta!`

Bonus, only matters for moist runs: the kernel that prepares the
substepper's `virtual_potential_temperature` field discards its mixture
properties and stores the **dry** θ in a field labeled "virtual". The fix
uses `Tᵥ = T · (Rᵐ / Rᵈ)` and stores `θᵥ = Tᵥ / π`. See `46df021c` for the
diff. You can skip this if you only care about the dry Δt cap.

## Fix 2 — operator-split microphysics for moist Δt

**Commit on `glw/hevi-imex-docs`:** `9a357e24` *Operator-split microphysics:
moist BW at advective-CFL Δt*

Even with Fix 1, the moist BW is capped at much smaller Δt than dry because
the in-stage WS-RK3 explicit forward-Euler integration of the microphysics
tendency overshoots when the rain process effective timescale (~10 s) is
shorter than the outer Δt. The auto-conversion + accretion processes
runaway within ~5–25 outer steps at Δt ≥ 60 s.

The fix adds a Strang-style operator split: when
`AcousticRungeKutta3.physics_substeps > 1`, the WS-RK3 dynamics is computed
with the microphysics tendency suppressed (by passing `nothing` for
`microphysics` in the `common_args` NamedTuple, so the kernel-side dispatch
falls through to `grid_microphysical_tendency(::Nothing, ...) = 0`), and
microphysics is then applied as `physics_substeps` explicit forward-Euler
substeps of size `Δt / physics_substeps` between calls to `update_state!`.

**Action:** the commit `9a357e24` adds:

1. A `physics_substeps :: Int` field to `AcousticRungeKutta3` with default
   `1` (existing behavior — backwards compatible).
2. A `physics_split :: Bool = false` kwarg threaded through
   `update_state!`, `compute_tendencies!`, `compute_slow_scalar_tendencies!`,
   and `acoustic_rk3_substep!`. When set, the `common_args` NamedTuple is
   built with `nothing` for the microphysics slot.
3. A new GPU kernel `apply_microphysics_substep!` in
   `src/TimeSteppers/acoustic_substep_helpers.jl` that walks
   `prognostic_field_names(model.microphysics)` to update the thermodynamic
   density (`ρθ`) plus the moisture density and every prognostic
   hydrometeor field (`ρqᶜˡ`, `ρqᶜⁱ`, `ρqʳ`, `ρqˢ`, ...) by forward-Euler
   with `grid_microphysical_tendency(... Val(name), ...)` evaluated at the
   current state.
4. A new branch in `time_step!` for `AcousticRungeKutta3` that, when
   `physics_substeps > 1`, runs the WS-RK3 stages with `physics_split=true`
   and then loops `for _ in 1:N; update_state!; apply_microphysics_substep!`
   after the outer step.

The vapor-only `MoistureMassFractions(qᵛ)` returned by the `nothing`
fallback in `grid_moisture_fractions` introduces an O(qˡ + qⁱ) error in the
EOS recovery during the dynamics-only WS-RK3 stages. For the moist BW the
cloud condensate mass fractions are O(1e-11) to O(1e-7) at the equilibration
phase so the EOS mismatch is sub-K and the operator-splitting error is the
standard O(Δt) Strang-split bound.

## Verification numbers (from `glw/hevi-imex-docs`)

DCMIP2016 baroclinic wave, 1° / Nz = 64 / latitude = (-80, 80), `Float32`,
`PressureProjectionDamping(coefficient = 0.5)`, mixed-phase non-equilibrium
1M ice microphysics with `τ_relax = 200 s`, bulk surface fluxes ON,
bounds-preserving WENO on the moisture tracers.

| Config | Δt | physics_substeps | Result |
|---|---|---|---|
| Dry BW | 225 s | n/a | 20 outer steps clean, max\|w\| ≈ 0.01 m/s |
| Moist BW | 225 s | 24 (Δτ_phys ≈ 9.4 s) | 20 outer steps clean, max\|w\| ≈ 0.007 m/s |
| Moist BW | 160 s | 16 (Δτ_phys = 10 s) | 100 outer steps clean, max\|w\| ≈ 0.014 m/s |
| **Control**: Moist BW | 160 s | **1** (no split) | NaN at step 6, qʳ → 3.0 between steps 4–5 |

## How to backport

### Option A — cherry-pick (cleanest if there are no merge conflicts)

```sh
git fetch origin
git checkout glw/bw-stable-base
git cherry-pick 46df021c   # substepper bug fixes
git cherry-pick 9a357e24   # operator-split microphysics
```

There will likely be conflicts in
`src/CompressibleEquations/acoustic_substepping.jl` and
`src/TimeSteppers/acoustic_runge_kutta_3.jl` because the substepper has
been refactored on `glw/hevi-imex-docs`. Resolve them by keeping the new
versions of the affected functions.

### Option B — manual port

Open both commits side by side and apply the changes file by file. Targets:

- `src/CompressibleEquations/acoustic_substepping.jl` — replace
  `linearized_pp` with `perturbation_pressure` and thread
  `substepper.frozen_pressure` into the kernel launch.
- `src/TimeSteppers/acoustic_runge_kutta_3.jl` — drop `ρθ` from
  `slow_tendency_snapshot`; add `physics_substeps :: Int = 1` field and
  kwarg; thread `physics_split` through `acoustic_rk3_substep!` and
  `time_step!`; add the post-WS-RK3 microphysics fractional-step loop.
- `src/TimeSteppers/acoustic_substep_helpers.jl` — add the
  `apply_microphysics_substep!` host function and
  `_apply_microphysics_substep_kernel!` GPU kernel; thread `physics_split`
  through `compute_slow_scalar_tendencies!`.
- `src/AtmosphereModels/update_atmosphere_model_state.jl` — thread
  `physics_split` through `update_state!` and `compute_tendencies!`; when
  set, build `common_args` with `nothing` for the microphysics slot.
- `examples/moist_baroclinic_wave.jl` — set `Δt = 225` and pass
  `timestepper_kwargs = (; physics_substeps = 24)` to `AtmosphereModel`.

## Caveats

1. The moist BW verification on `glw/hevi-imex-docs` covers ~100 outer
   steps at Δt = 160 s and 20 outer steps at Δt = 225 s — that's the
   equilibration phase, not the BCI peak (which develops around day 7). A
   full 14-day run is the next thing to verify.
2. The operator-split error is O(Δt) — standard Strang-split. For accurate
   moist physics during the BCI peak (large condensation/evaporation
   rates) we may eventually want a true implicit microphysics integrator,
   but the operator split is a much simpler stepping stone.
3. `physics_substeps` should be chosen so `Δτ_phys = Δt / physics_substeps`
   is below the rain process effective timescale (~10 s on this grid).
   For Δt = 225 s, `physics_substeps = 24` gives `Δτ_phys ≈ 9.4 s`.
4. Both fixes are gated on `physics_substeps > 1`, so the existing dry
   substepping path is unchanged when `physics_substeps == 1` (the
   default). Cherry-picking is safe for the dry-only callers on this
   branch — they just won't use the new path.
