# Design: Adaptive Implicit Vertical Advection in Breeze

Status: PR 1 implemented (scalars, anelastic `SSPRungeKutta3`) · Target: anelastic `AtmosphereModel`

> **Implemented in PR 1** (this branch): scalar AIVA on the anelastic `SSPRungeKutta3` path.
> Opt in per scalar, e.g. `scalar_advection = (; ρθ = WENO(time_discretization =
> AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)))`. Momentum AIVA is rejected at
> construction; the acoustic/compressible path is unchanged (AIVA is only wired into the
> SSP-RK3 substep). See §7 for the remaining phases.

## 1. Motivation

The explicit vertical advection CFL, `α = |w| Δt / Δz`, limits the time step in regimes
with strong vertical velocities or thin vertical layers: deep convective updrafts/downdrafts,
fast-falling hydrometeors, and stretched grids near the surface or tropopause. In these cases
a single column of fast cells forces a small global `Δt` even though the rest of the domain
is comfortably stable.

Oceananigans recently added **adaptive implicit vertical advection** (AIVA): each cell splits
its vertical velocity into an explicit part that respects a target CFL and an implicit part that
is solved with an unconditionally stable tridiagonal update. This document describes what it
would take to support the same option in Breeze.

The key complication is that Breeze advects in **mass-flux (density-weighted) form**, so
Oceananigans' implicit coefficients — which carry no density weighting — cannot be reused
directly. The bulk of this design is the density-weighted re-derivation.

## 2. Background

### 2.1 How AIVA works in Oceananigans

A scheme opts in through a time-discretization tag carried on the advection scheme:

```julia
WENO(time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5))
```

`AdaptiveVerticallyImplicitDiscretization` (alias `AVID`) holds a target `cfl` and a **mutable
`Δt :: Ref`**. Per cell face, with `α = |w| Δt / Δz`:

- **Explicit velocity scale** `s = min(1, cfl/α)`. Vertical fluxes are multiplied by `s`, so the
  explicit advection CFL never exceeds `cfl`. Horizontal advection stays fully explicit.
- **Implicit velocity** `wⁱ = w (1 − s) = w (1 − cfl/α)` when `α > cfl`, else `wⁱ = 0`.

The implicit remainder is a **first-order upwind** operator assembled into a batched tridiagonal
system. For a field at z-centers, the upwind flux at face `k+1` (top of cell `k`) is

```
F_{k+1} = Az_{k+1} [ max(wⁱ_{k+1}, 0) c_k + min(wⁱ_{k+1}, 0) c_{k+1} ]
```

and `(I − Δt L) cⁿ⁺¹ = c★` yields (Oceananigans `implicit_vertical_advection.jl`):

```
upper_k  =   Δt/V_k · Az_{k+1} · min(wⁱ_{k+1}, 0)
lower_k′ = − Δt/V_k · Az_k     · max(wⁱ_k,    0)
diag_k   =   Δt/V_k · [ Az_{k+1} max(wⁱ_{k+1},0) − Az_k min(wⁱ_k,0) ]    (added to I)
```

These coefficients are **added into the same tridiagonal system** already used for vertically
implicit diffusion (`get_coefficient(..., advection::AIVA, w)` in
`vertically_implicit_diffusion_solver.jl`).

Three time-loop hooks:

1. `update_advection_timestep!(advection, timestepper, clock)` — sets the `Δt` Ref to the next
   substep's effective `Δτ` *before* tendencies, so the `wᵉ` baked into `Gⁿ` matches the `wⁱ`
   used by the following solve. Specialized for `RungeKutta3TimeStepper` and
   `SplitRungeKuttaTimeStepper` only.
2. `compute_tendencies!` — builds explicit tendencies with the scaled `wᵉ` (automatic via flux
   dispatch on `::AVID`).
3. `implicit_step!(field, solver, closure, …, Δt, advection, velocities)` — the **10-arg**
   variant that includes advection diagonals in the solve.

### 2.2 What Breeze already has

- An implicit solver is constructed in `atmosphere_model.jl:221`:
  `implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)` —
  currently `nothing` unless the closure is vertically implicit.
- `SSPRungeKutta3` already calls `implicit_step!` each substep (`ssp_runge_kutta_3.jl:126`),
  but the **8-arg, diffusion-only** form.
- The compressible/acoustic path already builds and uses its own `BatchedTridiagonalSolver`
  for the implicit `(ρw)′` update (`acoustic_substepping.jl:329`).

So the scaffolding (a per-substep implicit solve embedded in the RK substep) exists; it must be
extended to carry advection, and the coefficients must be density-weighted.

### 2.3 Why it is not a drop-in: Breeze's mass-flux form

- **Scalars.** The prognostic variable is `ρc`, but advection is computed on the *specific*
  quantity with `div_ρUc = ∇·(ρ U c)` (`Advection.jl:30`), with `c` passed specific
  (`update_atmosphere_model_state.jl:336`). `div_ρUc` is a **custom operator that does not
  dispatch on `time_discretization`**, so neither the explicit flux-scaling nor the AIVA
  `advective_tracer_flux_z` dispatch fires today.
- **Momentum.** Breeze advects velocity by mass flux: it passes `momentum` (ρ𝐯) as the
  transporting velocity and `velocities.u/v/w` as the advected field into Oceananigans'
  `div_𝐯u/v/w`. The AIVA explicit-scaling *would* fire here, but Oceananigans' implicit
  coefficients build `wⁱ` from `velocities.w`, while Breeze's transporting flux is `ρw` —
  inconsistent by a factor of ρ.

Oceananigans' implicit coefficients (`Δt V⁻¹ Az · w`) carry **no density weighting**. Reusing
them would break consistency with Breeze's explicit `∇·(ρUc)`: the explicit and implicit halves
would not telescope, and mass would not be conserved.

## 3. The density-weighted split (the physics to get right)

We design for **scalars first**; momentum is a later extension (§7).

### 3.1 Continuous and discrete operators

Breeze evolves the density `ρc` of a specific quantity `c`:

```
∂_t (ρc) = − ∇·(ρ 𝐮 c) + (other terms)
```

Discretely (`Advection.jl`), the vertical contribution to `−∇·(ρ𝐮c)` is

```
−V⁻¹ᶜᶜᶜ δzᵃᵃᶜ [ ℑzᵃᵃᶠ(ρ) · Fz(w, c) ]
```

where `Fz(w, c) = _advective_tracer_flux_z(…, w, c)` is the (high-order) volume flux per unit
mass at the z-face. Define the **mass flux at the z-face**

```
m_{k+½} ≡ Az_{k+½} · ℑzᵃᵃᶠ(ρ)_{k+½} · w_{k+½}     [kg s⁻¹]
```

Splitting `w = wᵉ + wⁱ` with the same `s = min(1, cfl/α)` as Oceananigans:

```
wᵉ = s · w        (explicit, CFL-limited)
wⁱ = (1 − s) · w  (implicit, first-order upwind)
```

### 3.2 Explicit part — **no Breeze code required**

The explicit vertical flux is the existing high-order flux with the velocity scaled:

```
Fzᵉ = ℑzᵃᵃᶠ(ρ) · s(i,j,k) · Fz(w, c)
```

**This already happens automatically.** Breeze's `tracer_mass_flux_z` (`Advection.jl:25`) is

```julia
ℑzᵃᵃᶠ(i, j, k, grid, ρ) * _advective_tracer_flux_z(i, j, k, grid, advection, U.w, c)
```

and Oceananigans' underscore flux (`tracer_advection_operators.jl:5`) is

```julia
_advective_tracer_flux_z(i, j, k, grid, scheme, W, c) =
    advective_tracer_flux_z(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), W, c)
```

i.e. it **extracts `time_discretization(scheme)` and dispatches on it**. When `advection` is a scheme
carrying `AdaptiveVerticallyImplicitDiscretization`, the `::AVID` method fires and multiplies by
`s = min(1, cfl/α)`. So `div_ρUc` already yields `ℑz(ρ)·s·Fz(w,c)` for the explicit vertical part,
with horizontal fluxes (`tracer_mass_flux_x/y`) fully explicit — exactly what we want, **without any
edit to `Advection.jl`'s `div_ρUc`**. The only prerequisite is that `td.Δt[]` be populated (§4.1).

The one caveat: this auto-dispatch rides on the high-order `_advective_tracer_flux_*` path. The
`BoundsPreservingWENO` branch of `div_ρUc` (`Advection.jl:42`) uses a separate
`bounded_tracer_flux_divergence_z` that does not consult `time_discretization`; AIVA is therefore
**not supported with `BoundsPreservingWENO`** in the first cut (guard or document).

### 3.3 Implicit part (density-weighted tridiagonal)

First-order upwind on the implicit mass flux. The implicit flux at face `k+½`:

```
Fⁱ_{k+½} = Az_{k+½} ℑzᵃᵃᶠ(ρ)_{k+½} [ max(wⁱ_{k+½}, 0) c_k + min(wⁱ_{k+½}, 0) c_{k+1} ]
```

**Decision — solve for `c` or `ρc`?** The prognostic is `ρc`, but the upwind operator is naturally
written on `c`. Two consistent options:

- **(A) Solve for `c`.** Form `c★ = (ρc)★ / ρ` at centers, solve `(I − Δt Lᶜ) cⁿ⁺¹ = c★`, then
  set `(ρc)ⁿ⁺¹ = ρ · cⁿ⁺¹`. Here `ρ` is the (time-independent, anelastic reference) density at
  centers. The tridiagonal coefficients are the **mass-weighted** ones below divided through by
  `ρ_k` from the `V⁻¹` factor. Cleanest because the upwind operator is on `c`.
- **(B) Solve for `ρc` directly.** Coefficients carry `ℑzᵃᵃᶠ(ρ)/ρ_k` ratios. Algebraically equal
  to (A) for anelastic (ρ = reference, constant in time), but messier.

For the **anelastic formulation ρ = ρ₀(z)** is a fixed reference profile (no time dependence),
so (A) and (B) coincide and (A) is recommended for clarity. The mass-weighted diagonals:

```
upperᶜ_k  =   Δt/V_k · Az_{k+½} · ℑz(ρ)_{k+½}/ρ_k · min(wⁱ_{k+½}, 0)
lowerᶜ_k′ = − Δt/V_k · Az_{k−½} · ℑz(ρ)_{k−½}/ρ_k · max(wⁱ_{k−½}, 0)
diagᶜ_k   =   Δt/V_k · [ Az_{k+½} ℑz(ρ)_{k+½}/ρ_k max(wⁱ_{k+½},0)
                        − Az_{k−½} ℑz(ρ)_{k−½}/ρ_k min(wⁱ_{k−½},0) ]   (added to 1)
```

These are Oceananigans' coefficients with each `Az·w` replaced by `Az · ℑz(ρ)/ρ_k · w`. With
`ρ ≡ 1` they reduce exactly to the Oceananigans form — a useful correctness check.

### 3.4 Conservation / telescoping check

The split must conserve `∑_k V_k ρ_k c_k`. Since `wᵉ + wⁱ = w` and both the explicit (high-order,
already conservative as a flux divergence) and implicit (upwind, written as a flux divergence)
parts are in **flux form** with the *same* face mass flux `Az ℑz(ρ) w`, the discrete column
integral of the combined tendency telescopes to boundary fluxes only. This is the invariant the
tests in §6 must assert numerically.

Note: the implicit upwind part is only first-order in space, so AIVA trades some accuracy in
fast-CFL cells for stability — identical to the Oceananigans behavior and acceptable since it
only activates where `α > cfl`.

## 4. Time-stepper integration

### 4.1 The `Δt` Ref and `update_advection_timestep!`

Breeze's `SSPRungeKutta3` substeps as `u^(m) = (1−α) u^(0) + α(u^(m−1) + Δt G)` with
`α = 1, 1/4, 2/3`. The effective explicit increment in stage `m` is `α Δt`, and the implicit
solve in that substep uses `α Δt` (already passed as `α * Δt` to `implicit_step!`,
`ssp_runge_kutta_3.jl:133`).

**Conservation does not depend on the Ref value.** The split `wᵉ + wⁱ = w` holds pointwise for any
`s`, and both halves are flux-form with the *same* face mass flux, so the column integral telescopes
regardless of what `td.Δt[]` holds (§3.4). Moreover, within a single stage `td.Δt[]` is read by both
`Gⁿ` (built in the previous `update_state!`) and the implicit diagonals using the *same* value, so the
explicit and implicit halves always reference the same `s`. **The Ref value only sets the CFL
threshold — i.e. how much advection is treated implicitly — not correctness or conservation.**

Consequently the **generic upstream fallback** `update_advection_timestep!(::AIVA, ts, clock)` →
`td.Δt[] = clock.last_Δt` is already stable and conservative for SSP-RK3. It merely uses the *full*
`Δt` rather than the substep increment `α Δt`, so stages 2–3 (`α = 1/4, 2/3`) treat slightly more
of the flux implicitly than necessary — stable, marginally more upwind-diffusive. So **PR 1 can ship
with no `update_advection_timestep!` specialization at all**, only the call site (below).

An *optional accuracy refinement* is a `SSPRungeKutta3` specialization that sets the Ref to the
upcoming substep increment so `α` (hence `s`) matches the actual step:

```
td.Δt[] = α_next · Δt
```

where `α_next` is the coefficient of the upcoming stage. Concretely (Breeze-local — see §9):

```julia
function update_advection_timestep!(a::AIVA, ts::SSPRungeKutta3, clock)
    td = time_discretization(a)
    α_next = ssp_next_alpha(ts, clock.stage)   # 1 → 1/4 → 2/3 → (wrap) 1
    td.Δt[] = α_next * clock.last_Δt
    return nothing
end
```

(plus `FluxFormAdvection` / `NamedTuple` forwarding, mirroring Oceananigans
`adaptive_implicit_vertical_advection.jl:119-128`). Call it from
`update_atmosphere_model_state!` before tendencies are recomputed. The acoustic RK3 path needs
its own analogous method (§7).

Open question: the cleanest place to read `α_next`. Options: store the stage coefficients on the
clock, or pass the timestepper's α-tuple. Recommend a small helper `ssp_next_alpha(ts, stage)`.

### 4.2 Building the solver when advection is AIVA

`atmosphere_model.jl:221` currently builds the solver only from the closure. Change to build it
when **either** the closure is vertically implicit **or** any per-field advection scheme
`needs_implicit_solver`:

```julia
implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)
if implicit_solver === nothing && any(needs_implicit_solver, values(advection))
    implicit_solver = BatchedTridiagonalSolver(grid; ...)   # advection-only case
end
```

When both are present, the diffusion and advection diagonals are summed in the coefficient
functions (as Oceananigans does). Reconcile with the acoustic path, which already owns a
tridiagonal solver for `(ρw)′` — for the first PR (scalars on `SSPRungeKutta3`) these do not
overlap, but the interaction must be documented and guarded.

### 4.3 `implicit_step!` call site and the density seam

Upgrade `ssp_rk3_substep!` (`ssp_runge_kutta_3.jl:126`) from the 8-arg to a Breeze variant that
also passes the per-field advection scheme, `model.velocities`, **and the density** `ρ`:

```julia
implicit_step!(u, model.timestepper.implicit_solver,
               model.closure, model.closure_fields, field_index,
               model.clock, fields(model), α * Δt,
               model.advection[name], model.velocities, ρ)
```

Why `ρ` must be threaded explicitly: Oceananigans' `implicit_step!(…, advection, velocities)` calls
`solve!(…, advection, velocities.w)`, and its `get_coefficient(…, advection::AIVA, w)` /
`implicit_advection_*_diagonal(i,j,k,grid,advection,w,Δt,ℓx,ℓy)` functions **take only the velocity —
there is no density in the signature** (`vertically_implicit_diffusion_solver.jl:241`,
`implicit_vertical_advection.jl:59`). So Breeze cannot reuse them as-is for the mass-flux form.
Breeze instead defines its own `implicit_step!`/`get_coefficient` overloads that forward `ρ` and call
the density-weighted coefficients of §3.3. The diffusion diagonals (`_ivd_*_diagonal`) are summed in
exactly as upstream does, so a combined implicit-diffusion + AIVA solve still uses one tridiagonal
system. Breeze already depends on these `get_coefficient`/`solve!` internals for vertically implicit
diffusion, so this extends an existing dependency rather than creating a new one.

For fields with neither an AIVA scheme nor an implicit closure, `implicit_step!` must remain a cheap
no-op, preserving current behavior bit-for-bit.

## 5. New / changed code

| File | Change | Irreducible? |
|------|--------|------|
| `src/Advection.jl` | Explicit part: **no change** (auto-dispatch, §3.2). Add density-weighted implicit coefficients `breeze_implicit_advection_{upper,lower,diagonal}` (the only new physics; reuse `implicit_vertical_velocityᶜᶜᶠ` from Oceananigans for `wⁱ`). | Yes (core) |
| `src/AtmosphereModels/*` (seam) | Breeze-local `get_coefficient(..., advection::AIVA, w, ρ)` + a Breeze `implicit_step!` that threads `ρ` into the existing batched solve, so the density-weighted coefficients are summed with the diffusion diagonals. | Yes (seam) |
| `src/AtmosphereModels/atmosphere_model.jl` | Build `implicit_solver` when `any(needs_implicit_solver, model.advection)`, not only for an implicit closure; combine with the closure solver. | Yes |
| `src/TimeSteppers/ssp_runge_kutta_3.jl` | Switch the existing `implicit_step!` call to the variant that passes per-field advection + `velocities` (+ `ρ`). | Yes (wiring) |
| `src/AtmosphereModels/update_atmosphere_model_state.jl` | One call: `update_advection_timestep!(model.advection, timestepper, clock)` before recomputing tendencies (uses the **generic upstream fallback**; no new method needed). | Yes (one line) |
| `src/TimeSteppers/*` (new method) | *Optional accuracy refinement only:* `update_advection_timestep!(::AIVA, ::SSPRungeKutta3, clock)` setting `α_next·Δt`. Skippable in PR 1. | No |
| imports | Pull `AdaptiveImplicitVerticalAdvection`/`AIVA`, `needs_implicit_solver`, `update_advection_timestep!`, `implicit_vertical_velocityᶜᶜᶠ`, `time_discretization` from `Oceananigans.Advection`/`.TimeSteppers` (explicit-imports compliant). | Yes |

No `Project.toml` `[deps]` changes. Bump `[compat]` Oceananigans lower bound only if a needed
symbol is unexported in the current pin (verify against the resolved version).

## 6. Testing & validation

1. **Reduction to Oceananigans (ρ ≡ 1).** Unit-test the density-weighted coefficients against
   Oceananigans' `implicit_advection_*_diagonal` on a constant-density column; they must match.
2. **Conservation.** A closed column (no-flux top/bottom): assert `∑ V ρ c` is conserved to
   round-off across a step with AIVA active in some cells (§3.4).
3. **Stability above explicit CFL.** Tall thin cells or a strong prescribed updraft with
   `α > 1`: explicit advection blows up; AIVA stays bounded and converges under refinement.
4. **Consistency / order.** With `cfl` large enough that AIVA never activates, results must be
   bitwise (or round-off) identical to the explicit scheme — guarantees opt-in is non-breaking.
5. **GPU.** All new kernel functions `@inline`, `ifelse`-only, allocation-free; run the column
   tests on CUDA.
6. **Doctest.** Constructor wiring: build a model with
   `WENO(time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5))` and `show` the
   model / advection.

## 7. Scope & phasing

- **PR 1 (recommended first):** scalars only, `SSPRungeKutta3` (anelastic) path. Momentum and the
  acoustic/compressible path stay explicit. Opt-in via the per-field `model.advection` scheme, so
  default behavior is unchanged. Delivers the main practical value (θ, qᵗ, hydrometeor fall).
- **PR 2:** momentum AIVA. Requires density-weighted `wⁱ` consistent with the `ρ𝐯`-transport
  convention, and interaction with the anelastic pressure correction (the implicit solve happens
  inside the substep, before `compute_pressure_correction!` — verify the projected field remains
  divergence-free to the intended order).
- **PR 3:** acoustic / compressible path, reconciling with the existing `(ρw)′` tridiagonal
  solver (`acoustic_substepping.jl`).

## 8. Open questions / risks

- **Variable density.** For anelastic, `ρ = ρ₀(z)` is fixed, so options (A)/(B) in §3.3 coincide.
  A future compressible formulation has time-varying `ρ`; the implicit operator would then need
  `ρⁿ` (or `ρ★`) and the `c`-vs-`ρc` choice becomes substantive. Flag, defer.
- **Where to read `α_next`** for the SSP-RK3 `Δt` Ref (§4.1) — needs a clean helper; confirm the
  clock exposes `stage` and `last_Δt` at the point `update_advection_timestep!` runs.
- **Solver ownership** when both implicit diffusion and AIVA are active, vs. the acoustic path's
  pre-existing tridiagonal solver — must not double-allocate or collide.
- **First-order accuracy** in fast-CFL cells is inherent to the method (matches Oceananigans);
  document it as an expected trade-off.

## 9. Minimality review & what belongs upstream

This section answers two questions directly: is the plan minimal, and does any of it belong in
Oceananigans?

### 9.1 Minimality — verified reductions vs. the first draft

Checking the actual Oceananigans seams shrank the plan in two places:

- **The explicit half needs no Breeze code.** Because `_advective_tracer_flux_z` already extracts
  `time_discretization(scheme)` and dispatches (§3.2), the CFL-scaled explicit flux falls out of the
  *existing* `div_ρUc` the moment a scheme carries `AVID`. The earlier "make `div_ρUc` dispatch on
  `::AIVA`" item was unnecessary churn and has been removed.
- **No new time-stepper method is required for a working PR 1.** The generic upstream
  `update_advection_timestep!` fallback is stable and conservative for SSP-RK3; only the CFL
  threshold is slightly conservative in stages 2–3 (§4.1). The `SSPRungeKutta3` specialization is
  demoted to an optional accuracy refinement.

What remains is irreducible and small: (1) density-weighted implicit coefficients — the only genuinely
new physics, and unavoidable because the volume-conserving upstream coefficients omit `ρ`; (2) a thin
`implicit_step!`/`get_coefficient` overload to thread `ρ` into the *existing* batched solve; (3)
build-the-solver-when-advection-needs-it; (4) a one-line `update_advection_timestep!` call. Estimated
net new source: ~60–90 lines, no `div_ρUc` change, no new solver type, reusing the diffusion
tridiagonal machinery and `implicit_vertical_velocityᶜᶜᶠ`/`explicit_velocity_scaleᶜᶜᶠ` from upstream.

The split is opt-in per `model.advection[name]` scheme; with explicit schemes every new branch is a
fallback no-op, so default behavior is unchanged (asserted by test §6.4).

### 9.2 What belongs upstream (one candidate, non-blocking)

There is exactly one piece that would be *cleaner* upstream: Oceananigans'
`implicit_advection_{upper,lower,diagonal}` and the `get_coefficient(…, advection::AIVA, w)` seam
hardcode the volume-conserving form (`Δt · V⁻¹ · Az · w`) with **no hook for a face weight or mass
flux**. Generalizing them to accept an optional face-area/density weight (defaulting to unity, so the
current behavior is unchanged) would let anelastic/compressible models reuse them directly, and Breeze
would then pass `ℑzᵃᵃᶠ(ρ)` instead of maintaining its own copies.

**But it should not block this work, and arguably should not be done yet:**

- Oceananigans' `NonhydrostaticModel` is volume-conserving (effectively `ρ ≡ 1`); it has **no density
  concept in this path and no consumer** for the generalized form. Upstreaming now would add an
  abstraction with a single external user (Breeze).
- Per `AGENTS.md` ("when something would be better in Oceananigans, add a detailed TODO note"), the
  right move is to implement the density-weighted coefficients in Breeze now with a TODO pointing at
  `implicit_vertical_advection.jl`, and propose the upstream generalization once the Breeze
  implementation has proven the interface (ideally alongside a compressible formulation that gives
  upstream a second, time-varying-`ρ` use case to design against).

Everything else is necessarily Breeze-local:

- `update_advection_timestep!(::SSPRungeKutta3, …)` dispatches on a Breeze-only time-stepper type, so
  it cannot live upstream. The reusable generic method already exists upstream and Breeze inherits it.
- The mass-flux `div_ρUc` operator and the `ρ`-threading `implicit_step!` are Breeze's anelastic
  formulation; they have no meaning in Oceananigans' volume-conserving core.

### 9.3 Residual risk to the "minimal" claim

The one place the estimate could grow is the `get_coefficient`/`solve!` seam (§4.3): if threading `ρ`
through cleanly turns out to require touching more of the upstream `solve!` signature than expected,
the lower-churn alternative is a small Breeze-owned tridiagonal assembly for the advection operator,
solved as a second pass after the diffusion solve (additive operator splitting between diffusion and
advection). That is more lines but avoids overloading upstream internals — decide once the seam is
prototyped.
</content>
