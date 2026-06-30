# Design: Vector-invariant momentum for the compressible core

**Status:** Draft / design phase — formulation locked against MPAS source
**Branch:** `glw/compressible-vector-invariant` (off `main`)
**Author:** design notes (Greg + Claude)

## 1. Goal and scope

Add an MPAS-Atmosphere–style **vector-invariant** option for the nonlinear
*horizontal* momentum advection in `CompressibleEquations`, as an alternative to
the current flux-form divergence `∇·(ρ𝐯⊗u)`.

The prognostic variable **stays the coupled momentum** `ρ𝐮 = (ρu, ρv, ρw)`
(`AM.momentum`) — matching both Breeze's conservation-form architecture and MPAS,
which also prognoses coupled momentum. VI changes only how the nonlinear term is
*discretized*, not the prognostic variable, and only for the **horizontal**
components.

**Non-goals.** Not switching to a velocity prognostic. Not removing flux-form
advection (VI is opt-in via `momentum_advection`). Not applying VI to vertical
momentum `ρw` — MPAS keeps `w` in flux form (§3.2), so we do too.

## 2. What MPAS actually does (verified against source)

**This section corrects two earlier wrong claims of mine.** I had asserted (a)
that MPAS prognoses *velocity*, and (b) that the "algebraic split"
`ρ(𝐯·∇)𝐮 + 𝐮 ∇·(ρ𝐯)` is non-conservative and "no production model ships it." Both
are wrong. Reading `MPAS-Model/src/core_atmosphere/dynamics/mpas_atm_time_integration.F`
(`atm_compute_dyn_tend_work`):

- MPAS prognoses **coupled momentum** (`ru = ρu` on edges), and
- the horizontal momentum tendency **is exactly the split**, with the
  `𝐮 ∇·(ρ𝐯)` correction term included explicitly.

The relevant assembly (`tend_u`, lines ~6517, 6533–6562):

```fortran
! vertical transport of u  (flux form, 3rd-order)
tend_u(k,iEdge) = - rdzw(k)*(wduz(k+1)-wduz(k))            ! wduz = ½(rw)·u  →  ∂z(ρw·u)

! nonlinear Coriolis q : PV flux of velocity, pv_edge = (ζ+f) with 1/ρ REMOVED
q(k) += weightsOnEdge(j) * u(k,eoe) * ½(pv_edge(iEdge)+pv_edge(eoe))

! horizontal KE gradient + vorticity, ×ρ, MINUS the mass-divergence correction
tend_u(k,iEdge) += rho_edge(k,iEdge) * ( q(k) - (ke(cell2)-ke(cell1))*invDcEdge ) &
                 - u(k,iEdge) * ½(h_divergence(cell1)+h_divergence(cell2))
```

with `h_divergence = ∇ₕ·(ρ𝐮)` (sum of edge mass fluxes `ru` / cell area, lines
6421–6445) and `ke = ½|𝐮ₕ|²` at cell centres (line 7223).

Two design-critical observations from the source:

1. **The `𝐮 ∇·(ρ𝐯)` term is real and explicit** (`- u·h_divergence`). So the split
   *is* the production scheme.
2. **MPAS deliberately dropped the `1/ρ` from potential vorticity.** Verbatim
   comment: *"the original definition of pv_edge had a factor of 1/density. We have
   removed that factor given that it was not integral to any conservation property
   of the system."* So MPAS-A is **not** the strict potential-enstrophy-conserving
   TRiSK form. It uses the absolute-vorticity flux of *velocity*, multiplied by
   `ρ_edge` outside — which equals `(ζ+f) × (ρ𝐮)` in the continuum, i.e. the
   absolute vorticity acting on the *mass flux*, but discretized as
   `ρ_edge·(velocity vorticity flux)` rather than `q·(mass flux)` with `q=(ζ+f)/ρ`.

So the earlier "Tier A (reuse-split) vs Tier B (pure PV-flux of mass flux)"
framing was a false dichotomy. **MPAS is the reuse-split**, and the pure
PV-flux-of-mass-flux form is *not* what MPAS ships.

## 3. The formulation to implement

### 3.1 Horizontal momentum (vector-invariant split)

Decompose `−∇·(ρ𝐮⊗u) = −ρ(𝐮·∇)u − u ∇·(ρ𝐮)` and compute, for the `ρu` tendency:

```
d(ρu)/dt|adv = − ρ_edge · [ (ζ+f)-flux + ∂ₓK ]      horizontal VI advective part: ρ·(𝐮·∇)u|ₕ
             − ∂z(ρw · u)                            vertical advection, FLUX FORM (incl. its share of u·∇·(ρ𝐯))
             − u · ∇ₕ·(ρ𝐮)                           horizontal mass-divergence correction
```

(and symmetrically for `ρv`). This sums identically to `−∇·(ρ𝐮⊗u)` in the
continuum; the split just routes the horizontal advective part through the
vorticity+KE (vector-invariant) discretization while keeping the vertical
advection and the horizontal-divergence correction in flux form.

- `(ζ+f)` is absolute vertical vorticity (Coriolis folded in — no separate `f×U`).
- `K = ½(u²+v²)` at cell centres (horizontal KE; MPAS's active `ke`).
- `∇ₕ·(ρ𝐮)` is the horizontal mass-flux divergence at cell centres, interpolated
  to the velocity face.

### 3.2 Vertical momentum (`ρw`) — UNCHANGED, flux form

`tend_w` in the source (lines 6658–6777) is pure flux form:

```
d(ρw)/dt|adv = − ∇ₕ·(ρ𝐮 · w)        horizontal: ru_edge_w · w_reconstructed (3rd-order upwind)
             − ∂z(ρw · w)            vertical: wdwz = ½(rw)·w (flux3)
```

No vorticity, no `ζ₁/ζ₂`, no `U_dot_∇w`. **This resolves the earlier "novel /
highest-risk vertical VI" concern: there is none.** Breeze's existing flux-form
`div_𝐯w` already implements exactly this, so the `z_momentum_flux_divergence`
path needs **no change** for the VI scheme.

### 3.3 Continuity (unchanged)

`∂ρ/∂t = −∇ₕ·(ρ𝐮) − ∂z(ρw)` (source line 6466). The same `∇ₕ·(ρ𝐮)` computed for
continuity feeds the §3.1 correction term — compute once, reuse.

## 4. Design decisions — resolved

1. **Coupled-momentum bookkeeping** → **the split**, with explicit `−u ∇ₕ·(ρ𝐮)`.
   Horizontal advective part in VI (vorticity + KE), vertical advection in flux
   form, horizontal-divergence correction explicit. (Was the open crux; resolved.)
2. **PV density factor** → **drop the `1/ρ`**; use `ρ_edge·(absolute-vorticity flux
   of velocity)`, per MPAS. Do not chase strict potential-enstrophy TRiSK.
3. **Coriolis** → fold `f` into the absolute vorticity on the VI path; suppress the
   separate `x_f_cross_U`/`y_f_cross_U`. Flux-form path keeps `f×U`.
4. **Vertical momentum / KE** → `w` stays flux form (§3.2); `K = ½(u²+v²)`.

Remaining open items:

- **Carrier density (#5).** On `main` the prognostic is total `ρ`
  (`compressible_dynamics.jl:39`). MPAS uses dry `ρᵈ` (the in-flight
  `glw/compressible-dry-density` branch). Write operators against a single
  swappable `mass_density` accessor and reconcile at merge.
- **Vertical-momentum & divergence-correction reconstruction (#6).** A plain
  `VectorInvariant`'s `vertical_advection_scheme` is an `EnergyConserving`/WENO
  *velocity* operator, not an `advective_momentum_flux` reconstruction, so it can't
  drive the flux-form `ρw` path (verified: `div_𝐯w` `MethodError`s on it). The
  scaffold uses `Centered()` as a placeholder. The real choice — and the upwinding
  of the `−u ∇·(ρ𝐮)` correction (TOTAL vs horizontal divergence) — is the
  **`CompressibleWENOVectorInvariant`** design (see §11, pending Greg's notes).

## 5. Reuse from Oceananigans

`Oceananigans.Advection` VI operators give the *velocity-form* horizontal advective
part directly:

- `horizontal_advection_U/V` — absolute(relative)-vorticity flux of velocity,
  energy-conserving variants (Hollingsworth-aware).
- `bernoulli_head_U/V` — the `∂ₓK`, `∂yK` gradient (`K=½(u²+v²)`).
- `ζ₃ᶠᶠᶜ`, the conserving vorticity stencils, the `VectorInvariant` /
  `WENOVectorInvariant` scheme structs + upwinding.

`U_dot_∇u(scheme::VectorInvariant, U)` bundles `horizontal_advection_U +
vertical_advection_U + bernoulli_head_U`. We want the **horizontal** pieces only
(`horizontal_advection_U + bernoulli_head_U`), then `×ρ_edge`, because MPAS does
the vertical advection in flux form carried by the mass flux `ρw` (Oceananigans'
`vertical_advection_U` is carried by velocity `w`, not `ρw`). So either call the
horizontal sub-operators directly, or use `U_dot_∇u` and replace its vertical term.

What is **not** in Oceananigans and must be added in Breeze:
- the `ρ_edge` weighting of the horizontal VI part,
- the `−u ∇ₕ·(ρ𝐮)` correction term,
- folding `f` into the vorticity for the VI path (or relying on the relative-vort
  operator + a separate planetary-vorticity contribution inside the flux),
- the flux-form vertical advection of `ρu` carried by `ρw` (Breeze's existing
  `div_𝐯w`-style z-flux, restricted to the vertical direction).

## 6. Mapping to Breeze code

Seam: `src/AtmosphereModels/dynamics_kernel_functions.jl`.

- Add `x/y_momentum_flux_divergence(…, advection::VectorInvariant, …)` methods
  implementing §3.1. `z_momentum_flux_divergence` keeps the existing flux-form
  method (§3.2) — no VI dispatch needed for `w`.
- In `x/y_momentum_tendency`, suppress `x/y_f_cross_U` on the VI path (Decision #3)
  — gate on scheme type or route Coriolis through the advection dispatch.
- `ρ` (carrier) and the horizontal mass-flux divergence `∇ₕ·(ρ𝐮)` must be reachable
  in the kernel. `ρ` already arrives via `dynamics`; the divergence can be computed
  inline from `momentum` with the C-grid `δ`/`V⁻¹` operators (cf. `Advection.jl`
  `div_ρUc`), or precomputed once (it is also needed by continuity).
- `atmosphere_model.jl`: `VectorInvariant` is an `AbstractAdvectionScheme`, so it
  flows through `validate_momentum_advection` / `materialize_advection` /
  `adapt_advection_order`. Verify `validate_model_halo` accounts for the wider VI
  stencil.

## 7. Constraints

- **Split-explicit substepping.** The VI nonlinear term is a *slow* tendency
  (once per RK stage), consistent with MPAS acoustic substepping and Breeze's
  `SplitExplicitTimeDiscretization`. Keep it out of the fast acoustic loop.
- **Hollingsworth.** Use Oceananigans' energy-conserving consistent KE/vorticity
  discretization; keep it a validation target (§9). (MPAS mitigates via its mesh +
  consistent operators; we inherit the rectangular-grid risk.)
- **GPU/kernels.** New operators `@inline`, type-stable, allocation-free, `ifelse`
  not `?:`, literal zeros — per `AGENTS.md`.
- **Terrain-following coordinates.** Interaction with `TerrainMetrics` /
  contravariant `w̃` deferred; document.

## 8. Staged implementation plan

1. **This doc + decisions.** Done — formulation locked against MPAS source.
2. **Scaffolding + dispatch seam.** ✅ DONE. `VectorInvariant`-aware
   `*_momentum_flux_divergence` in `dynamics_kernel_functions.jl`: x/y route to
   `ρ_face · U_dot_∇u/∇v` (velocity-form VI advective part × ρ); z stays flux form
   via `Centered()` placeholder. Coriolis left as the separate `f×U` term. Verified
   on CPU: a `momentum_advection = VectorInvariant()` compressible model constructs
   (validation/materialization accept it) and steps to finite results matching
   flux-form to ~1e-9 over one small step. NOTE: the advective part is real (not
   throwaway) — step 3 ADDS the `−u ∇·(ρ𝐮)` correction to it.
3. **Horizontal VI split (u, v).** ✅ IMPLEMENTED in
   `src/AtmosphereModels/vector_invariant_advection.jl` as a `CompressibleVectorInvariant{S,D}`
   wrapper (inner Oceananigans scheme `S` + divergence trait `D`), per the agreed
   API. Both flavors are wired:
   - `HorizontalDivergence` (MPAS-faithful): `ρ_face·(horizontal_advection_U +
     bernoulli_head_U)` + flux-form vertical `∂z(ρw·u)` + `−u ∇ₕ·(ρ𝐮)` (`div_xyᶜᶜᶜ`).
   - `ThreeDimensionalDivergence`: `ρ_face·U_dot_∇u` + `−u ∇·(ρ𝐮)` (`divᶜᶜᶜ`, centered).
   Wrapper integrates via `required_halo_size_{x,y,z}` delegation +
   `adapt_advection_order` no-op; vertical reconstruction is a `Centered()`
   placeholder (`compressible_vi_vertical_scheme`). z-momentum stays flux form for
   both. Smoke-tested on CPU: constructs, steps finite; horizontal and 3D flavors
   agree bit-for-bit when `w=0` (correct), both ≈ flux-form over a small step.
   API mirrors Oceananigans (per review): `CompressibleVectorInvariant(FT; divergence,
   kwargs...)` forwards sub-scheme kwargs to `VectorInvariant` (no positional scheme
   arg); `CompressibleWENOVectorInvariant(FT; divergence=ThreeDimensionalDivergence(),
   kwargs...)` mirrors `WENOVectorInvariant`. Flavor dispatch via
   `CompressibleVectorInvariant{<:Any, <:HorizontalDivergence}` aliases.
   STILL TODO: rigorous validation (reduction with nonzero `w`, energy
   conservation, balanced vortex); the WENO path — `CompressibleWENOVectorInvariant`
   *constructs* and passes halo validation, but stepping hits an Oceananigans-
   internal `newton_div(::Type{Nothing}, …)` in the WENO reconstruction (nonhydrostatic
   context) — to resolve alongside the WENO-upwinded total-divergence (Greg's notes);
   curvilinear metric terms; sphere `validate_momentum_advection`.
4. **Coriolis folding.** Fold `f` into vorticity on the VI path; verify geostrophic
   balance / steady geostrophic mode.
5. **Conservation + Hollingsworth tests; docs.** Promote this doc into
   `docs/src/developer/`. (Note: `w` needs no work — already flux form.)

## 9. Validation plan

- **Reduction check:** constant `ρ`, `f=0` → VI horizontal term matches flux-form
  to discretization order.
- **Energy conservation** over a closed periodic box (VI's retained property; note
  MPAS does *not* claim enstrophy conservation here).
- **Hollingsworth:** sheared-jet / resting-atmosphere stability test.
- **Geostrophic adjustment / balanced vortex:** steady mode stays steady.
- Existing compressible regressions (dry rising bubble, etc.) unchanged with the
  flux-form default.

## 10. References

- Skamarock et al. (2012), MWR 140(9) — MPAS-A equations.
- **MPAS source:** `MPAS-Model/src/core_atmosphere/dynamics/mpas_atm_time_integration.F`,
  `atm_compute_dyn_tend_work` — `tend_u` lines ~6517/6533/6560, `tend_w` ~6658/6777,
  `h_divergence` 6421–6445, continuity 6466. (Primary source for §2–§3.)
- Ringler et al. (2010) / Thuburn et al. (2009) — TRiSK (note: MPAS-A relaxes the
  strict PV form, §2 obs. 2).
- Hollingsworth et al. (1983) — rectangular-C-grid VI instability.
- Oceananigans `src/Advection/vector_invariant_advection.jl` — reusable operators.

## 12. Validation status — global baroclinic wave (GPU)

Verified on an H100 at the full example resolution (360×150×32, Float32, MPAS-style
`SplitExplicitTimeDiscretization`):

- **Horizontal flavor** (`CompressibleWENOVectorInvariant(; divergence=HorizontalDivergence())`):
  **stable** over a 2-day integration — `max|u|` holds at ≈28 m/s (steady zonal jet,
  no growth), `max|w|` ≈ 0.006–0.013 m/s and slightly decaying (no grid-scale noise).
  This is the configuration the baroclinic-wave example now uses.
- **3D flavor** (`CompressibleWENOVectorInvariant()`, `ThreeDimensionalDivergence`):
  **unstable** — NaN by iteration 10. Cause: the 3D flavor multiplies Oceananigans'
  full `U_dot_∇u` (which for WENO already carries an internal divergence-upwinding
  term) by ρ and *also* adds the explicit `−u ∇·(ρ𝐮)` total-divergence correction,
  double-counting/unbalancing the divergence terms. The 3D flavor needs the proper
  WENO-upwinded total-divergence operator (the `CompressibleWENOVectorInvariant`
  divergence-scheme work, pending notes) rather than naive `ρ·U_dot_∇u + divᶜᶜᶜ`;
  until then it should be treated as experimental.

Note: running any compressible `SplitExplicitTimeDiscretization` on the resolved
Oceananigans (0.109.2) first required a pre-existing-compat fix — `NormalFlow` →
`Open` in `acoustic_substepping.jl` (Oceananigans renamed the open-BC
classification). This is independent of the vector-invariant work.
