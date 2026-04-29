# [Compressible dynamics](@id Compressible-section)

[`CompressibleDynamics`](@ref) solves the fully compressible Euler equations with prognostic
density ``ρ``. The formulation retains acoustic waves and is suitable for problems where full
compressibility is important — global atmospheric flows, baroclinic-wave benchmarks, and
acoustic-mode validation.

## Prognostic equations

The compressible formulation advances density ``ρ``, momentum ``ρ \boldsymbol{u}``, a
thermodynamic variable ``χ`` (see [Governing equations](@ref Dycore-section)), total moisture
``ρ q^t``, and tracers in flux form:

```math
\begin{aligned}
&\text{Mass:} && ∂_t ρ + ∇·(ρ \boldsymbol{u}) = 0 ,\\
&\text{Momentum:} && ∂_t(ρ \boldsymbol{u}) + ∇·(ρ \boldsymbol{u} \boldsymbol{u}) + ∇ p = - ρ g \hat{\boldsymbol{z}} + ρ \boldsymbol{f} + ∇·\boldsymbol{\mathcal{T}} ,\\
&\text{Thermodynamic:} && ∂_t χ + ∇·(χ \boldsymbol{u}) = Π \, ∇·\boldsymbol{u} + S_χ ,\\
&\text{Moisture:} && ∂_t(ρ q^t) + ∇·(ρ q^t \boldsymbol{u}) = S_q .
\end{aligned}
```

Pressure is closed by the moist ideal gas law

```math
p = ρ R^m T ,
```

where ``R^m`` is the mixture gas constant. For the potential-temperature thermodynamics the
prognostic is ``χ = ρ θ`` and ``Π = 0``; for static-energy thermodynamics ``χ = ρ e`` and ``Π``
encodes pressure work.

## Time integration options

`CompressibleDynamics` accepts a `time_discretization` keyword that selects between two
strategies:

- [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization):
  Wicker–Skamarock RK3 outer integration with an inner acoustic substep loop. The outer step
  is bounded by the **advective** CFL (``Δt \sim Δx / U``); the inner substep is bounded by the
  **horizontal acoustic** CFL (``Δτ \sim Δx / c_s``). This is the recommended choice and the
  rest of this page describes its design.

- [`ExplicitTimeStepping`](@ref Breeze.CompressibleEquations.ExplicitTimeStepping): All
  tendencies (advection, pressure gradient, buoyancy) computed together. The time step is
  bounded by the full 3-D acoustic CFL ``Δt < \min(Δx, Δy, Δz)/c_s``.

## Split-explicit time integration

Subcycling fast pressure and gravity-wave dynamics inside an outer Runge–Kutta integration is
the strategy introduced by [Klemp and Wilhelmson (1978)](@cite Klemp1978) and refined for
production models including WRF
([Skamarock and Klemp 1994](@cite SkamarockKlemp1994);
[Wicker and Skamarock 2002](@cite WickerSkamarock2002);
[Klemp, Skamarock, and Dudhia 2007](@cite KlempSkamarockDudhia2007)),
MPAS-Atmosphere ([Skamarock et al. 2012](@cite SkamarockEtAl2012)),
COSMO ([Baldauf et al. 2011](@cite BaldaufEtAl2011)),
and CM1 ([Bryan and Fritsch 2002](@cite BryanFritsch2002)). The presentation here follows the
linear stability analysis of [Baldauf (2010)](@cite Baldauf2010) and the divergence-damping
prescription of [Klemp, Skamarock, and Ha (2018)](@cite KlempSkamarockHa2018), with the
outer/inner coupling stability argument from
[Knoth and Wensch (2014)](@cite KnothWensch2014).

### Slow/fast decomposition and linearization point

Let ``U = (ρ, ρ\boldsymbol{u}, ρθ, ρq^t, …)`` be the prognostic state vector. The right-hand
side is decomposed into

```math
∂_t U = G^{\text{slow}}(U) + G^{\text{fast}}(U; U^0) ,
```

where the **slow operator** ``G^{\text{slow}}`` is evaluated once per outer RK stage from the
current state and held fixed during the substep loop. It contains advective flux divergences,
Coriolis and other body forces, subgrid stresses, microphysics, radiation, boundary fluxes,
and the **slow** part of the pressure gradient evaluated against the discrete hydrostatic
reference state (see [Reference state](@ref reference-state)).

The **fast operator** ``G^{\text{fast}}`` is the linearization of the acoustic and buoyancy
dynamics about the **outer-step-start state** ``U^0 = U^n``. The linearization point is held
fixed across all three RK stages of one outer ``Δt`` — not refreshed per stage. Background
fields snapshotted from ``U^0`` are

```math
ρ^0, \quad (ρθ)^0, \quad p^0 = R^d (ρθ)^0 \! \left(\frac{R^d (ρθ)^0}{p^{st}}\right)^{\!\!R^d/c_v^d}, \quad
Π^0 = (p^0 / p^{st})^κ, \quad θ^0 = (ρθ)^0/ρ^0 .
```

Holding these fixed across the outer step makes the substep operator strictly linear in the
perturbation variables, so the substep stability analysis is independent of the outer RK
machinery.

### Outer scheme: Wicker–Skamarock RK3

The [`AcousticRungeKutta3`](@ref) time stepper is the three-stage Wicker–Skamarock RK3
([Wicker and Skamarock 2002](@cite WickerSkamarock2002)) with stage fractions
``β = (1/3, 1/2, 1)``:

```math
\begin{aligned}
U^{(1)} &= U^n + β_1 \, Δt \, R(U^n) , \\
U^{(2)} &= U^n + β_2 \, Δt \, R(U^{(1)}) , \\
U^{n+1} &= U^n + β_3 \, Δt \, R(U^{(2)}) .
\end{aligned}
```

Each stage resets the prognostic state to ``U^n`` and applies a fraction ``β_k Δt`` of the
slow tendency evaluated at the previous-stage state. The acoustic substep loop is invoked
inside ``R(\cdot)`` to advance the perturbation about ``U^0``.

The acoustic substep size is **constant** across all stages,

```math
Δτ = Δt / N ,
```

while the substep count varies by stage:

```math
N_τ = \max(\mathrm{round}(β_k N), \, 1) ,
```

so the canonical Wicker–Skamarock distribution is ``N/3, N/2, N`` substeps in stages 1, 2, 3
respectively. This keeps the acoustic CFL number identical at every stage. The substep
distribution is selectable via the `substep_distribution` keyword
([`AcousticSubstepDistribution`](@ref Breeze.CompressibleEquations.AcousticSubstepDistribution));
[`MonolithicFirstStage`](@ref Breeze.CompressibleEquations.MonolithicFirstStage) is also
available as an alternative that collapses stage 1 to a single substep of size ``Δt/3``.

### Linearized perturbation equations

Let primes denote perturbations about ``U^0``: ``ρ' = ρ - ρ^0``, ``(ρθ)' = ρθ - (ρθ)^0``, and
``(ρu)' = ρu - (ρu)^0`` (likewise for ``v, w``). The linearized perturbation system advanced
inside the substep loop is

```math
\begin{aligned}
∂_τ ρ'    &+ ∇·(ρ\boldsymbol{u})' = G^s_ρ , \\
∂_τ (ρθ)' &+ ∇·\!\left(θ^0 (ρ\boldsymbol{u})'\right) = G^s_{ρθ} , \\
∂_τ (ρu)' &+ γ R^m \, Π^0 \, ∂_x (ρθ)' = G^s_{ρu} , \\
∂_τ (ρv)' &+ γ R^m \, Π^0 \, ∂_y (ρθ)' = G^s_{ρv} , \\
∂_τ (ρw)' &+ γ R^m \, Π^0 \, ∂_z (ρθ)' + g\, ρ' = G^s_{ρw} .
\end{aligned}
```

Each ``G^s`` is the slow tendency for that variable, held constant across the ``N_τ``
substeps of a given outer RK stage. The acoustic pressure-gradient coefficient
``γ R^m Π^0`` and the temperature-flux factor ``θ^0`` are read from the snapshotted
backgrounds, which is what makes the substep system linear.

### [Reference state and discrete hydrostatic balance](@id reference-state)

The slow vertical PGF ``-∂_z p^0 - ρ^0 g`` is the difference between two large numbers,
each ``\mathcal{O}(10^4)`` in SI units, whose true value is small everywhere and exactly zero
in a rest atmosphere. To preserve this cancellation at the discrete level,
`CompressibleDynamics` accepts a `reference_state` keyword that builds a
[`ExnerReferenceState`](@ref) ``(ρ_r, p_r)`` satisfying

```math
\frac{p_{r,k+1/2} - p_{r,k-1/2}}{Δz_{k}^f} + g \, \overline{ρ_r}^z\big|_{k+1/2} = 0
```

at every face — a discrete hydrostatic balance to machine precision. The slow vertical
momentum tendency uses the *imbalance* ``-∂_z(p^0 - p_r) - (ρ^0 - ρ_r) g`` so that a column
in exact discrete balance contributes zero buoyancy forcing, no matter how steeply
``ρ_r(z)`` and ``p_r(z)`` vary.

### Time discretization of the substep loop

Within each substep of size ``Δτ``, the perturbation update has two phases.

**Forward step — horizontal momenta.**

```math
\begin{aligned}
(ρu)'_{τ+Δτ} &= (ρu)'_τ + Δτ \! \left[ G^s_{ρu} - γ R^m Π^0 \, ∂_x (ρθ)'_τ \right] , \\
(ρv)'_{τ+Δτ} &= (ρv)'_τ + Δτ \! \left[ G^s_{ρv} - γ R^m Π^0 \, ∂_y (ρθ)'_τ \right] .
\end{aligned}
```

**Vertical implicit solve — column tridiag in ``(ρw)'``.** The vertical-momentum, density,
and ``ρθ`` perturbations are coupled through the vertical pressure gradient, the vertical
divergence in the mass and ``ρθ`` equations, and the buoyancy term. To remove the
``Δτ < Δz / c_s`` constraint that an explicit treatment would impose on vertically refined
grids, the vertical block is treated implicitly. Using the off-centering parameter
``ω`` (default `0.65`), the vertical update is split into explicit weight ``1 - ω`` and
implicit weight ``ω``:

```math
\begin{aligned}
(ρw)'_{τ+Δτ} &= (ρw)'_τ + Δτ \, G^s_{ρw} - g\, Δτ \! \left[ (1-ω) ρ'_τ + ω\, ρ'_{τ+Δτ}\right] \\
&\quad - γ R^m Π^0 \, Δτ \! \left[ (1-ω) ∂_z (ρθ)'_τ + ω\, ∂_z (ρθ)'_{τ+Δτ} \right] .
\end{aligned}
```

The horizontal divergence in the mass and ``ρθ`` equations is taken from the just-updated
horizontal momenta ``(ρu)'_{τ+Δτ}, (ρv)'_{τ+Δτ}`` (forward–backward coupling). Substituting
the discrete updates of ``ρ'`` and ``(ρθ)'`` into the ``(ρw)'`` equation yields a
tridiagonal Schur system for ``(ρw)'`` at z-faces, with diagonals proportional to
``ω^2 Δτ^2`` and the local ``γ R^m Π^0`` and ``g`` coefficients. After the tridiag is solved
the perturbations of ``ρ'`` and ``(ρθ)'`` are recovered by back-substitution.

The off-centering parameter ``ω = 1/2`` is classical centered Crank–Nicolson — neutrally
stable for the linearized inviscid system but susceptible to amplification of distributed
floating-point noise through the non-normal substep operator (see
[Stability analysis](@ref stability-analysis)). The default ``ω = 0.65`` adds modest
dissipation; the dimensionless parameter ``ε = 2ω - 1 = 0.3`` quantifies the deviation from
centered.

### Recovery

After ``N_τ`` substeps, the full prognostic state is recovered by addition:

```math
ρ = ρ^0 + ρ' , \qquad ρθ = (ρθ)^0 + (ρθ)' , \qquad ρ\boldsymbol{u} = (ρ\boldsymbol{u})^0 + (ρ\boldsymbol{u})' .
```

There is no Exner-to-``ρθ`` conversion and no convex blend, because the perturbation system
already advances the same prognostic variables as the outer scheme. The slow tendencies
``G^s`` are applied through the substep loop, so the WS-RK3 stage update
``U^{(k)} = U^n + β_k Δt R(U^{(k-1)})`` falls out of the same loop.

## [Klemp 3-D divergence damping](@id klemp-damping)

A bare split-explicit scheme amplifies floating-point noise on a rest atmosphere even when
all the discrete operators are formally consistent (see [Stability analysis](@ref stability-analysis)
for why). [Klemp, Skamarock, and Ha (2018)](@cite KlempSkamarockHa2018) prescribe a per-substep
3-D divergence-damping correction that targets the offending acoustic divergence modes
without affecting balanced flow, building on
[Skamarock and Klemp (1992)](@cite SkamarockKlemp1992) and the linear stability analysis of
[Baldauf (2010)](@cite Baldauf2010).

The discrete divergence proxy is the per-substep change in ``(ρθ)'``, normalized by the
background ``θ^0``:

```math
D_τ \equiv \frac{(ρθ)'_τ - (ρθ)'_{τ-Δτ}}{θ^0} \;≈\; -\, Δτ \, ∇·(ρ\boldsymbol{u})' .
```

After the implicit Schur solve, all three momentum perturbation components pick up the
correction

```math
\begin{aligned}
Δ(ρu)' &= - α_x \, ∂_x D_τ , \\
Δ(ρv)' &= - α_y \, ∂_y D_τ , \\
Δ(ρw)' &= - α_z \, ∂_z D_τ .
\end{aligned}
```

The vertical component is the piece that damps vertical acoustic modes; without it the
column tridiag remains susceptible to the non-normal amplification described below.

### Baldauf anisotropic scaling

The damping diffusivities follow [Baldauf (2010, §2.d)](@cite Baldauf2010) with a
**per-direction** scaling

```math
α_x = β_d \, \frac{Δx^2}{Δτ} , \qquad α_y = β_d \, \frac{Δy^2}{Δτ} , \qquad α_z = β_d \, \frac{Δz^2}{Δτ} .
```

The dimensionless coefficient ``β_d`` is the per-direction explicit-time Courant number of
the damping operator, and is invariant under both ``Δτ`` and grid spacing. The combined 3-D
explicit-time stability bound is

```math
2 β_d ≤ \tfrac{1}{2} \;⟹\; β_d ≤ 0.25 ,
```

so the empirical safe range is ``β_d ∈ [0.05, 0.20]``. The default ``β_d = 0.1`` sits well
below the bound and is the verified pairing for the default ``ω = 0.65``.

For very anisotropic grids (e.g. ``Δz / Δx ≪ 1``) the per-direction scaling produces a
weak vertical damping. Passing a `length_scale = ℓ` keyword to
[`ThermalDivergenceDamping`](@ref Breeze.CompressibleEquations.ThermalDivergenceDamping)
overrides the anisotropic form with an isotropic ``ν = β_d ℓ^2 / Δτ`` applied uniformly to
all three components — useful when a uniform damping coefficient across a stretched grid is
preferred.

## [Stability analysis](@id stability-analysis)

The split-explicit scheme has two distinct sources of instability that interact. Both are
addressed by the same divergence-damping correction.

### 1 — Substep-operator non-normality

Define the substep operator ``\mathcal{U}: U'_τ ↦ U'_{τ+Δτ}`` that advances the perturbation
through one substep at fixed slow tendency. For a stratified ``\bar{θ}(z)`` reference, the
column tridiag has *anti-symmetric* buoyancy off-diagonals (gravity-wave physics — these
*cannot* be symmetrized without breaking the physics) and *asymmetric* PGF off-diagonals
(stratified ``Π^0_z``). The eigenvalues of ``\mathcal{U}`` lie on the unit circle, so

```math
ρ(\mathcal{U}) = 1 ,
```

i.e. the spectral radius is exactly unity. But the operator is **non-normal**:
``\mathcal{U}\mathcal{U}^* ≠ \mathcal{U}^*\mathcal{U}``. Empirically, at ``Δt = 20\,``s and
``ω = 0.55`` on a ``8×8×32`` test grid one measures

```math
\|\mathcal{U}\|_2 ≈ 44 \;≫\; 1 = ρ(\mathcal{U}) .
```

The norm gap means distributed floating-point noise can excite a transient-amplification
subspace — an ``\mathcal{O}(40×)`` per-substep growth in the relevant non-normal sector — even
though every individual eigenmode is neutrally stable. This is the mechanism that destroys a
rest-atmosphere integration at production ``Δt`` when no damping is applied. The Klemp
divergence-damping correction shrinks ``\|\mathcal{U}^k\|`` over enough ``k``, eventually
contracting the unstable subspace.

### 2 — Outer/inner coupling

[Knoth and Wensch (2014)](@cite KnothWensch2014) analyze the coupled stability of an
outer Runge–Kutta scheme with an inner forward-backward substep and show that the
WS-RK3 + substepper combination is **conditionally unstable** for centered Crank–Nicolson
*regardless of how the substep operator itself is constructed*. The mechanism is that any
acoustic perturbation generated inside the substep loop is re-injected on the next outer
stage through the slow-tendency evaluation; for centered CN this re-injection has no
dissipative channel, so the coupled amplification factor exceeds unity for a non-empty
range of acoustic CFL.

This means damping is not an artifact of an imperfect substep discretization that could be
removed by a better scheme: it is a *structural* feature of the WS-RK3 + substepper
combination. The same conclusion follows from the analysis in
[Skamarock and Klemp (1992)](@cite SkamarockKlemp1992),
[Baldauf (2010)](@cite Baldauf2010), and
[Klemp, Skamarock, and Ha (2018)](@cite KlempSkamarockHa2018), which all prescribe
divergence damping as a **required** filter rather than an optional stabilizer.

### 3 — Damping as a filter

The Klemp 3-D damping acts as a wavenumber-controlled filter that targets the divergent
acoustic component while leaving balanced (non-divergent) modes essentially untouched. The
divergence proxy ``D_τ ≈ -Δτ ∇·(ρ\boldsymbol{u})'`` vanishes for any flow that is in
discrete mass balance, so the correction is zero for the rest atmosphere, zero for
hydrostatic balance, and zero for purely solenoidal large-scale flow. Only the divergent
acoustic perturbations — the modes responsible for both the non-normal transient
amplification and the K&W coupling instability — pick up dissipation.

## Stability constraints and practical guidance

Two CFL-like constraints govern the choice of ``Δt`` and the substep count ``N``:

1. **Acoustic substep CFL** (horizontal):

   ```math
   Δτ ≤ \frac{\min(Δx, Δy)}{c_s + |\boldsymbol{u}|} .
   ```

   The vertical implicit solve removes the vertical acoustic CFL constraint entirely.

2. **Advective CFL for the outer step**:

   ```math
   Δt ≤ \frac{\min(Δx, Δy, Δz)}{|\boldsymbol{u}|} .
   ```

   Linearization at the outer-step-start state ``U^0`` decouples the acoustic dynamics from
   the outer integrator, so the outer step is bounded only by the advective CFL.

The default `substeps = nothing` adaptively chooses ``N`` from the horizontal acoustic CFL
each step. For benchmarks this is normally what one wants; setting an explicit integer pins
``Δτ = Δt / N`` for reproducibility.

## Defaults and verification

The default split-explicit configuration is

```julia
SplitExplicitTimeDiscretization(
    forward_weight = 0.65,
    damping = ThermalDivergenceDamping(coefficient = 0.1),
    substep_distribution = ProportionalSubsteps(),
)
```

The pairing ``ω = 0.65, β_d = 0.1`` is verified by:

| Test                                     | Result                                          |
|------------------------------------------|-------------------------------------------------|
| `test/substepper_rest_state.jl`          | Rest atmosphere at machine ``ε`` over 200 outer steps × ``Δt = 20\,``s |
| DCMIP-2016 dry baroclinic wave           | Stable for 12 simulated h × ``Δt = 225\,``s on ``360 × 160 × 64`` lat-lon grid |
| DCMIP-2016 moist baroclinic wave         | Stable for 1 simulated h × ``Δt = 20\,``s on ``360 × 160 × 64`` lat-lon grid with one-moment microphysics |

Removing the damping (``β_d = 0`` or `NoDivergenceDamping()`) restores the
``≈ 1.8×``-per-outer-step rest-atmosphere blow-up at ``Δt = 20\,``s; lowering the
off-centering to ``ω = 0.55`` requires correspondingly larger ``β_d`` to remain stable on
the same tests.

## Comparison with anelastic dynamics

| Property             | [`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) | [`CompressibleDynamics`](@ref) |
|----------------------|-------------------|----------------------|
| Acoustic waves       | Filtered          | Resolved             |
| Density              | Reference ``ρ_r(z)`` only | Prognostic ``ρ(x,y,z,t)`` |
| Pressure             | Solved from Poisson equation | Computed from equation of state |
| Time step            | Limited by advective CFL | Advective CFL (split-explicit) or full acoustic CFL (explicit) |
| Typical applications | LES, mesoscale    | Global flows, baroclinic waves, acoustic studies, validation |
