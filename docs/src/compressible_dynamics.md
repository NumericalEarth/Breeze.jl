# [Compressible dynamics](@id Compressible-section)

[`CompressibleDynamics`](@ref) solves the fully compressible Euler equations with prognostic
total density ``ρ`` (including dry air, vapor, and condensate). The formulation retains acoustic waves and is suitable for problems where full
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

On spherical grids, Breeze can use either the full nontraditional spherical Coriolis
operator or the traditional approximation. `SphericalCoriolis()` includes the horizontal
component of planetary rotation and therefore the ``2Ω cosφ`` coupling between zonal and
vertical momentum. `HydrostaticSphericalCoriolis()` omits that coupling. This choice is
independent of whether the dynamics evolve prognostic vertical momentum: nonhydrostatic
models may still use the traditional approximation when the benchmark or forcing assumes it.

### Slow/fast decomposition and linearization point

Let ``U = (ρ, ρ\boldsymbol{u}, ρθ, ρq^t, …)`` be the prognostic state vector. The right-hand
side is decomposed into

```math
∂_t U = G^{\text{slow}}(U) + G^{\text{fast}}(U; U^L) ,
```

where the **slow operator** ``G^{\text{slow}}`` is evaluated once per outer RK stage from the
current RK predictor state and held fixed during that stage's substep loop. It contains
advective flux divergences, Coriolis and other body forces, subgrid stresses, microphysics,
radiation, and boundary flux tendencies. The ordinary momentum-tendency kernels run in a
mode that excludes pressure-gradient and buoyancy forces; those forces are reintroduced in
the acoustic substep loop as a stage-entry background contribution plus a linearized
perturbation contribution.

The **fast operator** ``G^{\text{fast}}`` is the linearization of the acoustic and buoyancy
dynamics about the **RK stage-entry state** ``U^L``. The cached background fields are
refreshed before every RK stage:

```math
ρ^L, \quad (ρθ)^L, \quad p^L, \quad
Π^L = (p^L / p^{st})^κ, \quad θ^L = (ρθ)^L/ρ^L, \quad γ^m R^m\big|_L .
```

The outer-step-start state ``U^n`` is also stored. Stages 2 and 3 initialize perturbations
with the rewind term ``U^n - U^L`` so that the full state at the beginning of every
substep loop is still ``U^n`` while the linearized coefficients come from the current RK
predictor. This is Breeze's current stage-rewind formulation for preserving the
Wicker-Skamarock RK3 invariant. It should not be read as identical to every production
small-step implementation: for example, MPAS-Atmosphere stores stage-state increments
with different bookkeeping rather than literally initializing these Breeze perturbation
fields to ``U^n - U^L``.

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

Each stage applies a fraction ``β_k Δt`` of the slow tendency evaluated at the
previous-stage state. The acoustic substep loop is invoked inside ``R(\cdot)`` to advance
perturbations about the current stage-entry state, initialized with the rewind term
described above.

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

Let primes denote perturbations about ``U^L``: ``ρ' = ρ - ρ^L``, ``(ρθ)' = ρθ - (ρθ)^L``, and
``(ρu)' = ρu - (ρu)^L`` (likewise for ``v, w``). The linearized perturbation system advanced
inside the substep loop is

```math
\begin{aligned}
∂_τ ρ'    &+ ∇·(ρ\boldsymbol{u})' = G^s_ρ , \\
∂_τ (ρθ)' &+ ∇·\!\left(θ^L (ρ\boldsymbol{u})'\right) = G^s_{ρθ} , \\
∂_τ (ρu)' &+ ∂_x p^L + ∂_x \left(C^L (ρθ)'\right) = G^s_{ρu} , \\
∂_τ (ρv)' &+ ∂_y p^L + ∂_y \left(C^L (ρθ)'\right) = G^s_{ρv} , \\
∂_τ (ρw)' &+             ∂_z \left(C^L (ρθ)'\right) + g\, ρ' = G^s_{ρw} .
\end{aligned}
```

Each ``G^s`` is the slow tendency for that variable, held constant across the ``N_τ``
substeps of a given RK stage. For vertical momentum, ``G^s_{ρw}`` is assembled by adding the
stage-entry vertical pressure-gradient and buoyancy imbalance,
``-∂_z(p^L - p_r) - g(ρ^L - ρ_r)``, to the slow non-pressure tendency. The acoustic
linearized pressure coefficient
``C^L = γ^m R^m\big|_L Π^L`` and the temperature-flux factor ``θ^L`` are cached for the
stage, which is what makes each stage's substep system linear.

### [Reference state and discrete hydrostatic balance](@id reference-state)

The slow vertical PGF ``-∂_z p^L - ρ^L g`` is the difference between two large numbers,
each ``\mathcal{O}(10^4)`` in SI units, whose true value is small everywhere and exactly zero
in a rest atmosphere. To preserve this cancellation at the discrete level,
`CompressibleDynamics` accepts a `reference_state` keyword that builds a
[`ExnerReferenceState`](@ref) ``(ρ_r, p_r)`` satisfying

```math
\frac{p_{r,k+1/2} - p_{r,k-1/2}}{Δz_{k}^f} + g \, \overline{ρ_r}^z\big|_{k+1/2} = 0
```

at every face — a discrete hydrostatic balance to machine precision. The slow vertical
momentum tendency uses the *imbalance* ``-∂_z(p^L - p_r) - (ρ^L - ρ_r) g`` so that a column
in exact discrete balance contributes zero buoyancy forcing, no matter how steeply
``ρ_r(z)`` and ``p_r(z)`` vary.

### Time discretization of the substep loop

Within each substep of size ``Δτ``, the perturbation update has two phases.

**Forward step — horizontal momenta.**

```math
\begin{aligned}
(ρu)'_{τ+Δτ} &= (ρu)'_τ + Δτ \! \left[ G^s_{ρu} - ∂_x p^L - ∂_x \left(C^L (ρθ)'_τ\right) \right] , \\
(ρv)'_{τ+Δτ} &= (ρv)'_τ + Δτ \! \left[ G^s_{ρv} - ∂_y p^L - ∂_y \left(C^L (ρθ)'_τ\right) \right] .
\end{aligned}
```

For the first substep of a multi-substep RK stage, Breeze follows the MPAS
forward-backward sequence and omits only the *acoustic perturbation* pressure
gradient:

```math
(ρu)'_{τ+Δτ} = (ρu)'_τ + Δτ \left[ G^s_{ρu} - ∂_x p^L \right], \qquad
(ρv)'_{τ+Δτ} = (ρv)'_τ + Δτ \left[ G^s_{ρv} - ∂_y p^L \right].
```

The perturbation pressure-gradient term ``∇(C^L (ρθ)')`` is applied on subsequent
substeps, after the mass and thermodynamic perturbations have been advanced
once. If a stage has only one acoustic substep, the perturbation pressure
gradient is applied immediately so the stage still includes the fast force.
The frozen ``∇p^L`` term is applied on every substep because the slow tendency
mode excludes pressure gradients. This matches MPAS's split: the first small
step skips the perturbation pressure gradient inside `atm_advance_acoustic_step`,
while the large-step pressure-gradient tendency is already present in
`tend_u_euler`.

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
&\quad - Δτ \! \left[ (1-ω) ∂_z \left(C^L (ρθ)'_τ\right) + ω\, ∂_z \left(C^L (ρθ)'_{τ+Δτ}\right) \right] .
\end{aligned}
```

The horizontal divergence in the mass and ``ρθ`` equations is taken from the just-updated
horizontal momenta ``(ρu)'_{τ+Δτ}, (ρv)'_{τ+Δτ}`` (forward–backward coupling). Substituting
the discrete updates of ``ρ'`` and ``(ρθ)'`` into the ``(ρw)'`` equation yields a
tridiagonal Schur system for ``(ρw)'`` at z-faces, with diagonals proportional to
``ω^2 Δτ^2`` and the local ``C^L = γ R^m Π^L`` and ``g`` coefficients. Importantly, the
pressure perturbation is ``p' = C^L (ρθ)'`` at cell centers, so the discrete pressure
gradient is the gradient of this product, not ``C^L`` interpolated to a face times
``∂(ρθ)'``. After the tridiag is solved the perturbations of ``ρ'`` and ``(ρθ)'`` are
recovered by back-substitution.

The off-centering parameter ``ω = 1/2`` is classical centered Crank–Nicolson — neutrally
stable for the linearized inviscid system but susceptible to amplification of distributed
floating-point noise through the non-normal substep operator (see
[Stability analysis](@ref stability-analysis)).
A fully implicit backward Euler scheme is obtained with ``ω = 1`` and offers the most dissipation.
The default ``ω = 0.65`` adds modest dissipation; the dimensionless parameter ``ε = 2ω - 1 = 0.3`` quantifies the deviation from centered.

### Recovery

After ``N_τ`` substeps, the full prognostic state is recovered by addition:

```math
ρ = ρ^L + ρ' , \qquad ρθ = (ρθ)^L + (ρθ)' , \qquad ρ\boldsymbol{u} = (ρ\boldsymbol{u})^L + (ρ\boldsymbol{u})' .
```

There is no Exner-to-``ρθ`` conversion and no convex blend, because the perturbation system
already advances the same prognostic variables as the outer scheme. The slow tendencies
``G^s`` are applied through the substep loop, so the WS-RK3 stage update
``U^{(k)} = U^n + β_k Δt R(U^{(k-1)})`` falls out of the same loop.

## [Klemp divergence damping](@id klemp-damping)

A bare split-explicit scheme amplifies floating-point noise on a rest atmosphere even when
all the discrete operators are formally consistent (see [Stability analysis](@ref stability-analysis)
for why). [Klemp, Skamarock, and Ha (2018)](@cite KlempSkamarockHa2018) prescribe a per-substep
divergence-damping correction that targets the offending acoustic divergence modes
without affecting balanced flow, building on
[Skamarock and Klemp (1992)](@cite SkamarockKlemp1992) and the linear stability analysis of
[Baldauf (2010)](@cite Baldauf2010).

The discrete divergence proxy is the per-substep change in ``(ρθ)'``, normalized by the
stage-entry ``θ^L`` cache used by the acoustic transport equation:

```math
D_τ \equiv \frac{(ρθ)'_τ - (ρθ)'_{τ-Δτ}}{θ^L} \;≈\; -\, Δτ \, ∇·(ρ\boldsymbol{u})' .
```

After the implicit Schur solve, the horizontal momentum perturbation components pick up the
explicit correction

```math
\begin{aligned}
Δ(ρu)' &= - γ_h \, ∂_x D_τ , \\
Δ(ρv)' &= - γ_h \, ∂_y D_τ .
\end{aligned}
```

This horizontal divergence damping is applied by default. Breeze can also fold the vertical component into the column tridiag by setting
`damp_vertical = true` on [`ThermalDivergenceDamping`](@ref
Breeze.CompressibleEquations.ThermalDivergenceDamping). The default leaves this explicit
vertical divergence-damping term off; vertical acoustic damping comes from the off-centered
implicit solve. This distinction matters when comparing the equations below to the code:
there is no default post-substep ``(ρw)'`` correction kernel.

### Horizontal scaling

The implemented default uses local per-direction horizontal diffusivities

```math
γ_x = α \, \frac{Δx^2}{Δτ}, \qquad γ_y = α \, \frac{Δy^2}{Δτ},
```

where ``α`` is the dimensionless Klemp/MPAS divergence-damping coefficient. On a uniform
square grid this is the finite-difference analogue of the MPAS small-step coefficient
`coef_divdamp = 2 * smdiv * config_len_disp / dts`. On anisotropic or latitude-longitude
grids the local spacings keep the nondimensional explicit damping strength approximately
uniform across the mesh. Passing a `length_scale = ℓ` keyword overrides the automatic
local scale with a fixed ``γ = α ℓ^2 / Δτ`` in both horizontal directions.
The combined 2-D explicit-time stability bound for the horizontal correction is

```math
8α ≤ 2 \;⟹\; α ≤ 0.25 ,
```

so the empirical safe range is ``α ∈ [0.05, 0.20]``. The default ``α = 0.1`` sits well below
the bound and is the verified pairing for the default ``ω = 0.65``.

If `damp_vertical = true`, the vertical part is represented implicitly as a Laplacian on
``(ρw)'`` inside the tridiagonal solve, with CN-split factors proportional to
``ω α Δz_{\min}^2`` and ``(1-ω) α Δz_{\min}^2`` on the implicit and explicit sides.

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
``\mathcal{U}\mathcal{U}^* ≠ \mathcal{U}^*\mathcal{U}``. The norm gap means distributed
floating-point noise can excite transient-amplification subspaces even when every
individual eigenmode is neutrally stable. Off-centering and divergence damping reduce this
amplification.

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

The Klemp damping acts as a wavenumber-controlled filter that targets the divergent
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

   The split-explicit treatment decouples the fastest acoustic propagation from the outer
   integrator, so the advective CFL is the first outer-step constraint to check. It is not
   the only practical constraint, however: moist cases with strong initial acoustic
   adjustment can still require a smaller outer-step cap than the advective CFL alone
   would choose. In reduced RICO validation with one-moment microphysics, uncapped adaptive
   stepping became unstable when the outer step grew to roughly ``30\,``s, while
   ``max_Δt = 20\,``s completed a 6-hour compact run.

For LES cases translated from anelastic dynamics, the advective CFL is therefore a
performance knob rather than a complete stability criterion. Same-resolution BOMEX and
RICO pilot runs with compressible substepping completed 6 simulated hours at
``\mathrm{CFL} = 1.4``, but this did not guarantee a faster end-to-end run: the
split-explicit acoustic loop can dominate the cost. In the same RICO harness the anelastic
run failed at ``\mathrm{CFL} = 1.4`` and completed at ``\mathrm{CFL} = 0.7``. Benchmark
both the accepted ``Δt`` and the acoustic substep count ``N`` when comparing formulations.

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

The pairing ``ω = 0.65, α = 0.1`` is verified by:

| Test                                     | Result                                          |
|------------------------------------------|-------------------------------------------------|
| `test/substepper_rest_state.jl`          | Rest atmosphere at machine ``ε`` over 200 outer steps × ``Δt = 20\,``s |
| DCMIP-2016 dry baroclinic wave           | Stable for 12 simulated h × ``Δt = 225\,``s on ``360 × 160 × 64`` lat-lon grid |
| DCMIP-2016 moist baroclinic wave         | Stable for 1 simulated h × ``Δt = 20\,``s on ``360 × 160 × 64`` lat-lon grid with one-moment microphysics |

The exact discrete rest atmosphere is also covered with
`NoDivergenceDamping()` and ``ω = 0.55`` in `test/substepper_rest_state.jl`; the
stage-rewind formulation keeps that state bounded at ``Δt = 20\,``s. This should not be
interpreted as a recommendation to remove damping in production: noisy baroclinic-wave and
LES cases still use the default horizontal Klemp damping to control grid-scale divergent
acoustic modes.

## Comparison with anelastic dynamics

| Property             | [`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) | [`CompressibleDynamics`](@ref) |
|----------------------|-------------------|----------------------|
| Acoustic waves       | Filtered          | Resolved             |
| Density              | Reference ``ρ_r(z)`` only | Prognostic ``ρ(x,y,z,t)`` |
| Pressure             | Solved from Poisson equation | Computed from equation of state |
| Time step            | Limited by advective CFL | Advective CFL (split-explicit) or full acoustic CFL (explicit) |
| Typical applications | LES, mesoscale    | Global flows, baroclinic waves, acoustic studies, validation |
