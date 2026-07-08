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

### Terrain-following fast acoustic system

When [`CompressibleDynamics`](@ref) is given terrain metrics, the acoustic substep loop
uses the contravariant vertical momentum ``ρ\tilde{w}`` for transport through
``\zeta``-surfaces. The perturbation stored in the substepper's vertical momentum slot is

```math
(ρ\tilde{w})' =
\left[ρw - \left(\frac{\partial z}{\partial x}\right)_\zeta ρu
          - \left(\frac{\partial z}{\partial y}\right)_\zeta ρv \right]
-
\left[ρw - \left(\frac{\partial z}{\partial x}\right)_\zeta ρu
          - \left(\frac{\partial z}{\partial y}\right)_\zeta ρv \right]^L .
```

The horizontally explicit momentum update still advances Cartesian momenta ``(ρu)'`` and
``(ρv)'``, but the pressure-gradient operator is the terrain-aware physical gradient
``(\partial p / \partial x)_z`` and ``(\partial p / \partial y)_z`` described in
[Terrain-following coordinates](@ref Terrain-following-section). The mass and thermodynamic
fast equations use the contravariant vertical perturbation flux:

```math
\begin{aligned}
∂_τ ρ'    &+ ∂_x(ρu)' + ∂_y(ρv)' + ∂_\zeta (ρ\tilde{w})' = G^s_ρ , \\
∂_τ (ρθ)' &+ ∂_x\!\left(θ^L (ρu)'\right)
            + ∂_y\!\left(θ^L (ρv)'\right)
            + ∂_\zeta\!\left(θ^L (ρ\tilde{w})'\right) = G^s_{ρθ}.
\end{aligned}
```

The vertical acoustic equation is projected onto the same contravariant
momentum. With static terrain slopes, the pressure part of the fast force is

```math
\partial_\zeta p'
- \left(\frac{\partial z}{\partial x}\right)_\zeta \partial_x p'
- \left(\frac{\partial z}{\partial y}\right)_\zeta \partial_y p' .
```

The slow vertical tendency is projected consistently as the Cartesian
vertical-momentum slow tendency minus the slope-weighted horizontal slow
tendencies, plus the slope-weighted frozen horizontal pressure gradients.
This prevents the recovery step below from adding a spurious slope times
horizontal momentum update to the Cartesian ``ρw`` tendency.

The vertically implicit column solve remains a tridiagonal solve in the contravariant
vertical-momentum perturbation. After the solve, Breeze recovers the Cartesian vertical
momentum needed by the rest of the model through

```math
(ρw)^{τ+Δτ} =
(ρ\tilde{w})^{τ+Δτ}
+ \left(\frac{\partial z}{\partial x}\right)_\zeta (ρu)^{τ+Δτ}
+ \left(\frac{\partial z}{\partial y}\right)_\zeta (ρv)^{τ+Δτ}.
```

At the lower boundary, the no-normal-flow condition is imposed as
``\tilde{w} = 0`` and ``ρ\tilde{w} = 0`` on the bottom face. This is the terrain-surface
impenetrability condition; enforcing only Cartesian ``w = 0`` would allow a nonzero normal
flux when the lower coordinate surface is sloped.

For zero terrain, ``(\partial z / \partial x)_\zeta =
(\partial z / \partial y)_\zeta = 0``. Therefore ``\tilde{w} = w``,
``ρ\tilde{w} = ρw``, the terrain pressure-gradient operator reduces to the Cartesian
operator, and the terrain acoustic substep equations reduce exactly to the height-coordinate
system above.

### [Reference state and discrete hydrostatic balance](@id reference-state)

The slow vertical PGF ``-∂_z p^L - ρ^L g`` is the difference between two large numbers,
each ``\mathcal{O}(10^4)`` in SI units, whose true value is small everywhere and exactly zero
in a rest atmosphere. To preserve this cancellation at the discrete level,
`CompressibleDynamics` accepts a `reference_state` keyword that builds a
[`ExnerReferenceState`](@ref) ``(ρ_r, p_r, Π_r)`` satisfying

```math
\frac{p_{r,k} - p_{r,k-1}}{Δz_k^f} + g \, \frac{ρ_{r,k} + ρ_{r,k-1}}{2} = 0
```

at every interior z-face ``k`` — the *discrete* hydrostatic balance with the same
two-point face derivative and arithmetic face-average that the substepper itself uses.
The slow vertical momentum tendency uses the *imbalance*
``-∂_z(p^L - p_r) - (ρ^L - ρ_r) g`` so that a column in exact discrete balance contributes
zero buoyancy forcing, no matter how steeply ``ρ_r(z)`` and ``p_r(z)`` vary.

#### Per-column Newton integration

For a prescribed background ``\bar{θ}(z)`` and optional ``\bar{q}^v(z)``, the level-local
moist EOS

```math
ρ_{r,k} = \frac{p_{r,k}}{R^m_k \, T_{r,k}}, \quad
T_{r,k} = \bar{θ}_k \, Π_{r,k}, \quad
Π_{r,k} = \!\left(\frac{p_{r,k}}{p^{st}}\right)^{κ^m_k}, \quad
κ^m_k = \frac{R^m_k}{c^{pm}_k}
```

is substituted into the discrete-balance constraint. The result is a scalar residual

```math
F_k(p) \;=\; \frac{p - p_{r,k-1}}{Δz_k^f}
      + g \, \frac{ρ(p) + ρ_{r,k-1}}{2},
\qquad
ρ(p) = \frac{p^{1-κ^m_k} \, (p^{st})^{κ^m_k}}{R^m_k \, \bar{θ}_k}
```

that is monotone increasing in ``p``; Newton iteration from a continuous-``Π`` guess
converges in O(few) iterations and runs inside the per-column loop of a single GPU
kernel. The first cell center is anchored by the continuous ``Π`` recurrence over the
half-step below it (face ``k=1`` is the impenetrability boundary — no discrete-balance
constraint applies there, so the anchor is free).

Why the discrete (rather than continuous) balance matters: an MPAS-style up-then-down
``Π`` integration satisfies the *continuous* hydrostatic equation but leaves an
``\mathcal{O}(Δz^2)`` truncation residual of order ``10^{-3}\,``N/m³ in the substepper's
discrete face operator. At production ``Δt`` that residual seeds an acoustic instability;
discrete balance brings it to machine precision and the rest-atmosphere test in
`test/substepper_rest_state.jl` holds at ulp.

#### Moist reference states

Passing `vapor_mass_fraction` to `ExnerReferenceState` replaces the dry constants with the
level-local moist mixture

```math
R^m_k = (1 - q^v_k)\, R^d + q^v_k\, R^v, \qquad
c^{pm}_k = (1 - q^v_k)\, c^{pd} + q^v_k\, c^{pv} .
```

The dry path (`vapor_mass_fraction === nothing`) uses a `ZeroField` for ``q^v`` and is
recovered *exactly* — same residual, same Newton trajectory, no bit-level drift.

`CompressibleDynamics` exposes this through the `reference_vapor_mass_fraction` keyword:
moist convection cases that target a state with ``\bar{q}^v(z) > 0`` must pass both
`reference_potential_temperature` and `reference_vapor_mass_fraction`, otherwise the
reference state is dry and the moist resting atmosphere is not in discrete balance —
which radiates spurious acoustic / gravity waves on startup.

#### 1D and 3D reference backgrounds

A constant or 1-argument ``\bar{θ}(z)`` builds a single column broadcast to all ``(i,j)``;
a multi-argument ``\bar{θ}(x, y, z)`` (e.g. the latitude-dependent profile of the
DCMIP-2016 baroclinic wave) triggers a per-column integration via the same kernel,
indexed over ``(i, j)``. Both paths support `vapor_mass_fraction`; the 3D path accepts
``\bar{q}^v(x, y, z)`` and allocates a `CenterField`, while the 1D path uses the column
field. Either way, every column individually satisfies the discrete balance above to
machine precision.

#### Pressure-balanced density for initial conditions

A reference state in discrete balance is necessary but not sufficient — an initial
``θ`` perturbation that leaves ``ρ = ρ_r`` unchanged shifts ``ρθ`` and therefore the
diagnosed initial pressure, putting the perturbed state *out* of balance even though
the background is balanced. The
[`pressure_balanced_density`](@ref Breeze.Thermodynamics.pressure_balanced_density)
helper preserves ``ρθ`` under a ``θ`` perturbation:

```math
ρ(x,y,z) \;=\; ρ_r(z)\, \frac{\bar{θ}(z)}{θ(x,y,z)} .
```

Initializing ``ρ`` from this helper, instead of ``ρ = ρ_r`` directly, keeps the resting
discrete balance under perturbations and suppresses the acoustic / gravity-wave noise
that an unbalanced startup would otherwise radiate.

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

**Vertical implicit solve — column tridiag in the acoustic vertical momentum.** The
vertical-momentum, density, and ``ρθ`` perturbations are coupled through the vertical pressure gradient, the vertical
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

For height-coordinate dynamics the tridiagonal unknown is ``(ρw)'``. For
terrain-following dynamics the same column solve advances the contravariant
vertical momentum perturbation ``(ρ\tilde{w})'`` and recovers Cartesian ``ρw``
after the acoustic update.

The horizontal divergence in the mass and ``ρθ`` equations is taken from the just-updated
horizontal momenta ``(ρu)'_{τ+Δτ}, (ρv)'_{τ+Δτ}`` (forward–backward coupling). Substituting
the discrete updates of ``ρ'`` and ``(ρθ)'`` into the vertical-momentum equation yields a
tridiagonal Schur system at z-faces, with diagonals proportional to
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

### Specified-zone boundary drive (`BoundaryTendencyMarch`)

The substep loop advances the fast system many times between the slow
(per-RK-stage) boundary updates, so a normal-flow lateral boundary needs
boundary information *within* the acoustic substeps.
[`BoundaryTendencyMarch`](@ref) supplies it with the specified-zone approach of
[MPAS](@cite SkamarockEtAl2012) (and WRF's specified boundary): when the
momentum `NormalFlowBoundaryCondition`s on a side carry the scheme, the
outermost interior cells on that side become a *specified zone* in which

1. the acoustic perturbation pressure gradient is **gated** on every face whose
   stencil reads a specified cell,
2. specified cells are **excluded** from the coupled acoustic update (mass and
   ``ρθ`` predictors and the implicit vertical solve),
3. the specified column's ``(ρw)'`` is closed by a per-substep **zero-gradient**
   copy from the nearest interior column (WRF `zero_grad_bdy` analog), and
4. the specified zone's momentum and scalar perturbations are **marched** by
   their boundary time-tendencies every acoustic substep:

```math
(ρu)' ← (ρu)' + Δτ \, ∂_t(ρu)_\mathrm{boundary}
```

the analog of MPAS's `ru_p += dts·lbc_tend_ru`. Because each RK stage
initializes its perturbations with the rewind ``(ρu)' = U⁰ − U^L_\mathrm{stage}``,
the per-substep *increment* composes to ``U⁰ + β\,Δt\,∂_t`` at each stage end —
the boundary state at its stage time. (An overwrite ``τ\,∂_t`` would instead
compound across stages into a secular ``(β₁+β₂+β₃) = 11/6`` over-advance per
outer step, invisible to steady-state tests.)

On marched sides the per-substep ``α`` relaxation of ``ρᵈ', (ρθ)'`` is
superseded and skipped: the march holds the same cells to the time-accurate
boundary state directly. Normal-flow boundaries without the scheme retain the
relaxation unchanged.

Tendency sources are callables of ``(x, y, z, t)`` evaluated over the specified
zone once per outer time step; alternatively, the underlying tendency fields are
exposed through [`boundary_tendency_fields`](@ref) for drivers whose boundary
data cannot be evaluated on the device (e.g. interpolated forcing files).

Two current restrictions: the scheme errors on `TerrainCompressibleDynamics`
(the terrain horizontal pressure-gradient stencils are not column-local, so
marched specified-cell scalars would leak into interior columns), and optional
post-loop stage physics (vertically implicit closures, microphysics updates)
is not excluded from the specified zone — with such physics enabled the zone
deviates from the driving data by one step of ungated tendencies.

## [Klemp divergence damping](@id klemp-damping)

[Klemp, Skamarock, and Ha (2018)](@cite KlempSkamarockHa2018) prescribe a per-substep
divergence-damping correction that targets acoustic divergence modes while leaving
discrete rest states and balanced flow essentially unchanged, building on
[Skamarock and Klemp (1992)](@cite SkamarockKlemp1992) and the linear stability analysis of
[Baldauf (2010)](@cite Baldauf2010). Breeze applies the horizontal part by default as a
practical acoustic filter for production runs; it is not intended as an explanation for every
resolved high-wavenumber feature in baroclinic-wave diagnostics.

The discrete divergence proxy is the per-substep change in ``(ρθ)'``, normalized by the
stage-entry ``θ^L`` cache used by the acoustic transport equation:

```math
D_τ \equiv \frac{(ρθ)'_τ - (ρθ)'_{τ-Δτ}}{θ^L} \;≈\; -\, Δτ \, ∇·(ρ\boldsymbol{u})' .
```

After the implicit Schur solve, the horizontal momentum perturbation components pick up the
explicit correction

```math
\begin{aligned}
Δ(ρu)' &= - γ_x \, ∂_x D_τ , \\
Δ(ρv)' &= - γ_y \, ∂_y D_τ .
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
the same vertical-momentum perturbation, ``(ρw)'`` in height coordinates or
``(ρ\tilde{w})'`` in terrain-following coordinates, inside the tridiagonal solve,
with CN-split factors proportional to ``ω α Δz_{\min}^2`` and
``(1-ω) α Δz_{\min}^2`` on the implicit and explicit sides.

## [Stability analysis](@id stability-analysis)

The split-explicit scheme has two sources of acoustic-mode amplification that can interact.
Off-centering and divergence damping are the available controls.

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
``\mathcal{U}\mathcal{U}^* ≠ \mathcal{U}^*\mathcal{U}``. The norm gap means perturbations
can transiently project onto amplified subspaces even when every individual eigenmode is
neutrally stable. The stage-rewind formulation preserves the exact discrete rest state in
`test/substepper_rest_state.jl`; off-centering and divergence damping reduce amplification
of noisy divergent acoustic components in production runs.

### 2 — Outer/inner coupling

[Knoth and Wensch (2014)](@cite KnothWensch2014) analyze the coupled stability of an
outer Runge–Kutta scheme with an inner forward-backward substep and show that the
WS-RK3 + substepper combination is **conditionally unstable** for centered Crank–Nicolson
*regardless of how the substep operator itself is constructed*. The mechanism is that any
acoustic perturbation generated inside the substep loop is re-injected on the next outer
stage through the slow-tendency evaluation; for centered CN this re-injection has no
dissipative channel, so the coupled amplification factor exceeds unity for a non-empty
range of acoustic CFL.

This means damping is not only a patch for one implementation error; it is a standard
control for the WS-RK3 + substepper coupling. The same practical conclusion follows from
the analysis in
[Skamarock and Klemp (1992)](@cite SkamarockKlemp1992),
[Baldauf (2010)](@cite Baldauf2010), and
[Klemp, Skamarock, and Ha (2018)](@cite KlempSkamarockHa2018), which all prescribe
divergence damping as a robust acoustic filter for practical integrations.

### 3 — Damping as a filter

The Klemp damping acts as a wavenumber-controlled filter that targets the divergent
acoustic component while leaving balanced (non-divergent) modes essentially untouched. The
divergence proxy ``D_τ ≈ -Δτ ∇·(ρ\boldsymbol{u})'`` vanishes for a discrete rest state and
for purely solenoidal flow. Divergent acoustic perturbations pick up dissipation; resolved
balanced modes with nonzero divergence should be assessed with convergence and benchmark
diagnostics rather than attributed to the filter alone.

## Stability constraints and practical guidance

Two CFL-like constraints govern the choice of ``Δt`` and the substep count ``N``:

1. **Acoustic substep CFL** (horizontal):

   ```math
   Δτ ≤ \frac{\min(Δx, Δy)}{c_s + |\boldsymbol{u}|} .
   ```

   The speed of sound ``c_s = \sqrt{\gamma^d R^d T_r}``, where the reference temperature is chosen to be ``T_r = 300\,``K.

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
each step:

```math
N \approx
\left\lceil \frac{Δt \, \mathbb{C}^{ac}}{ν \, Δx_\min} \right\rceil ,
```

with ``\mathbb{C}^{ac} = \sqrt{γ^d R^d T_r}`` evaluated at a nominal reference temperature ``T_r = 300\,``K and
``ν = `` `acoustic_cfl` (default ``0.5``, the ERF/WRF target — equivalent to the
conventional safety factor of ``2``). Lower ``ν`` produces more substeps and a shorter
``Δτ``; raise it (closer to the linear stability bound of ``1``) only after verifying that
the resulting acoustic noise level remains acceptable. The setting is ignored when
`substeps` is given an explicit integer, which pins ``Δτ = Δt / N`` for reproducibility.

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
