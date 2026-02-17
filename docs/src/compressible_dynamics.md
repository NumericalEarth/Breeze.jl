# [Compressible dynamics](@id Compressible-section)

[`CompressibleDynamics`](@ref) solves the fully compressible Euler equations with prognostic density ``ρ``.
This formulation retains acoustic waves and is suitable for problems where full compressibility is important.

## Prognostic equations

The compressible formulation advances density ``ρ``, momentum ``ρ \boldsymbol{u}``, a thermodynamic variable ``χ`` (see [Governing equations](@ref Dycore-section)), total moisture ``ρ q^t``, and tracers:

```math
\begin{aligned}
&\text{Mass:} && \partial_t ρ + \boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u}) = 0 ,\\
&\text{Momentum:} && \partial_t(ρ \boldsymbol{u}) + \boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u} \boldsymbol{u}) + \boldsymbol{\nabla} p = - ρ g \hat{\boldsymbol{z}} + ρ \boldsymbol{f} + \boldsymbol{\nabla \cdot}\, \boldsymbol{\mathcal{T}} ,\\
&\text{Thermodynamic:} && \partial_t χ + \boldsymbol{\nabla \cdot}\, (χ \boldsymbol{u}) = Π \, \boldsymbol{\nabla \cdot \, u} + S_χ ,\\
&\text{Moisture:} && \partial_t(ρ q^t) + \boldsymbol{\nabla \cdot}\, (ρ q^t \boldsymbol{u}) = S_q .
\end{aligned}
```

Pressure is computed from the ideal gas law:

```math
p = ρ R^m T .
```

## Time integration options

`CompressibleDynamics` supports two time discretization strategies controlled by the `time_discretization` keyword:

- [`SplitExplicitTimeDiscretization`](@ref Breeze.CompressibleEquations.SplitExplicitTimeDiscretization): Acoustic substepping with separate slow/fast tendency splitting. This allows advective CFL time steps (~10-20 m/s) instead of acoustic CFL time steps (~340 m/s).

- [`ExplicitTimeStepping`](@ref Breeze.CompressibleEquations.ExplicitTimeStepping): All tendencies computed together. The time step is limited by the acoustic CFL condition: ``Δt < Δx / c_s``.

## Split-explicit time integration

The split-explicit scheme separates acoustic wave dynamics from slower dynamical processes,
allowing the outer time step to be set by the advective CFL condition
(``Δt \sim Δx / U``, where ``U \sim 10\text{--}20\,``m/s) rather than the acoustic CFL condition
(``Δτ \sim Δx / c_s``, where ``c_s \sim 340\,``m/s).
The basic strategy---subcycling fast pressure and gravity-wave dynamics within each stage of an
outer Runge-Kutta integration---was introduced by [Klemp and Wilhelmson (1978)](@cite Klemp1978)
and has been widely adopted in production models including
WRF ([Skamarock and Klemp 1994](@cite SkamarockKlemp1994);
[Wicker and Skamarock 2002](@cite WickerSkamarock2002);
[Klemp, Skamarock, and Dudhia 2007](@cite KlempSkamarockDudhia2007)),
MPAS-Atmosphere ([Skamarock et al. 2012](@cite SkamarockEtAl2012)),
COSMO ([Baldauf et al. 2011](@cite BaldaufEtAl2011)),
and CM1 ([Bryan and Fritsch 2002](@cite BryanFritsch2002)).

### Slow-fast decomposition

Starting from the compressible equations for density ``ρ``, momentum
``\boldsymbol{m} = ρ \boldsymbol{u}``, and a conservative thermodynamic variable ``χ``
(e.g., ``χ = ρθ`` for the potential temperature formulation), we decompose the right-hand side
into slow and fast components:

```math
\partial_t U = \underbrace{G^{\text{slow}}(U)}_{\text{evaluated once per RK stage}}
             + \underbrace{G^{\text{fast}}(U; \bar{U})}_{\text{subcycled acoustically}}
```

The **slow operator** ``G^{\text{slow}}`` is evaluated once per outer Runge-Kutta stage from
the current state and held fixed during the acoustic substeps. It includes:

- Advective flux divergences in momentum, density, and thermodynamic variable
- Coriolis and other body forces
- Subgrid stresses and turbulent diffusion
- Microphysics sources and external forcing
- The full pressure gradient and buoyancy evaluated at the stage state

The **fast operator** ``G^{\text{fast}}`` resolves acoustic wave propagation within each stage
via a forward-backward substep loop. It contains:

- A *linearized* pressure gradient that couples perturbation momentum to the perturbation
  thermodynamic variable (see [Exner function linearization](@ref exner-linearization) below)
- Mass flux divergence of the perturbation momentum in the continuity and thermodynamic equations

The linearization around a stage-frozen reference state ``\bar{U}`` makes the fast dynamics
*linear* in the perturbation variables, allowing stable integration via a forward-backward
scheme regardless of the outer time step size.

### Outer Runge-Kutta integration

Two outer Runge-Kutta loop variants are available, both three-stage.

#### Wicker-Skamarock RK3

The default [`AcousticRungeKutta3`](@ref) time stepper
([Wicker and Skamarock 2002](@cite WickerSkamarock2002)) uses stage fractions
``β = 1/3, 1/2, 1``:

```math
\begin{aligned}
U^{(1)} &= U^n + \tfrac{Δt}{3} \, R(U^n) \\
U^{(2)} &= U^n + \tfrac{Δt}{2} \, R(U^{(1)}) \\
U^{n+1} &= U^n + Δt \, R(U^{(2)})
\end{aligned}
```

Each stage resets to the initial state ``U^n`` and advances by ``β \, Δt``. The acoustic
substep size is **constant** across all stages: ``Δτ = Δt / N_s``, while the substep count
varies: ``N_τ = \max(\mathrm{round}(β N_s), 1)``. This ensures the acoustic CFL number is
the same regardless of stage fraction.

#### SSP RK3 (alternative)

The [`AcousticSSPRungeKutta3`](@ref) time stepper uses the strong-stability-preserving (SSP) RK3
scheme in Shu-Osher form ([Shu and Osher 1988](@cite Shu1988Efficient)):

```math
\begin{aligned}
U^{(1)} &= \Phi(U^n; \, Δt) \\
U^{(2)} &= \tfrac{3}{4} U^n + \tfrac{1}{4} \Phi(U^{(1)}; \, Δt) \\
U^{n+1} &= \tfrac{1}{3} U^n + \tfrac{2}{3} \Phi(U^{(2)}; \, Δt)
\end{aligned}
```

where ``\Phi`` denotes the forward Euler + acoustic subcycling stage operator.
The convex combination mixes fields from different acoustic states.
The SSP property guarantees monotonicity preservation for advected scalars.

### Acoustic variables: velocity and Exner pressure

The acoustic substep loop advances **velocity** ``(u, v, w)`` and the **Exner pressure
perturbation** ``\pi' = \pi - \pi_0`` as prognostic variables, following
CM1 ([Bryan and Fritsch 2002](@cite BryanFritsch2002)).
This is a velocity-pressure formulation, distinct from the momentum-perturbation approach
used by MPAS and earlier versions of this code.

The Exner pressure is defined as

```math
\pi = \left( \frac{p}{p^{st}} \right)^{\!\kappa}, \qquad \kappa = R^d / c_p^d ,
```

and is related to temperature by ``T = \theta_v \, \pi``, where ``\theta_v`` is the
virtual potential temperature. The reference Exner profile ``\pi_0(z)`` satisfies
discrete hydrostatic balance to machine precision, ensuring no spurious vertical
pressure gradient from the reference state.

At the start of each Runge-Kutta stage, the following **stage-frozen** quantities are
computed from the current (evaluation) state and held fixed during the acoustic substeps:

- ``\theta_v``: virtual potential temperature
- ``\pi_0``: reference Exner pressure (from the ExnerReferenceState)
- ``S = (\gamma - 1) \pi``: Exner pressure tendency coefficient, where
  ``\gamma = c_p^m / c_v^m`` is the mixture heat capacity ratio

The acoustic loop also requires **slow tendencies** converted to velocity and pressure form
(see [Slow tendency conversion](@ref slow-tendency-conversion) below).

### Forward-backward acoustic substep loop

Within each RK stage, the acoustic substep loop iterates ``N_\tau`` times with a
**constant substep size** ``\Delta\tau = \Delta t / N_s`` for both time steppers.
For Wicker-Skamarock RK3, the substep count varies per stage:
``N_\tau = \max(\mathrm{round}(\beta N_s), 1)``, keeping ``\Delta\tau`` constant.
For SSP RK3, ``N_\tau = N_s`` at every stage.

Each substep consists of three phases:

**(A) Forward step --- horizontal velocity update:**

```math
\begin{aligned}
u^{\tau + \Delta\tau} &= u^\tau + \Delta\tau \left( \dot{u}^s - c_p^d \, \bar{\theta}_v \, \frac{\partial \tilde{\pi}'}{\partial x} \right) \\
v^{\tau + \Delta\tau} &= v^\tau + \Delta\tau \left( \dot{v}^s - c_p^d \, \bar{\theta}_v \, \frac{\partial \tilde{\pi}'}{\partial y} \right)
\end{aligned}
```

where ``\dot{u}^s`` and ``\dot{v}^s`` are the slow velocity tendencies (advection, Coriolis,
diffusion, and the full pressure gradient from the stage state, divided by density), and
``\tilde{\pi}'`` is the forward-extrapolation-filtered Exner perturbation
(see [Forward-extrapolation filter](@ref forward-extrapolation-filter) below).

**(B) Vertically implicit ``w``-``\pi'`` solve:**

The vertical velocity ``w`` and Exner pressure perturbation ``\pi'`` are coupled through the
vertical pressure gradient and vertical divergence. To avoid the severe ``\Delta\tau < \Delta z / c_s``
constraint on vertically refined grids, this coupling is treated implicitly via a tridiagonal
system each substep.

Using off-centering parameter ``\alpha`` (default 0.6), the update is split into explicit
(weight ``\beta = 1 - \alpha``) and implicit (weight ``\alpha``) parts:

```math
\begin{aligned}
w^{\tau + \Delta\tau} &= w^\tau + \Delta\tau \, \dot{w}^s
    - \Delta\tau \, c_p^d \bar{\theta}_v \left[ \beta \frac{\partial \pi'^{\,\tau}}{\partial z} + \alpha \frac{\partial \pi'^{\,\tau+\Delta\tau}}{\partial z} \right] \\
\pi'^{\,\tau+\Delta\tau} &= \pi'^{\,\tau} + \Delta\tau \, \dot{\pi}^s
    + \Delta\tau \, S \left[ \nabla_h \cdot \boldsymbol{u}^{\tau+\Delta\tau}
    + \beta \frac{\partial w^\tau}{\partial z} + \alpha \frac{\partial w^{\tau+\Delta\tau}}{\partial z} \right]
\end{aligned}
```

Substituting the ``w`` equation into the ``\pi'`` equation and rearranging yields a
tridiagonal system for ``\pi'^{\,\tau+\Delta\tau}`` with coupling coefficient
``\alpha^2 \Delta\tau^2 \, c_p^d \, \bar{\theta}_v \, S / \Delta z^2 = \alpha^2 \Delta\tau^2 c_s^2 / \Delta z^2``.
After solving for ``\pi'``, the vertical velocity is recovered via the ``w`` equation.

**(C) Filter and accumulate:**

After each substep, apply the forward-extrapolation filter to ``\pi'`` (see below) and
accumulate time-averaged velocities for scalar transport:

```math
\bar{\boldsymbol{u}} = \frac{1}{N_\tau} \sum_{n=1}^{N_\tau} \boldsymbol{u}^{(n)}
```

### Recovery: converting acoustic variables back to prognostic fields

After the acoustic substep loop, the velocity fields ``(u, v, w)`` and Exner perturbation
``\pi'`` must be converted back to Breeze's prognostic variables ``(\rho, \rho\boldsymbol{u},
\rho\theta)``.

The recovery differs between the two time steppers:

#### SSP RK3 recovery

Each SSP stage computes a forward Euler step from the evaluation state ``U^*`` over a full
``\Delta t``, then blends with ``U^0`` via a convex combination. The recovery is:

```math
\begin{aligned}
\rho\theta_{\text{new}} &= \frac{p^{st}}{R^d} \, \pi_{\text{new}}^{\,c_v / R} \\
\theta_{\text{new}} &= \bar{\theta}_v + \Delta t \, \dot{\theta}^s \\
\rho_{\text{new}} &= \rho\theta_{\text{new}} \, / \, \theta_{\text{new}}
\end{aligned}
```

where ``\pi_{\text{new}} = \pi_0 + \pi'`` is the total Exner pressure after the acoustic loop,
``\bar{\theta}_v`` is the evaluation state's virtual potential temperature, and
``\dot{\theta}^s = (G^s_{\rho\theta} - \bar{\theta}_v G^s_\rho) / \bar{\rho}`` is the slow
potential temperature tendency. The first line converts Exner pressure to ``\rho\theta`` via
the equation of state; the remaining lines diagnose density from the slowly-evolved ``\theta``.

#### Wicker-Skamarock RK3 recovery

WS-RK3 computes each stage as ``U_{\text{new}} = U^n + \beta \Delta t \, R(U^*)``, where
``U^n`` is the initial state and ``R(U^*)`` is evaluated at the current stage state. The
recovery applies the acoustic perturbation to the **initial** state:

```math
\begin{aligned}
\pi^n &= \left( \frac{R^d \, (\rho\theta)^n}{p^{st}} \right)^{R/c_v} \\
\pi_{\text{new}} &= \pi^n + \Delta\pi', \qquad \Delta\pi' = \pi'_{\text{final}} - \pi'_{\text{initial}} \\
\rho\theta_{\text{new}} &= \frac{p^{st}}{R^d} \, \pi_{\text{new}}^{\,c_v / R} \\
\theta_{\text{new}} &= \theta^n + \beta \, \Delta t \, \dot{\theta}^s \\
\rho_{\text{new}} &= \rho\theta_{\text{new}} \, / \, \theta_{\text{new}}
\end{aligned}
```

where ``\pi'_{\text{initial}}`` and ``\pi'_{\text{final}}`` are the Exner perturbation at the
start and end of the acoustic loop, and ``\theta^n = (\rho\theta)^n / \rho^n`` comes from the
stored initial state ``U^0``.

!!! warning "WS-RK3: consistent initial state in the acoustic loop"
    The acoustic loop must start from a **consistent** ``U^n`` state: both velocity and ``\pi'``
    must come from the initial state. If ``\pi'`` is initialized from the evaluation state
    ``U^*`` while velocities are reset to ``U^n``, the resulting velocity-pressure mismatch
    destabilizes the acoustic loop at advective time steps. This is because the horizontal
    pressure gradient seen by the initial velocities is inconsistent with the pressure field,
    generating spurious acoustic oscillations that grow over multiple time steps.

!!! warning "WS-RK3: use ``\\theta^n`` as the base for ``\\theta`` evolution"
    The slow ``\theta`` tendency must be applied to ``\theta^n`` from ``U^0``, **not** to
    ``\bar{\theta}_v`` from the evaluation state. Using the evaluation state's ``\theta``
    double-counts the ``\theta`` change from earlier stages (since ``\theta(U^*) = \theta^n +
    \beta_1 \Delta t \, \dot{\theta}^s`` already includes the stage 1 contribution), producing
    an ``O(\Delta t)`` error per time step that causes instability at larger ``\Delta t``.

### [Slow tendency conversion](@id slow-tendency-conversion)

The outer Runge-Kutta loop computes slow tendencies in **conservative** (momentum/density)
form: ``G^s_{\rho u}``, ``G^s_{\rho w}``, ``G^s_\rho``, ``G^s_{\rho\theta}``. Before entering
the acoustic substep loop, these are converted to the velocity and pressure form used by the
acoustic variables.

#### Velocity tendencies

The slow momentum tendencies are converted to velocity tendencies by dividing by the
stage-frozen density:

```math
\dot{u}^s = G^s_{\rho u} / \bar{\rho}, \qquad \dot{w}^s = G^s_{\rho w} / \bar{\rho} + B
```

where ``B = -c_p^d \bar{\theta}_v \, \partial\pi_0 / \partial z - g`` is the buoyancy term
arising from the mismatch between the reference state and the actual hydrostatic balance.

!!! note "Density correction in velocity tendencies"
    The exact conversion from momentum to velocity tendency is
    ``\dot{u} = (G^s_{\rho u} - u \, G^s_\rho) / \bar{\rho}``. However, the density correction
    term ``-u \, G^s_\rho / \bar{\rho}`` can cause slow secular growth at long integration
    times. The simpler form ``\dot{u} = G^s_{\rho u} / \bar{\rho}`` avoids this issue and
    produces accurate results for the SK94 benchmark.

#### Exner pressure tendency

The slow Exner pressure tendency represents the advective transport of ``\pi``:

```math
\dot{\pi}^s = -\boldsymbol{u} \cdot \nabla \pi
```

This is computed using centered differences (not WENO) to maintain consistency with the
centered-difference divergence operator in the acoustic loop.

!!! warning "No ``R/c_v`` factor in ``\\dot{\\pi}^s``"
    The chain rule for ``\pi = f(\rho\theta)`` gives
    ``\boldsymbol{u} \cdot \nabla\pi = (R/c_v)(\pi / \rho\theta) \, \boldsymbol{u} \cdot \nabla(\rho\theta)``,
    so the ``R/c_v`` factor is already embedded in the advection of ``\pi``.
    Writing ``\dot{\pi}^s = -(R/c_v) \, \boldsymbol{u} \cdot \nabla\pi`` would double-count
    this factor and reduce the perturbation amplitude by a factor of ``\sim 2.5``.

### [Forward-extrapolation filter](@id forward-extrapolation-filter)

A forward-extrapolation filter suppresses spurious computational-mode oscillations in the
forward-backward substep scheme. After each substep's implicit solve, the filtered Exner
perturbation used in the *next* substep's pressure gradient is

```math
\tilde{\pi}'^{\,\tau+\Delta\tau} = \pi'^{\,\tau+\Delta\tau} + \kappa^d \left( \pi'^{\,\tau+\Delta\tau} - \pi'^{\,\tau} \right)
```

where ``\kappa^d`` is the divergence damping coefficient (typically 0.05--0.10). The
**unfiltered** ``\pi'`` is used for the actual field update and recovery; only the filtered
``\tilde{\pi}'`` enters the next forward step's pressure gradient. This acts as a forward-in-time
extrapolation that damps the ``2\Delta\tau`` computational mode inherent in the forward-backward
scheme, following [Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007).

### [Why the Exner linearization is necessary](@id exner-linearization)

In a ``ρ``-based formulation where the perturbation pressure gradient takes the form
``c_s^2 \, \boldsymbol{\nabla} ρ''``, the sound speed coefficient ``c_s^2 = γ R^d T`` must
be recomputed from the equation of state at each Runge-Kutta stage. This recomputation couples
acoustic-amplitude density perturbations ``ρ''`` back into the pressure field used by the
next stage's slow tendency evaluation, effectively imposing an **acoustic CFL constraint on
the outer time step**.

[Skamarock and Klemp (1992)](@cite SkamarockKlemp1992) analyzed the stability of time-split
methods and showed that the amplification factor of the outer Runge-Kutta integrator, when
applied to the acoustic modes, limits the outer Courant number. For the Wicker-Skamarock RK3
this bound is approximately

```math
\frac{c_s \, Δt}{Δz} \lesssim \frac{\sqrt{3}}{2} \approx 0.87
```

With ``c_s \approx 347\,``m/s and ``Δz = 1000\,``m, this restricts
``Δt \lesssim 2.5\,``s---only marginally better than the unsplit acoustic CFL and far
short of what the advective CFL would allow.

The Exner pressure formulation resolves this by evolving ``\pi'`` as the acoustic prognostic
variable. Since the acoustic pressure gradient ``c_p^d \bar{\theta}_v \nabla\pi'`` uses
stage-frozen ``\bar{\theta}_v``, the acoustic dynamics are decoupled from the outer
Runge-Kutta integration. The equation of state is still re-evaluated between stages (providing
an accurate state for the next stage's slow tendencies), but this re-evaluation does not feed
back into the acoustic substep dynamics.

### Stability constraints and practical guidance

The split-explicit scheme involves two CFL-like constraints:

1. **Acoustic substep CFL**: Each substep must resolve horizontal acoustic wave propagation:

   ```math
   Δτ < \frac{Δx}{c_s + |\boldsymbol{u}|}
   ```

   The vertically implicit solver removes the vertical acoustic CFL constraint entirely.

2. **Advective CFL for the outer step**: The outer time step ``Δt`` must remain stable for
   the advective dynamics resolved by the outer Runge-Kutta integration. The Exner pressure
   formulation decouples the acoustic modes from the outer integrator, so the outer time step
   is limited only by the advective CFL.

These constraints determine the required number of substeps ``N_s``. For the default
Wicker-Skamarock RK3, the acoustic substep size is constant at ``Δτ = Δt / N_s``
while the substep count varies per stage (``N_\tau = \mathrm{round}(\beta N_s)``).

For the SK94 inertia-gravity wave benchmark (``\Delta x = \Delta z = 1\,``km, ``U = 20\,``m/s),
``N_s = 8`` with ``\Delta t = 12\,``s gives ``\Delta\tau = 1.5\,``s and an acoustic CFL of
``\approx 0.52``.

### Summary of the algorithm

The complete algorithm for one Wicker-Skamarock RK3 time step is:

1. **Store initial state**: ``U^0 = U^n``
2. **For each RK stage** ``k = 1, 2, 3`` with fractions ``β_k = 1/3, 1/2, 1``:
   1. Compute slow tendencies ``G^s`` from the current evaluation state
   2. Convert slow tendencies to velocity/pressure form: ``\dot{u}^s``, ``\dot{w}^s``, ``\dot{\pi}^s``
   3. Freeze stage quantities: ``\bar{\theta}_v``, ``S``, ``\pi_0``
   4. Initialize ``\pi' = \pi(U^n) - \pi_0`` (consistent with velocity reset to ``U^n``)
   5. Reset velocities ``(u, v, w)`` to ``U^n``
   6. **Acoustic substep loop** (``N_\tau`` iterations with ``\Delta\tau = \Delta t / N_s``):
      - Forward: update ``u, v`` from ``\dot{u}^s`` and ``c_p^d \bar{\theta}_v \nabla\tilde{\pi}'``
      - Implicit: solve tridiagonal for ``\pi'`` and recover ``w``
      - Filter: apply forward-extrapolation to ``\pi'``
      - Accumulate time-averaged velocities
   7. **Recover** ``(\rho\theta, \rho)`` from ``\pi'`` via the equation of state
   8. Reconstruct momentum: ``\rho\boldsymbol{u} = \rho \, \boldsymbol{u}``
   9. Update state: compute ``p``, ``T``, ``\theta_v`` from the equation of state

## Comparison with anelastic dynamics

| Property | [`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) | [`CompressibleDynamics`](@ref) |
|----------|-------------------|----------------------|
| Acoustic waves | Filtered | Resolved |
| Density | Reference ``ρᵣ(z)`` only | Prognostic ``ρ(x,y,z,t)`` |
| Pressure | Solved from Poisson equation | Computed from equation of state |
| Time step | Limited by advective CFL | Advective CFL (split-explicit) or acoustic CFL (explicit) |
| Typical applications | LES, mesoscale | Acoustic studies, validation |
