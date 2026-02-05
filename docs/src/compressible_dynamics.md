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

The split-explicit scheme follows [Wicker and Skamarock (2002)](@cite WickerSkamarock2002)
and [Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007).
It uses a three-stage SSP Runge-Kutta (SSPRK3) outer loop for slow tendencies
with an inner forward-backward acoustic substep loop for fast tendencies.

### Slow-fast splitting

The right-hand side is split as

```math
\partial_t U = G_{\mathrm{slow}}(U) + G_{\mathrm{fast}}(U; \bar{U}),
```

where ``\bar{U}`` is the stage-frozen reference state defined at the start of each RK stage.

The **fast operator** contains:
- Pressure gradient in the momentum equation
- Buoyancy in the vertical momentum equation
- Mass flux divergence in the continuity equation
- Linearized flux divergence in the thermodynamic equation

The **slow operator** contains everything else:
- Advection (in advective form)
- Coriolis force
- Turbulent diffusion
- Microphysics and forcing

### SSPRK3 outer loop

The three-stage SSP RK3 scheme in Shu-Osher form advances the state over a full time step ``Δt``:

```math
\begin{aligned}
U^{(1)} &= \Phi(U^n; \, Δt) \\
U^{(2)} &= \tfrac{3}{4} U^n + \tfrac{1}{4} \Phi(U^{(1)}; \, Δt) \\
U^{n+1} &= \tfrac{1}{3} U^n + \tfrac{2}{3} \Phi(U^{(2)}; \, Δt)
\end{aligned}
```

where ``\Phi`` denotes the forward Euler + acoustic subcycling stage operator.

### Stage-frozen reference state and perturbation variables

Following [Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007), the acoustic loop advances **perturbation variables** relative to the stage-level state, not the full fields directly. At the start of each RK stage, the reference state is frozen:

```math
\bar{ρ} = ρ^t, \qquad \bar{χ} = χ^t, \qquad \bar{s} = \bar{χ}/\bar{ρ}, \qquad \bar{\boldsymbol{m}} = (\rho \boldsymbol{u})^t .
```

Perturbation variables are defined as deviations from this reference:

```math
ρ'' = ρ - \bar{ρ}, \qquad \boldsymbol{m}'' = \boldsymbol{m} - \bar{\boldsymbol{m}}, \qquad χ'' = χ - \bar{χ} .
```

These start at zero and remain small (acoustic-amplitude) during the substep loop. The perturbation pressure is:

```math
p'' = ψ \, ρ'', \qquad \text{where} \quad ψ = R^m T
```

is held fixed during the substep loop.

The slow tendencies ``R^t`` (the full right-hand side evaluated at the stage-level state) are also computed once and held fixed. For momentum, ``R^t_{\boldsymbol{m}}`` excludes the pressure gradient and buoyancy (which are handled as fast terms). For density and thermodynamic variable, ``R^t`` is the full tendency including advection.

### Forward-backward acoustic substep loop

Within each RK stage, the acoustic substep loop iterates ``N_τ`` times with time step ``Δτ = Δt_{\mathrm{stage}} / N_τ``. Following CM1's convention:

| RK Stage | Stage ``Δt`` | Substeps ``N_τ`` |
|----------|-------------|----------|
| 1 | ``Δt`` | ``N_s/3`` |
| 2 | ``Δt/4`` | ``N_s/2`` |
| 3 | ``2Δt/3`` | ``N_s`` |

Each substep advances the perturbation variables:

**(A) Forward step --- perturbation momentum:**

```math
\boldsymbol{m}''^{\,τ+Δτ} = \boldsymbol{m}''^{\,τ} + Δτ \left( R^t_{\boldsymbol{m}} - \boldsymbol{\nabla} p''^{\,τ} - g ρ''^{\,τ} \hat{\boldsymbol{z}} \right)
```

The slow tendency ``R^t_{\boldsymbol{m}}`` includes advection and Coriolis at the stage level. The fast terms (perturbation pressure gradient and buoyancy) use the current perturbation density ``ρ''``.

**(B) Backward step --- perturbation density:**

```math
ρ''^{\,τ+Δτ} = ρ''^{\,τ} + Δτ \left( R^t_ρ - \boldsymbol{\nabla \cdot}\, \boldsymbol{m}''^{\,τ+Δτ} \right)
```

Only the **perturbation momentum** divergence appears --- not the full momentum. This eliminates double-counting of the advective velocity contribution already in ``R^t_ρ``.

**(C) Backward step --- perturbation thermodynamic variable:**

```math
χ''^{\,τ+Δτ} = χ''^{\,τ} + Δτ \left( R^t_χ - \bar{s} \, \boldsymbol{\nabla \cdot}\, \boldsymbol{m}''^{\,τ+Δτ} \right)
```

where ``\bar{s} = \bar{χ}/\bar{ρ}`` is the stage-frozen specific thermodynamic variable. For the potential temperature formulation (``χ = ρθ``), the compression source ``Π^{\mathrm{ac}} = 0``.

**(D) Recover full fields** at the end of the substep loop:

```math
ρ = \bar{ρ} + ρ'', \qquad \boldsymbol{m} = \bar{\boldsymbol{m}} + \boldsymbol{m}'', \qquad χ = \bar{χ} + χ'' .
```

**(E) Accumulate time-averaged velocities** each substep for scalar transport:

```math
\bar{\boldsymbol{u}} = \frac{1}{N_τ} \sum_{n=1}^{N_τ} \frac{\bar{\boldsymbol{m}} + \boldsymbol{m}''^{(n)}}{\bar{ρ} + ρ''^{(n)}} .
```

These time-averaged velocities are used for tracer advection in the outer RK loop.

### Vertically implicit solve (optional)

When ``Δz \ll Δx``, the explicit vertical acoustic step restricts the substep size ``Δτ < Δz / c_s``, which can be severe. Both CM1 (`sound.F`) and WRF (`advance_w` in `module_small_step_em.F`) treat vertical acoustic propagation implicitly.

Breeze supports this via the optional [`VerticallyImplicit`](@ref Breeze.CompressibleEquations.VerticallyImplicit) type:

```julia
# Explicit vertical step (default):
dynamics = CompressibleDynamics(time_discretization=SplitExplicitTimeDiscretization())
model = AtmosphereModel(grid; dynamics)

# Implicit vertical step with off-centering α = 0.5 (Crank-Nicolson):
dynamics = CompressibleDynamics(time_discretization=SplitExplicitTimeDiscretization(VerticallyImplicit(0.5)))
model = AtmosphereModel(grid; dynamics)
```

The implicit system couples the vertical momentum equation (which depends on ``∂ρ'/∂z``) with the continuity equation (which depends on ``∂(ρw)/∂z``). Eliminating ``ρ`` yields a tridiagonal system for ``ρw`` that is solved each substep via `BatchedTridiagonalSolver`. The off-centering parameter ``α`` controls acoustic damping: ``α = 0.5`` is Crank-Nicolson (second-order, undamped); ``α > 0.5`` damps vertically-propagating acoustic modes, following [Durran and Klemp (1983)](@cite DurranKlemp1983).

## Comparison with anelastic dynamics

| Property | [`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) | [`CompressibleDynamics`](@ref) |
|----------|-------------------|----------------------|
| Acoustic waves | Filtered | Resolved |
| Density | Reference ``ρᵣ(z)`` only | Prognostic ``ρ(x,y,z,t)`` |
| Pressure | Solved from Poisson equation | Computed from equation of state |
| Time step | Limited by advective CFL | Advective CFL (split-explicit) or acoustic CFL (explicit) |
| Typical applications | LES, mesoscale | Acoustic studies, validation |
