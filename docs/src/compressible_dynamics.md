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

- [`SplitExplicit`](@ref Breeze.CompressibleEquations.SplitExplicit): Acoustic substepping with separate slow/fast tendency splitting. This allows advective CFL time steps (~10-20 m/s) instead of acoustic CFL time steps (~340 m/s).

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

### Stage-frozen reference state and perturbations

At the start of each RK stage, the reference state ``\bar{U}`` is frozen:

```math
\bar{ρ} = ρ^{\mathrm{stage}}, \qquad \bar{χ} = χ^{\mathrm{stage}}, \qquad \bar{s} = \bar{χ}/\bar{ρ} .
```

During acoustic substeps, the perturbation pressure ``p'`` is computed from a linearized equation of state:

```math
p' = ψ \, (ρ - \bar{ρ}), \qquad \text{where} \quad ψ = R^m T
```

is held fixed during the substep loop. Using perturbation quantities for the pressure gradient avoids amplifying hydrostatic imbalance.

### Forward-backward acoustic substep loop

Within each RK stage, the acoustic substep loop iterates ``N_s`` times with time step ``Δτ = Δt_{\mathrm{stage}} / N_s``. Following CM1's convention:

| RK Stage | Stage ``Δt`` | Substeps |
|----------|-------------|----------|
| 1 | ``Δt/3`` | ``N_s/3`` |
| 2 | ``Δt/2`` | ``N_s/2`` |
| 3 | ``Δt`` | ``N_s`` |

Each substep applies a forward-backward scheme:

**(A) Forward step --- momentum update:**

```math
(\rho \boldsymbol{u})^{τ+Δτ} = (\rho \boldsymbol{u})^{τ} - Δτ \, \boldsymbol{\nabla} p'^{\,τ} + Δτ \, b'^{\,τ} \hat{\boldsymbol{z}} + Δτ \, G^s_{\rho \boldsymbol{u}} ,
```

where the slow momentum tendencies ``G^s_{\rho \boldsymbol{u}}`` are applied once at the start of the stage.

**(B) Update velocities:**

```math
\boldsymbol{u}^{τ+Δτ} = (\rho \boldsymbol{u})^{τ+Δτ} / ρ^τ .
```

**(C) Backward step --- density update:**

```math
ρ^{τ+Δτ} = ρ^{τ} - Δτ \, \boldsymbol{\nabla \cdot}\, \boldsymbol{m}^{τ+Δτ} ,
```

using the **newly computed** momentum (backward coupling). Divergence damping nudges density toward the stage-frozen reference: ``ρ \leftarrow ρ - κ^d (ρ - \bar{ρ})``.

**(D) Backward step --- thermodynamic variable update:**

Following [Klemp, Skamarock, and Dudhia (2007)](@cite KlempSkamarockDudhia2007) Eq. 15, the thermodynamic variable is updated using a linearized flux divergence:

```math
χ^{τ+Δτ} = χ^{τ} - Δτ \, \bar{s} \, \boldsymbol{\nabla \cdot}\, \boldsymbol{m}^{τ+Δτ} + Δτ \, G^s_χ ,
```

where ``\bar{s}`` is the stage-frozen specific thermodynamic variable and ``G^s_χ`` is the slow thermodynamic tendency. The linearization advects the reference-level specific variable by the current momentum, following WRF's approach.

For the potential temperature formulation (``χ = ρθ``), the compression source ``Π^{\mathrm{ac}} = 0``.

**(E) Accumulate time-averaged velocities:**

```math
\bar{\boldsymbol{u}} = \frac{1}{N_s} \sum_{n=1}^{N_s} \boldsymbol{u}^{(n)} .
```

These time-averaged velocities are used for scalar (tracer) transport in the outer RK loop, ensuring mass-consistent advection.

### Implicit vertical solve (planned)

Both CM1 (`sound.F`) and WRF (`advance_w` in `module_small_step_em.F`) treat vertical acoustic propagation implicitly using a tridiagonal solver. This removes the vertical CFL restriction on the acoustic substep size, which can be severe when ``Δz \ll Δx``. Breeze currently uses an explicit vertical step; the implicit solve is planned for a future iteration.

## Comparison with anelastic dynamics

| Property | [`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) | [`CompressibleDynamics`](@ref) |
|----------|-------------------|----------------------|
| Acoustic waves | Filtered | Resolved |
| Density | Reference ``ρᵣ(z)`` only | Prognostic ``ρ(x,y,z,t)`` |
| Pressure | Solved from Poisson equation | Computed from equation of state |
| Time step | Limited by advective CFL | Advective CFL (split-explicit) or acoustic CFL (explicit) |
| Typical applications | LES, mesoscale | Acoustic studies, validation |
