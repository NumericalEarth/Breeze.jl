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
It uses a three-stage Runge-Kutta outer loop for slow tendencies
with an inner forward-backward acoustic substep loop for fast tendencies.

Two RK3 outer loop variants are available:

- **Wicker-Skamarock RK3** ([`AcousticRungeKutta3`](@ref), default): Uses stage fractions ``Δt/3, Δt/2, Δt`` with simple recovery ``U = U^0 + U''``. Supports the base-state pressure correction for temperature-driven dynamics.
- **SSP RK3** ([`AcousticSSPRungeKutta3`](@ref)): Uses convex combinations with weights ``α = 1, 1/4, 2/3``. Does **not** support the base-state pressure correction.

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

### Wicker-Skamarock RK3 outer loop

The default [`AcousticRungeKutta3`](@ref) time stepper uses stage fractions ``β = 1/3, 1/2, 1``:

```math
\begin{aligned}
U^{(1)} &= U^n + \tfrac{Δt}{3} \, R(U^n) \\
U^{(2)} &= U^n + \tfrac{Δt}{2} \, R(U^{(1)}) \\
U^{n+1} &= U^n + Δt \, R(U^{(2)})
\end{aligned}
```

Each stage resets to the initial state ``U^n`` and advances by ``β \, Δt``. The acoustic substep size varies per stage: ``Δτ = β \, Δt / N_s``, where ``N_s`` is the number of substeps. Recovery at the end of each stage is simply ``U = U^0 + U''`` (no convex combination), which is compatible with the base-state pressure correction.

### SSP RK3 outer loop (alternative)

The [`AcousticSSPRungeKutta3`](@ref) time stepper uses the SSP RK3 scheme in Shu-Osher form:

```math
\begin{aligned}
U^{(1)} &= \Phi(U^n; \, Δt) \\
U^{(2)} &= \tfrac{3}{4} U^n + \tfrac{1}{4} \Phi(U^{(1)}; \, Δt) \\
U^{n+1} &= \tfrac{1}{3} U^n + \tfrac{2}{3} \Phi(U^{(2)}; \, Δt)
\end{aligned}
```

where ``\Phi`` denotes the forward Euler + acoustic subcycling stage operator. The convex combination introduces acoustic mode interference that prevents use of the base-state pressure correction (see below).

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

Within each RK stage, the acoustic substep loop iterates ``N_s`` times with time step ``Δτ = β \, Δt / N_s`` (for Wicker-Skamarock RK3) or ``Δτ = Δt / N_s`` (for SSP RK3).

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

### Base-state pressure correction

The perturbation pressure ``p'' = ψ \, ρ''`` only captures pressure changes from density variations *during* the acoustic loop (since ``ρ'' = 0`` at the start of each stage and ``ψ`` is frozen). This misses the temperature-driven pressure gradient.

To see why, decompose the full horizontal pressure gradient:

```math
\frac{\partial p}{\partial x} = \frac{\partial (ψ ρ)}{\partial x}
= \underbrace{ψ \frac{\partial ρ}{\partial x}}_{\text{density-driven}} + \underbrace{ρ \frac{\partial ψ}{\partial x}}_{\text{temperature-driven}}
```

The acoustic loop resolves the density-driven term through ``ψ \, \partial ρ'' / \partial x``. But the temperature-driven term ``ρ \, \partial ψ / \partial x`` --- which arises from horizontal variations in ``ψ = R^m T`` (e.g., from potential temperature perturbations) --- is not captured because ``ψ`` is frozen and ``ρ''`` starts at zero.

For problems driven by temperature perturbations (such as inertia-gravity waves triggered by a ``θ'`` anomaly), this missing term means the split-explicit scheme cannot generate the correct pressure gradients, and vertical velocity remains zero.

The fix is to add the temperature-driven pressure gradient ``ρ \, \partial ψ / \partial x`` to the slow momentum tendencies:

```math
R^t_{\boldsymbol{m}} \mathrel{-}= ρ \, \boldsymbol{\nabla}_h ψ
```

where ``\boldsymbol{\nabla}_h`` denotes the horizontal gradient only. The vertical pressure gradient and buoyancy are handled entirely by the acoustic forward step through ``ψ \, \partial ρ'' / \partial z`` and ``g \, ρ''``.

This correction is activated by providing `reference_potential_temperature` when constructing `CompressibleDynamics`:

```julia
dynamics = CompressibleDynamics(;
    surface_pressure = p₀,
    time_discretization = SplitExplicitTimeDiscretization(
        VerticallyImplicit(0.5),
        substeps = 12,
        divergence_damping_coefficient = 0.2),
    reference_potential_temperature = θ₀)
```

!!! note "Requirements for the base-state pressure correction"
    The correction requires two conditions for stability:

    1. **Vertically implicit substepping**: Use `VerticallyImplicit(α)` as the first argument to `SplitExplicitTimeDiscretization`. The horizontal correction drives vertical motion that explicit vertical stepping cannot stabilize.

    2. **Sufficient divergence damping**: The damping coefficient ``κ_d`` and number of substeps ``N_s`` must satisfy

       ```math
       (1 - κ_d)^{N_s} \lesssim 0.1
       ```

       Practical combinations include:

       | ``N_s`` | ``κ_d`` | ``(1-κ_d)^{N_s}`` |
       |---------|---------|-----------------|
       | 12      | 0.2     | 0.069           |
       | 24      | 0.1     | 0.080           |
       | 48      | 0.05    | 0.085           |

!!! note "Only available with Wicker-Skamarock RK3"
    The base-state pressure correction is only supported by the [`AcousticRungeKutta3`](@ref) time stepper (the default). The SSP RK3 convex combination creates acoustic mode interference that makes the correction unstable.

### Divergence damping

Divergence damping suppresses spurious acoustic oscillations by multiplicatively
damping the perturbation density each substep:

```math
ρ''^{\,τ+Δτ} \mathrel{*}= (1 - κ_d)
```

Over ``N_s`` substeps this accumulates to a total damping factor of ``(1 - κ_d)^{N_s}``.
The default value is ``κ_d = 0.05``. When using the base-state pressure correction,
a larger value (e.g., ``κ_d = 0.2``) may be needed to satisfy the stability constraint.

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
