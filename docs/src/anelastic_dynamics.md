# [Anelastic dynamics](@id Anelastic-section)

[`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) filters acoustic waves by linearizing about a hydrostatic reference state, following the thermodynamically consistent framework of [Pauluis2008](@citet). This formulation is suitable for most large-eddy simulations and mesoscale applications.

## Anelastic approximation

To filter acoustic waves while retaining compressibility effects in buoyancy and thermodynamics, we linearize about a hydrostatic, horizontally uniform reference state ``(pᵣ(z), ρᵣ(z))`` with constant reference potential temperature ``θᵣ``. The key assumptions are

- Small Mach number and small relative density perturbations except in buoyancy.
- Hydrostatic reference balance: ``\partial_z pᵣ = -ρᵣ g``.
- Mass flux divergence constraint: ``\boldsymbol{\nabla \cdot}\, (ρᵣ\,\boldsymbol{u}) = 0``.

Define the specific volume of moist air and its reference value as

```math
α = \frac{R^m T}{pᵣ} , \qquad αᵣ = \frac{R^d θᵣ}{pᵣ} ,
```

where ``R^m`` is the mixture gas constant and ``R^{d}`` is the dry-air gas constant. The buoyancy appearing in the vertical momentum is

```math
b ≡ g \frac{α - αᵣ}{αᵣ} .
```

## Conservative anelastic system

With ``ρᵣ(z)`` fixed by the reference state, the prognostic equations advanced in Breeze are written in conservative form for the ``ρᵣ``-weighted fields:

- Continuity (constraint):

```math
\boldsymbol{\nabla \cdot}\, (ρᵣ \boldsymbol{u}) = 0 .
```

- Momentum:

```math
\partial_t(ρᵣ \boldsymbol{u}) + \boldsymbol{\nabla \cdot}\, (ρᵣ \boldsymbol{u} \boldsymbol{u}) = - ρᵣ \boldsymbol{\nabla} \phi + ρᵣ \, b \hat{\boldsymbol{z}} + ρᵣ \boldsymbol{f} + \boldsymbol{\nabla \cdot}\, \boldsymbol{\mathcal{T}} ,
```

where ``\phi`` is a nonhydrostatic pressure correction potential defined by the projection step (see below). Pressure is decomposed as ``p = pᵣ(z) + p_h'(x, y, z, t) + p_n``, where ``p_h'`` is a hydrostatic anomaly (obeying ``\partial_z p_h' = -ρᵣ b``) and ``p_n`` is the nonhydrostatic component responsible for enforcing the anelastic constraint. In the discrete formulation used here, ``\phi`` coincides with the pressure correction variable.

- Total water:

```math
\partial_t(ρᵣ q^t) + \boldsymbol{\nabla \cdot}\, (ρᵣ q^t \boldsymbol{u}) = S_q .
```

## Moist static energy

Breeze advances a conservative moist static energy density

```math
ρᵣ e ≡ ρᵣ \left ( c^{pm} T + g z - \mathscr{L}^l_r q^l - \mathscr{L}^i_r q^i \right ),
```

where ``c^{p m}`` is the mixture heat capacity, ``T`` is temperature, ``g`` is gravitational acceleration,
``z`` is height,
``\mathscr{L}^l_r`` is the latent heat of condensation (vapor to liquid) at the energy reference temperature,
and
``\mathscr{L}^i_r`` is the latent heat of deposition (vapor to ice) at the energy reference temperature.

According to [Pauluis2008](@citet), the moist static energy obeys

```math
\partial_t(ρᵣ e) + \boldsymbol{\nabla \cdot}\, (ρᵣ e \boldsymbol{u}) = ρᵣ w b + S_e ,
```

with vertical velocity ``w``, buoyancy ``b`` as above, and ``S_e`` including microphysical and external energy sources/sinks.
The ``ρᵣ w b`` term is the buoyancy flux that links the energy and momentum budgets in the anelastic limit.

Thermodynamic closures needed for ``R^m``, ``c^{pm}`` and the Exner function ``Π = (pᵣ / p_0)^{R^m / c^{pm}}`` are given in [Thermodynamics](@ref Thermodynamics-section) section.

## Time discretization and pressure correction

The anelastic formulation uses a multi-stage time integrator for advection, Coriolis, buoyancy, forcing, and tracer terms, coupled with a projection step to enforce the anelastic constraint at each substep. Denote the predicted momentum by ``\widetilde{(ρᵣ \boldsymbol{u})}``. The projection is

1. Solve the variable-coefficient Poisson problem for the pressure correction potential ``\phi``:

   ```math
   \boldsymbol{\nabla \cdot}\, \big( ρᵣ \, \boldsymbol{\nabla} \phi \big) = \frac{1}{Δt} \, \boldsymbol{\nabla \cdot}\, \widetilde{(ρᵣ \boldsymbol{u})} ,
   ```

   with periodic lateral boundaries and homogeneous Neumann boundary conditions in ``z``.

2. Update momentum to enforce ``\boldsymbol{\nabla \cdot}\, (ρᵣ \boldsymbol{u}^{n+1}) = 0``:

   ```math
   ρᵣ \boldsymbol{u}^{n+1} = \widetilde{(ρᵣ \boldsymbol{u})} - Δt \, ρᵣ \boldsymbol{\nabla} \phi .
   ```

In Breeze this projection is implemented as a Fourier-tridiagonal solve in the vertical with variable ``ρᵣ(z)``, aligning with the hydrostatic reference state. The hydrostatic pressure anomaly ``p_h'`` can be obtained diagnostically by vertical integration of buoyancy and used when desired to separate hydrostatic and nonhydrostatic pressure effects.

## Symbols

- ``ρᵣ(z)``, ``pᵣ(z)``: Reference density and pressure satisfying hydrostatic balance for a constant ``θᵣ``.
- ``α = R^m T / pᵣ``, ``αᵣ = R^d θᵣ / pᵣ``: Specific volume and its reference value.
- ``b = g (α - αᵣ) / αᵣ``: Buoyancy.
- ``e``: Moist static energy.
- ``q^t``: Total specific humidity (vapor + condensates).
- ``\phi``: Nonhydrostatic pressure correction potential used by the projection.
