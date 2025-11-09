# [Dycore equations and algorithms](@id Dycore-section)

This section summarizes the governing equations behind Breeze’s atmospheric dynamics and the anelastic formulation used by [`AtmosphereModel`](@ref), following the thermodynamically consistent framework of [Pauluis2008](@citet).

We begin with the compressible Navier-Stokes momentum equations and reduce them to an anelastic, conservative form.
We then introduce the moist static energy equation and outline the time-discretized pressure correction used to enforce the anelastic constraint.

## Compressible momentum equations

Let ``ρ`` denote density, ``\boldsymbol{u}`` velocity, ``p`` pressure, ``\boldsymbol{f}`` non-pressure body forces (e.g., Coriolis), and ``\boldsymbol{\tau}`` subgrid/viscous stresses. With gravity ``- g \hat{\boldsymbol{z}}``, the inviscid compressible equations in flux form are

```math
\begin{aligned}
&\text{Mass:} && \partial_t ρ + \boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u}) = 0 ,\\
&\text{Momentum:} && \partial_t(ρ \boldsymbol{u}) + \boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u} \boldsymbol{u}) + \boldsymbol{\nabla} p = - ρ g \hat{\boldsymbol{z}} + ρ \boldsymbol{f} + \boldsymbol{\nabla \cdot}\, \boldsymbol{\tau} .
\end{aligned}
```

Notation ``\boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u} \boldsymbol{u})`` above denotes a vector whose components are
``[\boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u} \boldsymbol{u})]_i = \boldsymbol{\nabla \cdot}\, (ρ u_i \boldsymbol{u})``.

For moist flows we also track total water (vapor + condensates) via

```math
\partial_t(ρ q^t) + \boldsymbol{\nabla \cdot}\, (ρ q^t \boldsymbol{u}) = S_q ,
```

where ``q^t`` is total specific humidity and ``S_q`` accounts for sources/sinks from microphysics and boundary fluxes.

Thermodynamic relations (mixture gas constant ``R^m``, heat capacity ``c^{pm}``, Exner function, etc.)
are summarized in the [Thermodynamics](@ref Thermodynamics-section) section.

## Anelastic approximation

To filter acoustic waves while retaining compressibility effects in buoyancy and thermodynamics, we linearize about a hydrostatic, horizontally uniform reference state ``(p_r(z), ρ_r(z))`` with constant reference potential temperature ``\theta_r``. The key assumptions are

- Small Mach number and small relative density perturbations except in buoyancy.
- Hydrostatic reference balance: ``\partial_z p_r = -ρ_r g``.
- Mass flux divergence constraint: ``\boldsymbol{\nabla \cdot}\, (ρ_r\,\boldsymbol{u}) = 0``.

Define the specific volume of moist air and its reference value as

```math
α = \frac{R^m T}{p_r} , \qquad α_r = \frac{R^d θ_r}{p_r} ,
```

where ``R^m`` is the mixture gas constant and ``R^{d}`` is the dry-air gas constant. The buoyancy appearing in the vertical momentum is

```math
b ≡ g \frac{α - α_r}{α_r} .
```

## Conservative anelastic system

With ``ρ_r(z)`` fixed by the reference state, the prognostic equations advanced in Breeze are written in conservative form for the ``ρ_r``-weighted fields:

- Continuity (constraint):

```math
\boldsymbol{\nabla \cdot}\, (ρ_r \boldsymbol{u}) = 0 .
```

- Momentum:

```math
\partial_t(ρ_r \boldsymbol{u}) + \boldsymbol{\nabla \cdot}\, (ρ_r \boldsymbol{u} \boldsymbol{u}) = - ρ_r \boldsymbol{\nabla} \phi + ρ_r \, b \hat{\boldsymbol{z}} + ρ_r \boldsymbol{f} + \boldsymbol{\nabla \cdot}\, \boldsymbol{\tau} ,
```

where ``\phi`` is a nonhydrostatic pressure correction potential defined by the projection step (see below). Pressure is decomposed as ``p = p_r(z) + p_h'(x, y, z, t) + p_n``, where ``p_h'`` is a hydrostatic anomaly (obeying ``\partial_z p_h' = -ρ_r b``) and ``p_n`` is the nonhydrostatic component responsible for enforcing the anelastic constraint. In the discrete formulation used here, ``\phi`` coincides with the pressure correction variable.

- Total water:

```math
\partial_t(ρ_r q^t) + \boldsymbol{\nabla \cdot}\, (ρ_r q^t \boldsymbol{u}) = S_q .
```

### Moist static energy (Pauluis, 2008)

Following [Pauluis2008](@citet), Breeze advances a conservative moist static energy density

```math
ρ_r e ≡ ρ_r c^{pd} θ ,
```

where ``c^{p d}`` is the dry-air heat capacity at constant pressure and ``θ`` is the (moist) potential temperature. The prognostic equation reads

```math
\partial_t(ρ_r e) + \boldsymbol{\nabla \cdot}\, (ρ_r e \boldsymbol{u}) = ρ_r w b + S_e ,
```

with vertical velocity ``w``, buoyancy ``b`` as above, and ``S_e`` including microphysical and external energy sources/sinks.
The ``ρ_r w b`` term is the buoyancy flux that links the energy and momentum budgets in the anelastic limit.

Thermodynamic closures needed for ``R^m``, ``c^{pm}`` and the Exner function ``Π = (p_r / p_0)^{R^m / c^{pm}}`` are given in [Thermodynamics](@ref Thermodynamics-section) section.

## Time discretization and pressure correction

Breeze uses an explicit multi-stage time integrator for advection, Coriolis, buoyancy, forcing, and tracer terms, coupled with a projection step to enforce the anelastic constraint at each substep. Denote the predicted momentum by ``\widetilde{(ρ_r \boldsymbol{u})}``. The projection is

1. Solve the variable-coefficient Poisson problem for the pressure correction potential ``\phi``:

   ```math
   \boldsymbol{\nabla \cdot}\, \big( ρ_r \, \boldsymbol{\nabla} \phi \big) = \frac{1}{Δt} \, \boldsymbol{\nabla \cdot}\, \widetilde{(ρ_r \boldsymbol{u})} ,
   ```

   with periodic lateral boundaries and homogeneous Neumann boundary conditions in ``z``.

2. Update momentum to enforce ``\boldsymbol{\nabla \cdot}\, (ρ_r \boldsymbol{u}^{n+1}) = 0``:

   ```math
   ρ_r \boldsymbol{u}^{n+1} = \widetilde{(ρ_r \boldsymbol{u})} - Δt \, ρ_r \boldsymbol{\nabla} \phi .
   ```

In Breeze this projection is implemented as a Fourier–tridiagonal solve in the vertical with variable ``ρ_r(z)``, aligning with the hydrostatic reference state. The hydrostatic pressure anomaly ``p_h'`` can be obtained diagnostically by vertical integration of buoyancy and used when desired to separate hydrostatic and nonhydrostatic pressure effects.

## Symbols and closures used here

- ``ρ_r(z)``, ``p_r(z)``: Reference density and pressure satisfying hydrostatic balance for a constant ``θ_r``.
- ``α = R^m T / p_r``, ``α_r = R^d θ_r / p_r``: Specific volume and its reference value.
- ``b = g (α - α_r) / α_r``: Buoyancy.
- ``e = c^{pd} \, θ``: Energy variable used for moist static energy in the conservative equation.
- ``q^t``: Total specific humidity (vapor + condensates).
- ``\phi``: Nonhydrostatic pressure correction potential used by the projection.

See [Thermodynamics](@ref Thermodynamics-section) section for definitions of ``R^m(q)``, ``c^{pm}(q)``, and ``Π``.
