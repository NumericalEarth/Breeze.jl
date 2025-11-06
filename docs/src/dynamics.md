# Dynamics

This page summarizes the governing equations behind Breeze’s atmospheric dynamics and the anelastic formulation used by `AtmosphereModel`, following the thermodynamically consistent framework of Pauluis (2008) [Pauluis2008](@citet).

We begin with the compressible Navier–Stokes momentum equations and reduce them to an anelastic, conservative form. We then introduce the moist static energy equation and outline the time-discretized pressure correction used to enforce the anelastic constraint.

## Compressible momentum equations

Let ``ρ`` denote density, ``\boldsymbol{u}`` velocity, ``p`` pressure, ``\boldsymbol{f}`` non-pressure body forces (e.g., Coriolis), and ``\boldsymbol{\tau}`` subgrid/viscous stresses. With gravity ``g \hat{\boldsymbol{z}}``, the inviscid compressible equations in flux form are

```math
\begin{aligned}
&\text{Mass:} && \partial_t \rho + \nabla\!\cdot(\rho\,\boldsymbol{u}) = 0. \\
&\text{Momentum:} && \partial_t(\rho\,\boldsymbol{u}) + \nabla\!\cdot(\rho\,\boldsymbol{u}\,\boldsymbol{u}) + \nabla p 
\;=\; \rho\,g\,\hat{\boldsymbol{z}} + \rho\,\boldsymbol{f} + \nabla\!\cdot\boldsymbol{\tau}. 
\end{aligned}
```

For moist flows we also track total water (vapor + condensates) via

```math
\partial_t(\rho q^t) + \nabla\!\cdot(\rho q^t\,\boldsymbol{u}) = S_q ,
```

where ``q^t`` is total specific humidity and ``S_q`` accounts for sources/sinks from microphysics and boundary fluxes.

Thermodynamic relations (mixture gas constant ``R^{m}``, heat capacity ``c^{pm}``, Exner function, etc.) are summarized on the Thermodynamics page.

## Anelastic approximation

To filter acoustic waves while retaining compressibility effects in buoyancy and thermodynamics, we linearize about a hydrostatic, horizontally uniform reference state ``(p_r(z), \rho_r(z))`` with constant reference potential temperature ``\theta_r``. The key assumptions are

- Small Mach number and small relative density perturbations except in buoyancy.
- Hydrostatic reference balance: ``\partial_z p_r = -\rho_r g``.
- Mass flux divergence constraint: ``\nabla\!\cdot(\rho_r\,\boldsymbol{u}) = 0``.

Define the specific volume of moist air and its reference value as

```math
\alpha = \frac{R^{m} T}{p_r} , \qquad \alpha_{r} = \frac{R^{d}\,\theta_r}{p_r} ,
```

where ``R^{m}`` is the mixture gas constant and ``R^{d}`` is the dry-air gas constant. The buoyancy appearing in the vertical momentum is

```math
b \equiv g\,\frac{\alpha - \alpha_{r}}{\alpha_{r}} .
```

## Conservative anelastic system

With ``\rho_r(z)`` fixed by the reference state, the prognostic equations advanced in Breeze are written in conservative form for the ``\rho_r``-weighted fields:

- Continuity (constraint):

```math
\nabla\!\cdot(\rho_r\,\boldsymbol{u}) = 0 .
```

- Momentum:

```math
\partial_t(\rho_r\,\boldsymbol{u}) + \nabla\!\cdot(\rho_r\,\boldsymbol{u}\,\boldsymbol{u}) 
\;=\; -\,\rho_r\,\nabla \phi + \rho_r\,b\,\hat{\boldsymbol{z}} + \rho_r\,\boldsymbol{f} + \nabla\!\cdot\boldsymbol{\tau} ,
```

where ``\phi`` is a nonhydrostatic pressure correction potential defined by the projection step (see below). Pressure is decomposed as ``p = p_r(z) + p_h'(x,y,z,t) + p_n``, where ``p_h'`` is a hydrostatic anomaly (obeying ``\partial_z p_h' = -\rho_r b``) and ``p_n`` is the nonhydrostatic component responsible for enforcing the anelastic constraint. In the discrete formulation used here, ``\phi`` coincides with the pressure correction variable.

- Total water:

```math
\partial_t(\rho_r q^t) + \nabla\!\cdot(\rho_r q^t\,\boldsymbol{u}) = S_q .
```

### Moist static energy (Pauluis, 2008)

Following [Pauluis2008](@citet), Breeze advances a conservative moist static energy density

```math
\rho_r e \equiv \rho_r\,c^{pd}\,\theta ,
```

where ``c^{p^{d}}`` is the dry-air heat capacity at constant pressure and ``\theta`` is the (moist) potential temperature. The prognostic equation reads

```math
\partial_t(\rho_r e) + \nabla\!\cdot(\rho_r e\,\boldsymbol{u}) 
\;=\; \rho_r\,w\,b \; + \; S_e ,
```

with vertical velocity ``w``, buoyancy ``b`` as above, and ``S_e`` including microphysical and external energy sources/sinks. The ``\rho_r w b`` term is the buoyancy flux that links the energy and momentum budgets in the anelastic limit.

Thermodynamic closures needed for ``R^{m}``, ``c^{pm}`` and the Exner function ``\Pi = (p_r/p_0)^{R^{m}/c^{pm}}`` are given in Thermodynamics.

## Time discretization and pressure correction

Breeze uses an explicit multi-stage time integrator for advection, Coriolis, buoyancy, forcing, and tracer terms, coupled with a projection step to enforce the anelastic constraint at each substep. Denote the predicted momentum by ``\widetilde{(\rho_r\,\boldsymbol{u})}``. The projection is

1. Solve the variable-coefficient Poisson problem for the pressure correction potential ``\phi``:

```math
\nabla\!\cdot\big( \rho_r \, \nabla \phi \big) 
\;=\; \frac{1}{\Delta t} \, \nabla\!\cdot\,\widetilde{(\rho_r\,\boldsymbol{u})} ,
```

   with periodic lateral boundaries and homogeneous Neumann boundary conditions in ``z``.

2. Update momentum to enforce ``\nabla\!\cdot(\rho_r\,\boldsymbol{u}^{\,n+1})=0``:

```math
\rho_r\,\boldsymbol{u}^{\,n+1} \;=\; \widetilde{(\rho_r\,\boldsymbol{u})} \; - \; \Delta t\, \rho_r \, \nabla \phi .
```

In Breeze this projection is implemented as a Fourier–tridiagonal solve in the vertical with variable ``\rho_r(z)``, aligning with the hydrostatic reference state. The hydrostatic pressure anomaly ``p_h'`` can be obtained diagnostically by vertical integration of buoyancy and used when desired to separate hydrostatic and nonhydrostatic pressure effects.

## Symbols and closures used here

- ``\rho_r(z)``, ``p_r(z)``: Reference density and pressure satisfying hydrostatic balance for a constant ``\theta_r``.
- ``\alpha = R^{m} T/p_r``, ``\alpha_{r} = R^{d}\,\theta_r/p_r``: Specific volume and its reference value.
- ``b = g (\alpha - \alpha_{r})/\alpha_{r}``: Buoyancy.
- ``e = c^{pd}\,\theta``: Energy variable used for moist static energy in the conservative equation.
- ``q^t``: Total specific humidity (vapor + condensates).
- ``\phi``: Nonhydrostatic pressure correction potential used by the projection.

See Thermodynamics for definitions of ``R^{m}(q)``, ``c^{pm}(q)``, and ``\Pi``.

## References

```@bibliography
```
