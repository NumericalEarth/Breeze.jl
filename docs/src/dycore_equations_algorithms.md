# [Governing equations](@id Dycore-section)

This section summarizes the governing equations behind Breeze's atmospheric dynamics used by [`AtmosphereModel`](@ref). Breeze supports two dynamical formulations:

- **[`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics)**: Filters acoustic waves by linearizing about a hydrostatic reference state, following the thermodynamically consistent framework of [Pauluis2008](@citet). Suitable for most large-eddy simulations and mesoscale applications. See the [Anelastic dynamics](@ref Anelastic-section) page for details.

- **[`CompressibleDynamics`](@ref)**: Solves the fully compressible Euler equations with prognostic density. Retains acoustic waves and supports split-explicit time integration for efficiency. See the [Compressible dynamics](@ref Compressible-section) page for details.

Both formulations share the same compressible Navier-Stokes equations as a starting point, with the anelastic formulation obtained as a special case through linearization.

## Compressible Navier-Stokes equations

Let ``ρ`` denote density, ``\boldsymbol{u}`` velocity, ``p`` pressure, ``\boldsymbol{f}`` non-pressure body forces (e.g., Coriolis), and ``\boldsymbol{\tau}`` the kinematic (per-mass) subgrid/viscous stresses. We denote the corresponding dynamic (per-volume) stresses by ``\boldsymbol{\mathcal{T}} = ρ \, \boldsymbol{\tau}``. With gravity ``- g \hat{\boldsymbol{z}}``, the compressible equations in flux form are

```math
\begin{aligned}
&\text{Mass:} && \partial_t ρ + \boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u}) = S_ρ ,\\
&\text{Momentum:} && \partial_t(ρ \boldsymbol{u}) + \boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u} \boldsymbol{u}) + \boldsymbol{\nabla} p = - ρ g \hat{\boldsymbol{z}} + ρ \boldsymbol{f} + \boldsymbol{\nabla \cdot}\, \boldsymbol{\mathcal{T}} .
\end{aligned}
```

Notation ``\boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u} \boldsymbol{u})`` above denotes a vector whose components are
``[\boldsymbol{\nabla \cdot}\, (ρ \boldsymbol{u} \boldsymbol{u})]_i = \boldsymbol{\nabla \cdot}\, (ρ u_i \boldsymbol{u})``.

## Thermodynamic equation

In addition to mass and momentum, Breeze advances a thermodynamic prognostic variable ``χ`` in conservative (flux) form:

```math
\partial_t χ + \boldsymbol{\nabla \cdot}\, (χ \boldsymbol{u}) = Π(ρ, χ, \ldots) \, \boldsymbol{\nabla \cdot \, u} + S_χ ,
```

where ``Π`` is a formulation-specific compression source and ``S_χ`` represents diabatic and diffusive sources.

Breeze supports two concrete choices for ``χ``:

- **Liquid-ice potential temperature density** (``χ = ρ θ^{li}``): The potential temperature is materially conserved under adiabatic motion, so ``Π = 0``. This is the simplest thermodynamic formulation.

- **Static energy density** (``χ = ρ e``): The moist static energy ``e = c^{pm} T + g z - \mathscr{L}^l_r q^l - \mathscr{L}^i_r q^i`` includes gravitational potential energy and latent heat. In this case ``Π \neq 0`` and encodes the pressure work term.

The thermodynamic equation couples to the momentum equation through the equation of state (below) and buoyancy.

## Moisture transport

For moist flows we track total water (vapor + condensates) via

```math
\partial_t(ρ q^t) + \boldsymbol{\nabla \cdot}\, (ρ q^t \boldsymbol{u}) = S_q ,
```

where ``q^t`` is total specific humidity and ``S_q`` accounts for sources/sinks from microphysics and boundary fluxes.

## Equation of state

Pressure is related to density and temperature through the ideal gas law for moist air:

```math
p = ρ R^m T ,
```

where ``R^m = (1 - q^t) R^d + q^v R^v`` is the mixture gas constant.

Thermodynamic relations (mixture gas constant ``R^m``, heat capacity ``c^{pm}``, Exner function, etc.) are summarized in the [Thermodynamics](@ref Thermodynamics-section) section.

## Symbols and notation

### Core variables
- ``ρ``: Density (prognostic for compressible; reference ``ρᵣ(z)`` for anelastic)
- ``\boldsymbol{u} = (u, v, w)``: Velocity
- ``\boldsymbol{m} = ρ \boldsymbol{u}``: Momentum
- ``p``: Pressure
- ``T``: Temperature
- ``θ``: Potential temperature
- ``χ``: Thermodynamic prognostic variable (``ρθ`` or ``ρe``)

### Moisture
- ``q^t``: Total specific humidity (vapor + condensates)
- ``q^v, q^l, q^i``: Vapor, liquid, and ice mass fractions
- ``R^m``: Mixture gas constant
- ``c^{pm}``: Mixture heat capacity at constant pressure

### Stresses and forces
- ``\boldsymbol{\tau}``: Kinematic (per-mass) subgrid/viscous stress tensor returned by Oceananigans closures.
- ``\boldsymbol{\mathcal{T}} = ρ \, \boldsymbol{\tau}``: Dynamic (per-volume) stress used in the momentum equation; Breeze computes flux divergences as ``\boldsymbol{\nabla\cdot}\, \boldsymbol{\mathcal{T}}``.
- ``\boldsymbol{f}``: Non-pressure body forces (Coriolis)

### Thermodynamic closures
- ``Π = (p / p_0)^{R^m / c^{pm}}``: Exner function
- ``\mathbb{C}^{ac} = \sqrt{γ^m R^m T}``: Acoustic sound speed, where ``γ^m = c^{pm} / c^{vm}``
- ``b``: Buoyancy

See [Thermodynamics](@ref Thermodynamics-section) for full definitions of ``R^m(q)``, ``c^{pm}(q)``, and ``Π``.
