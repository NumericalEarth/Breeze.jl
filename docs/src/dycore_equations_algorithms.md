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

In addition to mass and momentum, Breeze advances a thermodynamic prognostic variable ``ρᵡ`` — the density-weighted form of a thermodynamic state function ``χ(T, p, q)`` — in conservative (flux) form:

```math
\partial_t (ρ χ) + \boldsymbol{\nabla \cdot}\, (ρ χ \boldsymbol{u}) = ρ \, \mathcal{E}_χ \frac{\mathrm{D} \ln p}{\mathrm{D} t} + S_χ ,
```

where the first term on the right is the expansion source — ``\mathcal{E}_χ = (∂χ / ∂\ln p)_{η, q}`` is the change of ``χ`` along an adiabat at fixed composition, derived for a general ``χ`` in the section on [the prognostic equation of a thermodynamic state function](@ref "The prognostic equation of a thermodynamic state function") — and ``S_χ`` collects the diabatic and diffusive sources: heating, sedimentation and subgrid fluxes. No choice of ``χ`` makes both the expansion source and the phase-change source vanish. Breeze's formulations put the explicit source on expansion, so that every phase change is exact and microphysics schemes act on the temperature alone.

Breeze supports two concrete choices for ``χ``:

- **Liquid-ice potential temperature** (``ρᵡ = ρ θˡⁱ``, the default): ``θˡⁱ = H / (\hat{c}ᵖ \hat{Π})`` is the enthalpy ``H = c^{pm} T - \mathscr{L}^l_0 q^l - \mathscr{L}^i_0 q^i`` in temperature units, referred to the standard pressure with the vapor-equivalent Exner function ``\hat{Π} = (p / pˢᵗ)^{\hat{κ}}``; see [Liquid-ice potential temperature](@ref liquid-ice-potential-temperature). It is exactly invariant under phase change, and its expansion source ``\mathcal{E}_θ = (R^m T - \hat{R} \hat{Π} θˡⁱ) / (\hat{c}ᵖ \hat{Π})`` is proportional to the condensate, so that ``θˡⁱ`` is materially conserved under adiabatic motion in clear air.

- **Static energy** (``ρᵡ = ρ s``): The moist static energy ``s = H + g z`` adds the gravitational potential energy, with ``\mathscr{L}^l_0`` and ``\mathscr{L}^i_0`` the latent heats of condensation and deposition extrapolated to absolute zero. Its source is the pressure work ``\mathrm{D} p / \mathrm{D} t`` plus the geopotential change ``ρ g w``, which on the anelastic core combine into the buoyancy flux ``- ρᵣ w b``; see [Anelastic dynamics](@ref Anelastic-section).

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
- ``θ``: Liquid-ice potential temperature, shorthand for ``θˡⁱ``
- ``χ``: Thermodynamic state function (``θˡⁱ`` or ``s``); ``ρᵡ`` is its density-weighted prognostic (``ρθ`` or ``ρs``)

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
- ``\hat{c}ᵖ = q^d c^{pd} + q^t c^{pv}``, ``\hat{R} = q^d R^d + q^t R^v``: Vapor-equivalent heat capacity and gas constant, the values the mixture would have if all of its water were vapor
- ``\hat{Π} = (p / pˢᵗ)^{\hat{κ}}``, ``\hat{κ} = \hat{R} / \hat{c}ᵖ``: Vapor-equivalent Exner function and its exponent
- ``\mathcal{E}_χ``: Expansion coefficient of the thermodynamic variable, ``(∂χ / ∂\ln p)_{η, q}``
- ``c^{ac} = \sqrt{γ^m R^m T}``: Acoustic sound speed, where ``γ^m = c^{pm} / c^{vm}``
- ``b``: Buoyancy

See [Thermodynamics](@ref Thermodynamics-section) for full definitions of ``R^m(q)``, ``c^{pm}(q)``, and ``\hat{Π}``.
