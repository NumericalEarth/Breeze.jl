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
ρᵣ s ≡ ρᵣ \left ( c^{pm} T + g z - \mathscr{L}^l_r q^l - \mathscr{L}^i_r q^i \right ),
```

where ``c^{p m}`` is the mixture heat capacity, ``T`` is temperature, ``g`` is gravitational acceleration,
``z`` is height,
``\mathscr{L}^l_r`` is the latent heat of condensation (vapor to liquid) at the energy reference temperature,
and
``\mathscr{L}^i_r`` is the latent heat of deposition (vapor to ice) at the energy reference temperature.

According to [Pauluis2008](@citet), the moist static energy obeys

```math
\partial_t(ρᵣ s) + \boldsymbol{\nabla \cdot}\, (ρᵣ s \boldsymbol{u}) = - ρᵣ w b + S_s ,
```

with vertical velocity ``w``, buoyancy ``b`` as above, and ``S_s`` including microphysical and external energy sources/sinks.
The ``ρᵣ w b`` term is the buoyancy flux that links the energy and momentum budgets in the anelastic limit.

When hydrometeors sediment, the condensate part of ``s`` falls with them: Breeze bins each
sedimenting species' advective sedimentation mass flux — the flux its own advection scheme
applies at the combined resolved and terminal velocity, minus the flux at the resolved
velocity alone, so the energy moves exactly with the mass — by thermodynamic phase, and
transports ``∂s/∂q^x`` per unit falling mass of phase ``x``, evaluated at fixed temperature
with the dry mass fraction ``q^d = 1 - q^v - q^l - q^i`` taking up the departed mass:
``(c^l - c^{pd}) T - \mathscr{L}^l_r`` for liquid and ``(c^i - c^{pd}) T - \mathscr{L}^i_r``
for ice, the latent deficit plus the sensible heat of the condensate relative to the dry air
that replaces it, taken in the cell each flux drains. The cell the condensate leaves therefore
keeps its temperature, the cell it arrives in gains the sensible heat carried from another
level, and rain-out leaves the latent heat of formation behind, pre-cooling the layer that
later evaporates the falling rain. The liquid-ice potential temperature formulation carries the
same enthalpy but converts it locally through ``1 / (c^{pm} Π)``, losing ``∂θ^{li}/∂q^x`` per
unit departing mass (to leading order ``-\mathscr{L}^x_r / (c^{pm} Π)``); transporting that
Jacobian between pressure levels would instead conserve ``∫ρθ^{li}``, which precipitation does
not. (On the compressible core the total density falls with the sedimenting condensate, so the
local mixture rather than dry air takes up the departed mass and the content becomes the
enthalpy of the condensate relative to that of the mixture; see `sedimentation_composition_increment`.)

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

## Boussinesq dynamics

In domains much shallower than the atmospheric scale height, such as direct numerical simulations of cloud turbulence or laboratory convection, the reference state is effectively independent of height.
[`BoussinesqDynamics`](@ref Breeze.AnelasticEquations.BoussinesqDynamics) is `AnelasticDynamics` expanded about a [`BoussinesqReferenceState`](@ref Breeze.Thermodynamics.BoussinesqReferenceState) with uniform pressure ``pᵣ``, temperature ``Tᵣ``, and density ``ρᵣ = pᵣ / (R^m Tᵣ)``.
With uniform ``ρᵣ``, the anelastic constraint reduces to ``\boldsymbol{\nabla \cdot}\, \boldsymbol{u} = 0``, the buoyancy reduces to the Boussinesq buoyancy of virtual temperature anomalies, and the projection becomes a Poisson equation with constant coefficients,

```math
\nabla^2 \phi = \frac{1}{ρᵣ Δt} \, \boldsymbol{\nabla \cdot}\, \widetilde{(ρᵣ \boldsymbol{u})} ,
```

which is solved with FFTs when the vertical direction is periodic.

```jldoctest
using Breeze

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
model = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid, temperature=290))
model.dynamics

# output
BoussinesqDynamics(pᵣ=101325.0, Tᵣ=290.0, ρᵣ=1.2174)
└── pressure_anomaly: 16×16×16 Field{Center, Center, Center} on RectilinearGrid on CPU
```

Because the reference pressure is uniform, the Exner function is uniform, and temperature does not decrease adiabatically with height within the domain.
Across a 1 m domain, the neglected adiabatic temperature difference is about 0.01 K.

!!! warning "Mean buoyancy in triply-periodic domains"
    A periodic pressure field cannot balance a domain-mean buoyancy force, so a nonzero domain-mean buoyancy accelerates the domain-mean vertical momentum in a vertically-periodic domain.
    Setting `frame_vertical_velocity` (to zero, for a fixed frame) measures buoyancy relative to the domain mean, which removes the domain-mean buoyancy; see below.

### Moving reference frames

A small domain inside a cloud rises or sinks with the surrounding air.
Setting `frame_vertical_velocity` ``W`` moves the reference frame of a `BoussinesqReferenceState` vertically, and its reference pressure evolves hydrostatically,

```math
\frac{\mathrm{d} pᵣ}{\mathrm{d} t} = - \bar{ρ} \, g \, W , \qquad \bar{ρ} = \frac{pᵣ}{R^m \bar{T}} ,
```

where ``\bar{T}`` is the domain-mean temperature.
Because the liquid-ice potential temperature is conserved during adiabatic motion, the decreasing (or increasing) reference pressure cools (or warms) the domain at the moist adiabatic rate through the Exner function, and condensation releases latent heat as usual.
The reference temperature and moisture mass fractions follow the domain mean, so buoyancy is measured relative to the domain mean, and the reference density used by the dynamics is fixed.

```julia
dynamics = BoussinesqDynamics(grid; pressure=9e4, temperature=285, frame_vertical_velocity=2)
```

The uniform reference pressure means that air displaced by ``w′`` relative to the frame does not expand on its own.
A moving frame therefore carries [`AdiabaticExpansion`](@ref) by default, which adds the sources ``-Gᶿ w′`` and ``-Gᵛ w′``
to potential temperature and vapor, where ``Gᶿ`` and ``Gᵛ`` are the gradients of a saturated moist-adiabatic background
evaluated from the frame state, so that every parcel's supersaturation responds to its displacement at the adiabatic-parcel rate.
Pass `adiabatic_expansion = nothing` to disable it.

### Forcing homogeneous isotropic turbulence

Direct numerical simulations of homogeneous isotropic turbulence in a triply-periodic domain sustain
the turbulence by forcing the largest scales.
Breeze provides two forcings that act on the Fourier modes of the velocity whose wavevectors lie in a band
``k_\mathrm{min} ≤ |\boldsymbol{k}| ≤ k_f``, by default ``|\boldsymbol{k}| ≤ 1.5 × 2π / L``.
In a cubic domain of side ``L``, where ``\boldsymbol{k} = 2π \boldsymbol{n} / L``, the default band holds the
9 pairs of modes with ``|\boldsymbol{n}|² = 1`` or ``2``.

- [`ConstantPowerBandForcing`](@ref) amplifies the forced modes of the velocity without changing their phases,
  ```math
  \boldsymbol{F} = \frac{ε}{2 E_f} \, \boldsymbol{u}_f , \qquad E_f = \frac{1}{2} ⟨|\boldsymbol{u}_f|²⟩ ,
  ```
  so that the power input ``⟨\boldsymbol{u} ⋅ \boldsymbol{F}⟩`` is exactly the energy injection rate ``ε``.
  This is the continuous-time form of the forcing of the NCAR parcel-DNS code of [Chen et al. (2016)](@cite Chen2016),
  which, after every leapfrog step, rescales the forced modes so that their energy ``E_f`` grows by ``2 ε Δt``
  (a leapfrog step advances each of its two interleaved solutions by ``2 Δt``).
  Breeze evaluates ``\boldsymbol{u}_f`` and ``E_f`` at every Runge-Kutta stage, so the power input is ``ε`` at every
  stage and the forcing work per time step is ``ε Δt``. The forcing requires ``E_f ≫ ε Δt``, which holds in developed
  turbulence: start from a velocity field with energy in the forced band, not from rest or from grid-scale noise alone.
- [`StochasticBandForcing`](@ref) is the forcing of [Eswaran and Pope (1988)](@cite EswaranPope1988):
  the amplitudes of the forced modes are Ornstein-Uhlenbeck processes with correlation time ``T_f``, independent
  of the velocity, whose variance gives ``⟨|\boldsymbol{F}|²⟩ T_f = ε``. The mean power input is ``ε`` when ``T_f`` is
  short compared with the turnover time of the forced eddies, and somewhat less otherwise.
  The amplitudes are drawn once per time step, on the host, from a seeded counter-based random number generator.

Both forcings are projected onto the plane perpendicular to the modified wavenumber of each mode, so that they are
divergence-free on the staggered grid and do not change the pressure.
The Fourier coefficients of the velocity in the band are computed by partial discrete Fourier transforms
(summing over ``z``, then ``y``, then ``x``, for the few wavenumbers of the band), and the forcing is evaluated in
physical space as a sum over the forced modes, so no three-dimensional FFTs are needed.
The forcings return velocity tendencies and are supplied under the velocity names, so Breeze multiplies them by ``ρᵣ``:

```jldoctest
using Breeze

grid = RectilinearGrid(size=(16, 16, 16), extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))
forcing = ConstantPowerBandForcing(grid; energy_injection_rate=0.1)

model = AtmosphereModel(grid; dynamics = BoussinesqDynamics(grid),
                        forcing = (u=forcing, v=forcing, w=forcing))

model.forcing.ρw

# output
SpecificForcing with forcing: BandForcingComponent(w) of ConstantPowerBandForcing(ε=0.1)
```

## Symbols

- ``ρᵣ(z)``, ``pᵣ(z)``: Reference density and pressure satisfying hydrostatic balance for a constant ``θᵣ``.
- ``α = R^m T / pᵣ``, ``αᵣ = R^d θᵣ / pᵣ``: Specific volume and its reference value.
- ``b = g (α - αᵣ) / αᵣ``: Buoyancy.
- ``s``: Moist static energy.
- ``q^t``: Total specific humidity (vapor + condensates).
- ``\phi``: Nonhydrostatic pressure correction potential used by the projection.
