# [Anelastic dynamics](@id Anelastic-section)

[`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) filters acoustic waves by linearizing about a hydrostatic reference state, following the thermodynamically consistent framework of [Pauluis2008](@citet). This formulation is suitable for most large-eddy simulations and mesoscale applications.

## Anelastic approximation

To filter acoustic waves while retaining compressibility effects in buoyancy and thermodynamics, we linearize about a hydrostatic, horizontally uniform reference state ``(pᵣ(z), ρᵣ(z))`` with constant reference potential temperature ``θ₀``. The key assumptions are

- Small Mach number and small relative density perturbations except in buoyancy.
- Hydrostatic reference balance: ``\partial_z pᵣ = -ρᵣ g``.
- Mass flux divergence constraint: ``\boldsymbol{\nabla \cdot}\, (ρᵣ\,\boldsymbol{u}) = 0``.

Define the specific volume of moist air and its reference value as

```math
α = \frac{R^m T}{pᵣ} , \qquad αᵣ = \frac{R^d θ₀}{pᵣ} ,
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
ρᵣ s ≡ ρᵣ \left ( c^{pm} T + g z - \mathscr{L}^l_0 q^l - \mathscr{L}^i_0 q^i \right ),
```

where ``c^{p m}`` is the mixture heat capacity, ``T`` is temperature, ``g`` is gravitational acceleration,
``z`` is height,
``\mathscr{L}^l_0 = \mathscr{L}^l_r - (c^{pv} - c^l) T_r`` is the latent heat of condensation (vapor to liquid)
extrapolated to absolute zero,
and
``\mathscr{L}^i_0 = \mathscr{L}^i_r - (c^{pv} - c^i) T_r`` is the latent heat of deposition (vapor to ice)
extrapolated to absolute zero, with ``\mathscr{L}^x_r`` the latent heat at the energy reference temperature ``T_r``.
Because ``c^{pm} T`` measures sensible enthalpy from absolute zero, the latent terms are referenced there too,
so a phase change at fixed ``s`` releases the temperature-dependent latent heat ``\mathscr{L}^x(T)``.

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
``(c^l - c^{pd}) T - \mathscr{L}^l_0`` for liquid and ``(c^i - c^{pd}) T - \mathscr{L}^i_0``
for ice, the latent deficit plus the sensible heat of the condensate relative to the dry air
that replaces it, taken in the cell each flux drains. The cell the condensate leaves therefore
keeps its temperature, the cell it arrives in gains the sensible heat carried from another
level, and rain-out leaves the latent heat of formation behind, pre-cooling the layer that
later evaporates the falling rain. The liquid-ice potential temperature formulation carries the
same enthalpy but converts it locally through ``1 / (\hat{c}ᵖ \hat{Π})``, losing ``∂θˡⁱ/∂q^x`` per
unit departing mass (to leading order ``-\mathscr{L}^x(T) / (\hat{c}ᵖ \hat{Π})``; the exact content
is given in the [Thermodynamics](@ref liquid-ice-potential-temperature) section); transporting that
Jacobian between pressure levels would instead conserve ``∫ρθˡⁱ``, which precipitation does
not. (On the compressible core the total density falls with the sedimenting condensate, so the
local mixture rather than dry air takes up the departed mass and the content becomes the
enthalpy of the condensate relative to that of the mixture; see `sedimentation_composition_increment`.)

Thermodynamic closures for ``R^m``, ``c^{pm}`` and the vapor-equivalent Exner function ``\hat{Π} = (pᵣ / pˢᵗ)^{\hat{κ}}`` are given in the [Thermodynamics](@ref Thermodynamics-section) section.

## Liquid-ice potential temperature

The default formulation advances the liquid-ice potential temperature density ``ρᵣ θˡⁱ``, with
``θˡⁱ = H / (\hat{c}ᵖ \hat{Π})`` defined exactly from the enthalpy ``H = s - g z`` and the
vapor-equivalent heat capacity ``\hat{c}ᵖ`` and Exner function ``\hat{Π}``, as derived in
[Liquid-ice potential temperature](@ref liquid-ice-potential-temperature). Because the
thermodynamic pressure is the reference pressure ``pᵣ(z)``, the material derivative of the
pressure that drives expansion is exact and local,
``\mathrm{D} \ln pᵣ / \mathrm{D} t = w \, \partial_z \ln pᵣ = - g ρᵣ w / pᵣ``, and the prognostic
equation is

```math
\partial_t(ρᵣ θˡⁱ) + \boldsymbol{\nabla \cdot}\, (ρᵣ θˡⁱ \boldsymbol{u}) =
- ρᵣ \, \frac{R^m T - \hat{R} \hat{Π} θˡⁱ}{\hat{c}ᵖ \hat{Π}} \, \frac{g ρᵣ w}{pᵣ} + S_θ ,
```

where ``R^m T - \hat{R} \hat{Π} θˡⁱ = \hat{κ} (\mathscr{L}^l qˡ + \mathscr{L}^i qⁱ) - R^v (qˡ + qⁱ) T``
vanishes in clear air, so that ``θˡⁱ`` is materially conserved there, and in cloud supplies the
expansion work that the condensate's share of the enthalpy would otherwise miss — between one and
two percent of the adiabatic cooling for 2 g kg⁻¹ of condensate. Phase change contributes no
source at all. ``S_θ`` holds the heating ``(Q_{\mathscr{I}} + F_{ρE}) / (\hat{c}ᵖ \hat{Π})``, where
``Q_{\mathscr{I}}`` is the radiative heating and ``F_{ρE}`` an interior energy forcing, the subgrid
flux divergence, and the sedimentation content tendency described above. With the dry-exponent
option, ``\hat{Π}`` is replaced by ``Πᵈ(z) = (pᵣ / pˢᵗ)^{R^d / c^{pd}}``, computed once per level,
and the source acquires a small clear-air part; see
[Choosing the Exner exponent](@ref "Choosing the Exner exponent").

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
├── pressure_solver_float_type: Float64
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

```jldoctest
using Breeze

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
dynamics = BoussinesqDynamics(grid; pressure=9e4, temperature=285, frame_vertical_velocity=2)

# output
BoussinesqDynamics(pᵣ=90000.0, Tᵣ=285.0, ρᵣ=1.1003, frame_vertical_velocity=2.0, AdiabaticExpansion)
├── pressure_solver_float_type: Float64
└── pressure_anomaly: not materialized
```

The uniform reference pressure means that air displaced by ``w′`` relative to the frame does not expand on its own.
A moving frame therefore carries [`AdiabaticExpansion`](@ref) by default, which adds the sources ``-Gᶿ w′`` and ``-Gᵛ w′``
to potential temperature and vapor, where ``Gᶿ`` and ``Gᵛ`` are the gradients of a saturated moist-adiabatic background
evaluated from the frame state, so that every parcel's supersaturation responds to its displacement at the adiabatic-parcel rate.
Pass `adiabatic_expansion = nothing` to disable it.

## Symbols

- ``ρᵣ(z)``, ``pᵣ(z)``: Reference density and pressure satisfying hydrostatic balance for a constant ``θ₀``.
- ``α = R^m T / pᵣ``, ``αᵣ = R^d θ₀ / pᵣ``: Specific volume and its reference value.
- ``b = g (α - αᵣ) / αᵣ``: Buoyancy.
- ``s``: Moist static energy.
- ``q^t``: Total specific humidity (vapor + condensates).
- ``\phi``: Nonhydrostatic pressure correction potential used by the projection.
