# [Choosing a thermodynamic variable](@id thermodynamic-formulations)

Breeze advances one thermodynamic variable per model, and offers a choice. The default is the
liquid-ice potential temperature ``θˡⁱ``; the static energy ``s`` is the alternative on the anelastic
core; the moist entropy ``η`` and the total energy ``E`` are being added for the compressible core.
All four are functions of the same state. They differ in which processes leave them unchanged —
adiabatic expansion, phase change, mixing — and therefore in which source terms a model must compute
and which errors it makes when it drops one. This page derives the equations of moist air once,
from the conservation laws, and then treats each variable with the same questions: how it is
defined, how temperature is recovered from it, which sources it needs, what it conserves exactly,
and what it costs.

The derivation of the prognostic equation for a general state function ``χ(T, p, q)`` and the
exact definition of ``θˡⁱ`` are on the [Thermodynamics](@ref Thermodynamics-section) page and are
not repeated here; this page builds on them.

## Conservation laws for moist air

### Mass

Moist air is a mixture of dry air, vapor, liquid and ice, ``β ∈ \{d, v, l, i\}``, with partial
densities ``ρ^β`` and total density ``ρ = \sum_β ρ^β``. Each constituent may move with its own
velocity ``\boldsymbol{u}^β``: condensate falls, and subgrid motions need not be the same for every
species. The barycentric velocity ``\boldsymbol{u} = \sum_β ρ^β \boldsymbol{u}^β / ρ`` is the
velocity of the momentum equation, and the flux of constituent ``β`` relative to it,
``\boldsymbol{J}^β = ρ^β (\boldsymbol{u}^β - \boldsymbol{u})``, sums to zero by construction,
``\sum_β \boldsymbol{J}^β = 0``. Each constituent then obeys

```math
∂_t ρ^β + \boldsymbol{\nabla \cdot} \big( ρ^β \boldsymbol{u} + \boldsymbol{J}^β \big) = ρ \dot{S}^β ,
\qquad \dot{S}^d = 0 , \qquad \sum_β \dot{S}^β = 0 ,
```

where ``\dot{S}^β`` is the net rate at which phase changes produce ``β``. Summing over ``β``
gives the continuity equation ``∂_t ρ + \boldsymbol{\nabla \cdot} (ρ \boldsymbol{u}) = 0``, and
dividing by it gives the mass-fraction equations
``ρ \, \mathrm{D} q^β / \mathrm{D} t = - \boldsymbol{\nabla \cdot} \boldsymbol{J}^β + ρ \dot{S}^β``
used in the [general prognostic equation](@ref "The prognostic equation of a thermodynamic state function").
Dry air and total water, ``qᵗ = qᵛ + qˡ + qⁱ``, have no source: they are the conserved substances
of the mixture, and the individual phases of water are not.

!!! note "Carrier densities in Breeze"
    Breeze's dynamical cores weight the prognostic momentum and thermodynamic variable by a
    *coupling density* — the reference density ``ρᵣ`` on the anelastic core, the dry-air density
    ``ρᵈ`` on the compressible core — rather than by ``ρ``; see the
    [notation](@ref "Notation and conventions"). The derivations below are written for the
    barycentric form, which is the physics; the weighting is bookkeeping that does not change
    which sources a variable needs.

### Momentum and kinetic energy

With ``p`` the pressure, ``g`` gravity and ``\boldsymbol{\mathcal{T}}`` the viscous and subgrid
stress, the momentum equation is

```math
ρ \frac{\mathrm{D} \boldsymbol{u}}{\mathrm{D} t} = - \boldsymbol{\nabla} p - ρ g \hat{\boldsymbol{z}}
+ \boldsymbol{\nabla \cdot} \boldsymbol{\mathcal{T}}
```

(the Coriolis force does no work and is omitted). Taking the dot product with ``\boldsymbol{u}``
gives the budget of the kinetic energy per unit mass, ``K = |\boldsymbol{u}|^2 / 2``,

```math
ρ \frac{\mathrm{D} K}{\mathrm{D} t} = - \boldsymbol{u \cdot \nabla} p - ρ g w
+ \boldsymbol{\nabla \cdot} (\boldsymbol{\mathcal{T} \cdot u}) - ρ ε ,
\qquad ρ ε ≡ \boldsymbol{\mathcal{T}} : \boldsymbol{\nabla u} ≥ 0 ,
```

where ``ε`` is the rate at which stress converts kinetic energy into heat. Two terms exchange
energy with other reservoirs: the pressure work ``- \boldsymbol{u \cdot \nabla} p`` with the
internal energy, and the gravity work ``- ρ g w`` with the potential energy.

### Total energy

The total energy per unit mass is ``E = 𝓊 + K + g z``, with ``𝓊`` the internal energy. The first
law states that ``E`` changes only by fluxes and by external heating ``Q`` per unit mass (radiation,
for instance):

```math
∂_t (ρ E) + \boldsymbol{\nabla \cdot} \Big[ (ρ E + p) \boldsymbol{u} + \boldsymbol{J}_q
+ \sum_β h^β \boldsymbol{J}^β - \boldsymbol{\mathcal{T} \cdot u} \Big] = ρ Q .
```

The flux contains the pressure work ``p \boldsymbol{u}``, the conductive heat flux
``\boldsymbol{J}_q`` (and its subgrid counterpart), the stress work, and the energy that each
constituent carries when it moves relative to the mixture. The last is the *enthalpy* ``h^β``,
not the internal energy, because mass that moves through the mixture also does pressure work on it.

Every other form of the energy equation follows by subtraction. Removing the kinetic and
potential energy budgets gives the internal energy,

```math
ρ \frac{\mathrm{D} 𝓊}{\mathrm{D} t} = - p \, \boldsymbol{\nabla \cdot u} + ρ (Q + ε)
- \boldsymbol{\nabla \cdot} \Big( \boldsymbol{J}_q + \sum_β h^β \boldsymbol{J}^β \Big) ,
```

in which the dissipation ``ε`` reappears as heating and the pressure work appears as compression;
and with ``h = 𝓊 + p / ρ`` and the continuity equation,
``- p \, \boldsymbol{\nabla \cdot u} = - ρ \, \mathrm{D}(p/ρ) / \mathrm{D} t + \mathrm{D} p / \mathrm{D} t``,
gives the enthalpy form that the [Thermodynamics](@ref "Moist enthalpy and Kirchhoff's law") page
starts from. These are one law written three ways. In the total-energy form, pressure work,
dissipation and gravity work are exchanges *between the parts of* ``E`` and leave it unchanged; in
the other forms the same exchanges are sources. Which form is source-free is exactly the question
of which variable to advance.

## Thermodynamic functions of the mixture

### Internal energy, enthalpy and the origin of energy

Dry air and vapor are ideal gases, liquid and ice are incompressible, and all four have constant
heat capacities. With liquid water at zero kelvin as the origin of energy, the constituent
enthalpies are those of the [Thermodynamics](@ref "Moist enthalpy and Kirchhoff's law") page,
``hᵈ = cᵖᵈ T``, ``hᵛ = cᵖᵛ T + ℒˡ₀``, ``hˡ = cˡ T``, ``hⁱ = cⁱ T + ℒˡ₀ - ℒⁱ₀``, and the internal
energies follow from ``𝓊^β = h^β - p^β / ρ^β`` for the gases and ``𝓊^β = h^β`` for the condensate,
whose volume is neglected:

```math
𝓊ᵈ = cᵛᵈ T , \qquad 𝓊ᵛ = cᵛᵛ T + ℒˡ₀ , \qquad 𝓊ˡ = cˡ T , \qquad 𝓊ⁱ = cⁱ T + ℒˡ₀ - ℒⁱ₀ ,
```

with ``cᵛ = cᵖ - R`` for each gas. Summing with the mass fractions,

```math
h = cᵖᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ , \qquad
𝓊 = h - \frac{p}{ρ} = cᵛᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ , \qquad
cᵛᵐ ≡ cᵖᵐ - Rᵐ ,
```

where ``cᵛᵐ = qᵈ cᵛᵈ + qᵛ cᵛᵛ + qˡ cˡ + qⁱ cⁱ`` is the isochoric heat capacity of the mixture.
Kirchhoff's law is built in: the enthalpy released by condensing a unit mass at temperature ``T``
is ``hᵛ - hˡ = ℒˡ₀ + (cᵖᵛ - cˡ) T = ℒˡ(T)``, the same temperature-dependent latent heat that
Breeze integrates in the [Clausius–Clapeyron relation](@ref "Temperature-dependent latent heat"),
and the internal energy released at fixed volume is ``𝓊ᵛ - 𝓊ˡ = ℒˡ(T) - Rᵛ T``.

The origin of energy is not entirely free. The first law fixes the energy of each *conserved
substance* only up to a constant, and the conserved substances are dry air and total water, not
the individual phases. The energy may therefore be shifted by

```math
E → E - aᵈ qᵈ - aʷ qᵗ ,
```

with one constant for dry air and one shared by vapor, liquid and ice; a different shift for two
phases of water would change a latent heat. Because ``qᵈ`` and ``qᵗ`` are conserved, the shift
is carried by the mass fluxes and changes nothing but the stored numbers. The enthalpy
``H = h - ℒˡ₀ qᵗ`` of the liquid-ice potential temperature and the static energy ``s = H + g z``
are such shifts (``aʷ = ℒˡ₀``), which is why they are legitimate energy variables. The total-energy
formulation uses the same freedom to reduce round-off: its `reference_temperature` ``T₀`` sets
``aʷ = cˡ T₀`` and ``aᵈ = cᵖᵈ T₀``, so that liquid water has zero internal energy and dry air zero
enthalpy at ``T₀``, which makes ``|E|`` several times smaller for ``T₀ ≈ 273`` K without touching a
latent heat.

### Entropy and the Gibbs relation

The specific entropy of the mixture is the mass-weighted sum of partial specific entropies,
``η = \sum_β q^β η^β``. With constant heat capacities, and the entropy of liquid water zero at
the triple point ``(Tᵗʳ, pᵗʳ)`` [Emanuel (1994)](@cite Emanuel1994), they are

```math
ηᵈ = cᵖᵈ \ln \frac{T}{Tᵗʳ} - Rᵈ \ln \frac{pᵈ}{pᵗʳ} , \qquad
ηᵛ = cᵖᵛ \ln \frac{T}{Tᵗʳ} - Rᵛ \ln \frac{pᵛ}{pᵗʳ} + \frac{ℒˡ(Tᵗʳ)}{Tᵗʳ} , \qquad
ηˡ = cˡ \ln \frac{T}{Tᵗʳ} , \qquad
ηⁱ = cⁱ \ln \frac{T}{Tᵗʳ} - \frac{ℒᶠʳ(Tᵗʳ)}{Tᵗʳ} ,
```

with the partial pressures ``pᵈ = ρ qᵈ Rᵈ T`` and ``pᵛ = ρ qᵛ Rᵛ T``. The gases carry an entropy
of mixing through their partial pressures: this is what makes the mixing of moist air irreversible
below. The constants in ``ηᵛ`` and ``ηⁱ`` are the latent heats at the triple point divided by its
temperature, and they are chosen so that the specific Gibbs functions ``\mathcal{G}^β = h^β - T η^β``
satisfy

```math
\mathcal{G}^v - \mathcal{G}^l = Rᵛ T \ln \frac{pᵛ}{pᵛ⁺ˡ(T)} , \qquad
\mathcal{G}^i - \mathcal{G}^l = Rᵛ T \ln \frac{pᵛ⁺ⁱ(T)}{pᵛ⁺ˡ(T)} ,
```

exactly, with ``pᵛ⁺`` the saturation vapor pressures that Breeze integrates from the
Clausius–Clapeyron relation with the same latent heats. Vapor and condensate have equal Gibbs
function at saturation: the saturation curve, Kirchhoff's law and the entropy are one consistent
set of constants.

Differentiating ``η`` and using the ideal gas law gives the Gibbs relation of the mixture,

```math
T \, \mathrm{d} η = \mathrm{d} h - α \, \mathrm{d} p - \sum_{β ∈ \{v, l, i\}} (\mathcal{G}^β - \mathcal{G}^d) \, \mathrm{d} q^β ,
```

with ``α = 1 / ρ`` and the sum over the water constituents, each exchanged for dry air. Substituting
the enthalpy and mass equations, and using ``\sum_β \boldsymbol{J}^β = 0`` and ``\sum_β \dot{S}^β = 0``
to extend the sums over all four constituents, gives the entropy equation

```math
ρ \frac{\mathrm{D} η}{\mathrm{D} t} = \frac{ρ Q}{T}
- \boldsymbol{\nabla \cdot} \Big( \frac{\boldsymbol{J}_q}{T} + \sum_β η^β \boldsymbol{J}^β \Big)
+ ρ σ ,
```

```math
ρ σ = \frac{ρ ε}{T}
+ \boldsymbol{J}_h \boldsymbol{\cdot \nabla} \frac{1}{T}
- \sum_β \boldsymbol{J}^β \boldsymbol{\cdot \nabla} \frac{\mathcal{G}^β}{T}
- \frac{ρ}{T} \sum_β \mathcal{G}^β \dot{S}^β
≥ 0 ,
\qquad \boldsymbol{J}_h ≡ \boldsymbol{J}_q + \sum_β h^β \boldsymbol{J}^β .
```

Heating enters as ``Q / T``, conduction and the relative fluxes carry entropy, and everything else
is the irreversible production ``σ``: each term is a flux times the gradient of its conjugate
force, and the second law requires their sum to be non-negative. The last term is the entropy of
phase change,

```math
- \frac{1}{T} \sum_β \mathcal{G}^β \dot{S}^β
= - Rᵛ \left[ \ln \frac{pᵛ}{pᵛ⁺ˡ} \, \dot{S}^v + \ln \frac{pᵛ⁺ⁱ}{pᵛ⁺ˡ} \, \dot{S}^i \right] ,
```

which vanishes for condensation at saturation and for freezing at the triple point, and is positive
when a supersaturation relaxes or when condensate evaporates into subsaturated air. A process with
``σ = 0`` — adiabatic expansion at fixed composition, phase change at saturation — is reversible,
and conserves ``η``. That is the precise sense in which entropy conservation is the reversible
limit, and it is what the moist-entropy formulation builds on.

## The prognostic equation for a state function, and where the sources go

Every variable except ``E`` is a function ``χ(T, p, q)`` of the local thermodynamic state, and its
prognostic equation is derived once on the Thermodynamics page,
[The prognostic equation of a thermodynamic state function](@ref "The prognostic equation of a thermodynamic state function"):

```math
ρ \frac{\mathrm{D} χ}{\mathrm{D} t} =
ρ \, \mathcal{E}_χ \frac{\mathrm{D} \ln p}{\mathrm{D} t}
+ ρ \sum_β \mathcal{P}^β_χ \dot{S}^β
+ χ_h \Big[ ρ (Q + ε) - \boldsymbol{\nabla \cdot} \boldsymbol{J}_q - \sum_β (c^β - cᵖᵈ) \, \boldsymbol{J}^β \boldsymbol{\cdot \nabla} T \Big]
- \sum_β χ_{q^β} \boldsymbol{\nabla \cdot} \boldsymbol{J}^β ,
```

with the expansion coefficient ``\mathcal{E}_χ = (∂χ / ∂ \ln p)_{η, q}``, the phase-change
coefficients ``\mathcal{P}^β_χ = (∂χ / ∂q^β)_{h, p}``, the heating coefficient
``χ_h = (∂χ / ∂h)_{p, q}`` and the composition derivatives ``χ_{q^β} = (∂χ / ∂q^β)_{T, p}``.
For the three state functions of this page they are:

| ``χ`` | ``χ_h`` | ``\mathcal{E}_χ`` | ``\mathcal{P}^β_χ`` |
|:---|:---|:---|:---|
| ``H = s - g z`` | ``1`` | ``Rᵐ T`` | ``- ℒˡ₀`` for every ``β`` |
| ``θˡⁱ = H / (\hat{c}ᵖ \hat{Π})`` | ``\dfrac{1}{\hat{c}ᵖ \hat{Π}}`` | ``\dfrac{\hat{κ} (ℒˡ qˡ + ℒⁱ qⁱ) - Rᵛ (qˡ + qⁱ) T}{\hat{c}ᵖ \hat{Π}}`` | equal for every ``β`` |
| ``η`` | ``\dfrac{1}{T}`` | ``0`` | ``- \dfrac{\mathcal{G}^β - \mathcal{G}^d}{T}``, so that ``\mathcal{P}^l_η - \mathcal{P}^v_η = Rᵛ \ln \dfrac{pᵛ}{pᵛ⁺ˡ}``, ``\mathcal{P}^i_η - \mathcal{P}^v_η = Rᵛ \ln \dfrac{pᵛ}{pᵛ⁺ⁱ}`` |

The ``H`` row follows from ``∂H / ∂T = cᵖᵐ``, ``∂H / ∂ \ln p = 0`` and
``(∂H / ∂q^β)_{T,p} - (h^β - hᵈ) = - ℒˡ₀`` for each water constituent; the ``θˡⁱ`` row is derived
on the [Thermodynamics page](@ref "The equation for the liquid-ice potential temperature");
and the ``η`` row is the Gibbs relation read off term by term: ``(∂η / ∂h)_{p,q} = 1/T``,
``(∂η / ∂ \ln p)_{T,q} = - Rᵐ`` so that ``\mathcal{E}_η = (Rᵐ T / cᵖᵐ)(cᵖᵐ / T) - Rᵐ = 0``, and
``(∂η / ∂q^β)_{h,p} = - (\mathcal{G}^β - \mathcal{G}^d) / T``.

The table is the whole story in miniature. ``θˡⁱ`` and ``H`` have equal phase-change
coefficients, so a phase change needs no source at all, in or out of equilibrium; but they change
under expansion, ``H`` by the full pressure work and ``θˡⁱ`` by the condensate's share of it.
``η`` is invariant under expansion, and under phase change *at saturation*, where its coefficients
coincide; away from saturation it needs the entropy produced by the relaxation. No ``χ`` can have
both ``\mathcal{E}_χ = 0`` and equal ``\mathcal{P}^β_χ``: the first requires ``χ`` to be a
function of the mixture potential temperature and ``q``, the second a function of ``H``, ``p``
and ``qᵗ``, and no function is both. Every formulation therefore carries a *source burden*, and
the choice of variable decides where it sits — on expansion, on phase change, or, for ``E``, inside
the flux.

Total energy is outside this class because it depends on velocity and height as well as on
``(T, p, q)``. That is precisely what lets it absorb every source: its equation is the first law
in flux form, with no term on the right but ``ρ Q``.

## The four variables

### Static energy ``s``: the energy variable of the anelastic core

The moist static energy ``s = H + g z = cᵖᵐ T + g z - ℒˡ₀ qˡ - ℒⁱ₀ qⁱ`` is the enthalpy shifted
by ``- ℒˡ₀ qᵗ`` plus the geopotential; see [Moist static energy](@ref moist-static-energy).
Temperature is recovered in closed form with no transcendental function, since ``s`` is linear
in ``T``:

```math
T = \frac{s - g z + ℒˡ₀ qˡ + ℒⁱ₀ qⁱ}{cᵖᵐ} .
```

Its equation follows from the ``H`` row of the table and ``\mathrm{D}(g z) / \mathrm{D} t = g w``:

```math
ρ \frac{\mathrm{D} s}{\mathrm{D} t} =
\underbrace{\frac{\mathrm{D} p}{\mathrm{D} t} + ρ g w}_{\text{pressure and gravity work}}
+ ρ (Q + ε)
- \boldsymbol{\nabla \cdot} \Big( \boldsymbol{J}_q + \sum_{β ∈ \{v, l, i\}} \big( h^β - hᵈ - ℒˡ₀ \big) \boldsymbol{J}^β \Big) ,
```

with no phase-change term. Each relative water flux carries the enthalpy of what moves relative
to what replaces it: ``(cˡ - cᵖᵈ) T - ℒˡ₀`` per unit falling liquid, which is the sedimentation
content of the [anelastic page](@ref Anelastic-section), in flux form, so that precipitation
conserves ``∫ ρ s`` up to what leaves through the surface.

On the anelastic core the thermodynamic pressure is the hydrostatic reference pressure
``pᵣ(z)``, so ``\mathrm{D} pᵣ / \mathrm{D} t = - ρᵣ g w`` exactly, and the pressure and gravity
work per unit mass combine into one local term:

```math
α \frac{\mathrm{D} pᵣ}{\mathrm{D} t} + g w = - g w \, \frac{α - αᵣ}{αᵣ} = - w b ,
\qquad α = \frac{Rᵐ T}{pᵣ} , \quad αᵣ = \frac{1}{ρᵣ} ,
```

so that the anelastic equation reads ``ρᵣ \, \mathrm{D} s / \mathrm{D} t = - ρᵣ w b + \dots``,
with the buoyancy flux of [Pauluis (2008)](@cite Pauluis2008) as its only mechanical source. The
same ``b`` drives the vertical momentum equation, so the buoyancy work cancels between the kinetic
energy and ``s``, and ``ρᵣ (s + K)`` is conserved up to boundary fluxes and dissipation — on the
anelastic core, ``s`` plays the role that ``E`` plays on the compressible core. Because ``s`` is linear in
the enthalpy, any conservative transport conserves it: subgrid mixing, numerical mixing by the
advection scheme, and sedimentation all conserve ``∫ ρᵣ s`` exactly, which is the property the
other state functions lack (see [Mixing](@ref thermodynamic-formulations-mixing)). Its costs are
few. It stores a number of order ``3 × 10^5`` J kg⁻¹, of which the geopotential is a large inactive
part aloft; and the dissipative heating ``ε`` and the drag heating of falling precipitation,
which it could take as sources, are presently omitted. On the compressible core the pressure
work ``\mathrm{D} p / \mathrm{D} t`` is no longer local and needs the same pressure tendency as
the [expansion source of ``θˡⁱ``](@ref "The expansion source on the two dynamical cores"); there,
``E`` is the variable that avoids it.

### Total energy ``E``: the energy variable of the compressible core

The total-energy formulation of [Satoh (2003)](@cite Satoh2003), used in NICAM, advances the total
energy density ``ρ E`` with

```math
E = 𝓊 + K + g z , \qquad 𝓊 = cᵛᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ ,
```

in the conservation form derived [above](@ref "Total energy"). Its temperature is a closed-form
subtraction,

```math
T = \frac{E - K - g z - ℒˡ₀ qᵛ - (ℒˡ₀ - ℒⁱ₀) qⁱ}{cᵛᵐ} ,
```

and the pressure follows from ``p = ρ Rᵐ T``. With saturation-adjustment microphysics the
adjustment is made at fixed ``𝓊`` and ``ρ``, the isochoric analogue of the adjustment at fixed
``θˡⁱ`` and ``p``.

Every process that is a source for a state function is an exchange inside ``E`` or a flux
through its boundary:

- **Expansion**: the pressure work is the ``p \boldsymbol{u}`` inside the flux; nothing is
  evaluated. The total enthalpy per unit mass, ``E + p / ρ = h + K + g z``, is what the advection
  scheme transports.
- **Phase change**: ``𝓊`` already contains the latent energy, so condensation at fixed ``𝓊``
  and ``ρ`` warms the air by ``(ℒˡ(T) - Rᵛ T) / cᵛᵐ`` per unit mass condensed, through the change
  of ``cᵛᵐ T`` alone. Microphysics schemes act on the temperature at fixed ``𝓊``, as they act at
  fixed ``θˡⁱ``.
- **Dissipation and buoyancy work** move energy between ``K``, ``g z`` and ``𝓊`` and leave ``E``
  unchanged; the kinetic energy that the momentum scheme loses becomes heat.
- **Precipitation**: falling condensate carries its enthalpy ``hˣ(T)`` and its potential energy,
  as a flux. The air it falls through gains the potential energy the condensate loses — the drag
  heating that the other formulations omit — and ``∫ ρ E`` changes only by what reaches the
  surface, ``cˡ T₁`` per unit precipitated liquid.
- **Surface fluxes**: a sensible heat flux adds itself, and evaporated vapor brings its enthalpy
  ``hᵛ = cᵖᵛ T₁ + ℒˡ₀`` at the temperature ``T₁`` of the lowest cell.
- **Mixing**, resolved, subgrid, or numerical, conserves ``∫ ρ E`` because ``E`` is the conserved
  quantity. The subgrid flux is written for the moist static energy ``h + g z`` so that mixing at
  fixed pressure also conserves enthalpy, as physical mixing does.

Energy conservation is structural: the budget closes to round-off without any source being right,
because there are none. The costs are numerical.

- **Large-number cancellation.** ``E`` reaches ``3 × 10^5`` J kg⁻¹ at 20 km, where ``g z ≈ 2 × 10^5``
  exceeds ``cᵛᵐ T ≈ 1.5 × 10^5``, and ``ℒˡ₀ qᵛ ≈ 4 × 10^4`` near the surface; the inversion
  subtracts them. In `Float64` this costs a digit of temperature aloft.
- **`Float32`.** At LES time steps the increment of any thermodynamic variable in one step is
  below one unit in the last place of its stored value: a heating of 1 K day⁻¹ changes
  ``θ ≈ 300`` K by ``10^{-5}`` K per second against an ulp of ``3 × 10^{-5}`` K, and ``E`` by
  ``0.008`` J kg⁻¹ against an ulp of ``0.016`` near the surface and ``0.03`` at 20 km. All four
  variables therefore need the increment accumulated separately from the state
  ([issue #1122](https://github.com/NumericalEarth/Breeze.jl/issues/1122)); ``E``'s margin is
  the worst aloft, where the geopotential is the largest part of it. The `reference_temperature`
  option reduces ``|E|``, and storing ``ρ E`` as a perturbation from a time-independent
  hydrostatic reference removes the geopotential from the stored number while keeping the flux
  form, and so the conservation, exact.
- **Kinetic energy at the grid scale.** ``K`` is formed at cell centers from the staggered
  velocities, and the discrete kinetic-energy budget of the momentum scheme is now part of the
  thermodynamics. That is right when the lost kinetic energy is dissipation, and it is a heating
  error when it is a conservation error of the momentum scheme.
- **The acoustic loop** needs the pressure as a function of ``ρ E``; see
  [below](@ref thermodynamic-formulations-acoustic).

### Liquid-ice potential temperature ``θˡⁱ``

The default variable is defined exactly from the enthalpy,

```math
θˡⁱ = \frac{H}{\hat{c}ᵖ \hat{Π}} = \frac{1}{\hat{Π}} \left[ T - \frac{ℒˡ(T) \, qˡ + ℒⁱ(T) \, qⁱ}{\hat{c}ᵖ} \right] ,
```

with the vapor-equivalent heat capacity ``\hat{c}ᵖ``, gas constant ``\hat{R}`` and Exner function
``\hat{Π} = (p / pˢᵗ)^{\hat{R} / \hat{c}ᵖ}`` — the values the air would have if all of its water were
vapor, which no phase change alters — as derived in
[Liquid-ice potential temperature](@ref liquid-ice-potential-temperature). Temperature is
recovered in closed form, ``T = (\hat{c}ᵖ \hat{Π} θˡⁱ + ℒˡ₀ qˡ + ℒⁱ₀ qⁱ) / cᵖᵐ``, at the cost of
one power for ``\hat{Π}``; on the compressible core, where ``p = ρ Rᵐ T`` makes ``\hat{Π}`` depend
on ``T``, the same relation is solved iteratively.

Its sources are, from the table, one expansion source and the heating, converted by
``1 / (\hat{c}ᵖ \hat{Π})``:

- **Phase change**: none, for every transition, at every temperature, in or out of equilibrium.
  Microphysics couples only through the temperature.
- **Expansion**: ``ρ \mathcal{E}_θ \, \mathrm{D} \ln p / \mathrm{D} t``, proportional to the
  condensate and zero in clear air, where ``θˡⁱ`` is the exact potential temperature of the
  moist mixture. On the anelastic core ``\mathrm{D} \ln pᵣ / \mathrm{D} t = - g ρᵣ w / pᵣ`` is
  local and exact; on the compressible core the pressure tendency is
  [approximated hydrostatically](@ref "The expansion source on the two dynamical cores").
- **Sedimentation**: a cell-local change of ``θˡⁱ`` per unit departing mass, consistent in energy,
  rather than a flux; ``∫ ρ θˡⁱ`` is not conserved by precipitation, nor should it be.
- **Omitted**: the dissipative heating ``ε`` and the drag heating of falling precipitation,
  together one to a few W m⁻² in radiative-convective equilibrium. Both are local and could be
  added as optional sources; see
  [issue #1152](https://github.com/NumericalEarth/Breeze.jl/issues/1152).

What it does not conserve is energy under mixing. Since ``h = \hat{c}ᵖ \hat{Π} θˡⁱ + ℒˡ₀ qᵗ``
is bilinear in ``(θˡⁱ, qᵗ)``, transporting ``θˡⁱ`` conservatively errs by a term proportional to
the covariance of ``θˡⁱ`` and ``qᵗ`` — a few J kg⁻¹ per mixing event between cloudy and clear
air, zero between parcels of equal composition — and mixing between pressure levels, numerical
or subgrid, creates potential energy without spending kinetic energy, about 3 W m⁻² in the
radiative-convective equilibrium simulation of the
[Thermodynamics page](@ref theta-li-unconserved). Both are small, and neither can be removed
from the implicit mixing of the advection scheme.

In clear air ``θˡⁱ`` and ``η`` carry the same information:

```math
η = \hat{c}ᵖ \ln θˡⁱ + F(qᵗ) \qquad (qˡ = qⁱ = 0) ,
```

since ``η = \hat{c}ᵖ \ln T - \hat{R} \ln p`` plus functions of the composition alone. Conserving
one is conserving the other; they differ in cloud, where ``η`` is also invariant under expansion
but only under saturated phase change, and they differ in how they mix, through the convexity of
the exponential.

### Moist entropy ``η``

The moist-entropy formulation advances ``ρ η``, with ``η`` the entropy of the mixture defined
[above](@ref "Entropy and the Gibbs relation"), condensate included. Written at fixed density,
``η`` separates into a temperature part and a part that depends on the density and composition
only,

```math
η = cᵛᵐ \ln \frac{T}{Tᵗʳ} - f(ρ, q) , \qquad
f = qᵈ Rᵈ \ln \frac{ρ qᵈ}{ρᵈᵗʳ} + qᵛ Rᵛ \ln \frac{ρ qᵛ}{ρᵛᵗʳ} - qᵛ \frac{ℒˡ(Tᵗʳ)}{Tᵗʳ} + qⁱ \frac{ℒᶠʳ(Tᵗʳ)}{Tᵗʳ} ,
```

with ``ρᵈᵗʳ`` and ``ρᵛᵗʳ`` the dry-air and vapor densities at the triple point, so that on the
compressible core, where ``ρ`` is known, temperature is one exponential,
``T = Tᵗʳ \exp[(η + f) / cᵛᵐ]``, with or without condensate, as long as the water mass fractions
are prognostic. With saturation adjustment the vapor depends on ``T`` and the inversion must
iterate, as the saturated ``θˡⁱ`` inversion does; the formulation does not support it, for the
reason given below. Like the energy, ``η`` is defined up to one constant per conserved substance;
the implementation shifts the entropy of all water by a constant per unit ``qᵗ`` to tame the
acoustic loop's response to vapor gradients, which changes no physics.

The attraction is the ``η`` row of the table: ``\mathcal{E}_η = 0``, so expansion needs no source
on either core, which on the compressible core removes the pressure tendency entirely; and the
phase-change coefficients coincide at saturation, so reversible condensation is free. Heating
enters as ``Q / T``. The price is that energy is no longer the prognosed quantity, so every
irreversible process must add its entropy production — each term of ``σ`` — explicitly, and an
omission is silent, appearing only as an energy error:

- **Non-equilibrium phase change**: ``- \sum_β \mathcal{G}^β \dot{S}^β / T``, from the
  microphysical tendencies of every scheme, including freezing and melting away from the triple
  point.
- **Dissipation**: ``ε / T``, from the closure.
- **Precipitation**: falling condensate carries its own entropy ``η^β(T)`` as a flux, and its
  drag heats the air at ``g |wˣ| ρ qˣ / T``.
- **Surface fluxes**: ``\mathcal{Q}^T / T + Jᵛ ηᵛ``, the sensible heat at the temperature of the
  lowest cell plus the entropy of the evaporated vapor at that cell's state.
- **Saturation adjustment** at fixed ``η``, as in PyCLES [Pressel et al. (2015)](@cite Pressel2015),
  drops the entropy of relaxing each step's supersaturation and cools: the adjustment must be made
  at fixed ``𝓊`` and ``ρ`` (or ``h`` and ``p``) and ``η`` written back.
- **Mixing**: physical mixing produces entropy, the ``\boldsymbol{J}_h`` and ``\boldsymbol{J}^β``
  terms of ``σ``. An explicit closure can supply them; implicit numerical mixing cannot be
  corrected, and conservative transport of ``η`` then destroys energy at first order, as the
  [next section](@ref thermodynamic-formulations-mixing) shows.

The entropy potential temperatures of [Emanuel (1994)](@cite Emanuel1994) and
[Marquet (2011)](@cite Marquet2011) are functions of ``η`` and ``qᵗ`` alone; they inherit every
property above and change only the range of values.

### [On the compressible core: the acoustic loop](@id thermodynamic-formulations-acoustic)

The split-explicit scheme of [Klemp et al. (2007)](@cite KlempSkamarockDudhia2007) needs the
pressure as a function of the prognostic variables, linearized about the stage-entry state
``L``; see [Compressible dynamics](@ref Compressible-section). For a prognostic ``ρ χ`` with the
water densities and the momentum frozen,

```math
p' = \Big( \frac{∂p}{∂ρχ} \Big)^{\!L} (ρχ)' + \Big( \frac{∂p}{∂ρ} \Big)^{\!L} ρ' ,
```

and the loop advances ``(ρχ)'`` with the flux ``\hat{χ}^L (ρ \boldsymbol{u})'``, where ``\hat{χ}``
is the value per unit carrier mass that the flux of ``ρχ`` transports. For dry air the three
compressible variables give

| ``ρχ`` | ``∂p / ∂ρχ`` | ``\hat{χ}`` | ``∂p / ∂ρ`` |
|:---|:---|:---|:---|
| ``ρ θ`` | ``γ R Π`` | ``θ`` | ``0`` |
| ``ρ E`` | ``R / cᵛ`` | ``E + p / ρ = h + K + g z`` | ``(R / cᵛ)(K - g z)`` |
| ``ρ η`` | ``R T / cᵛ`` | ``η`` | ``(R T / cᵛ)(cᵖ - η)`` |

with ``γ = cᵖ / cᵛ``. A uniform parcel compressed acoustically has ``(ρχ)' = \hat{χ} ρ'``, so the
sound speed each linearization produces is ``(∂p/∂ρχ) \hat{χ} + ∂p/∂ρ``: ``γ R Π θ``,
``(R / cᵛ)(cᵛ T + R T + 2K)`` and ``(R T / cᵛ) cᵖ``, all equal to ``γ R T = c_s^2`` up to a kinetic
term of order the Mach number squared. The three linearizations are the same acoustic and
gravity-wave operator in different coordinates; only the coefficients differ, and with them the
round-off. For ``ρ E`` the two terms of ``p'`` cancel to the extent that ``g z`` and the latent
offset exceed ``cᵛ T``; for ``ρ η`` they cancel to about one bit, since ``η`` and ``cᵖ - η`` are of
the same order with the triple-point origin. The ``ρ θ`` linearization has no density term and
no cancellation.

## [Mixing](@id thermodynamic-formulations-mixing)

Mixing is where the variables differ most sharply, and the reason fits in one formula. Consider
two equal masses at the same pressure that mix completely. Physical mixing conserves the enthalpy
and the total water, ``\bar{h} = \overline{h}`` and ``\bar{q}ᵗ = \overline{qᵗ}``, and the mixture's temperature
and phase partition follow from ``(\bar{h}, p, \bar{q}ᵗ)``. A model that transports ``χ`` and ``qᵗ``
conservatively lands instead at ``\bar{χ} = \overline{χ}``, and its enthalpy is ``h(\bar{χ}, \bar{q}ᵗ, p)``.
Expanding ``h`` about the mean,

```math
h(\bar{χ}, \bar{q}ᵗ) - \overline{h} = - \frac{1}{2} \Big[ h_{χχ} \, \mathrm{var}(χ) + 2 h_{χ q} \, \mathrm{cov}(χ, qᵗ) + h_{qq} \, \mathrm{var}(qᵗ) \Big] + \dots ,
```

with the second derivatives of ``h(χ, qᵗ, p)`` at fixed pressure. The energy error of
conservative mixing is set by the *curvature of the enthalpy as a function of the transported
variables*, and it vanishes for a variable in which ``h`` is linear.

- **Static energy and total energy.** ``h = s - g z`` is linear in ``s``, and ``E`` is the energy:
  the error is zero.

- **Liquid-ice potential temperature.** ``h = \hat{c}ᵖ \hat{Π} θˡⁱ + ℒˡ₀ qᵗ`` has
  ``h_{θθ} = 0``, ``h_{qq} ≈ 0`` and ``h_{θ q} = ∂(\hat{c}ᵖ \hat{Π}) / ∂qᵗ``, so the error is
  ``- h_{θ q} \, \mathrm{cov}(θˡⁱ, qᵗ) = - \tfrac{1}{4} Δ(\hat{c}ᵖ \hat{Π}) \, Δθˡⁱ`` for two parcels.
  Cloudy air with 17 g kg⁻¹ of water, 2 g kg⁻¹ of it liquid, mixing with clear air at 12 g kg⁻¹
  at the same temperature and 1000 hPa has ``Δ\hat{c}ᵖ = 4.3`` J kg⁻¹ K⁻¹ and ``Δθˡⁱ = -4.9`` K:
  an error of ``+5`` J kg⁻¹, of either sign in general, and zero between parcels of equal
  composition.

- **Moist entropy.** Here the error has a thermodynamic meaning. At fixed ``p`` and ``qᵗ``,
  ``(∂h / ∂η) = T``, so landing at ``η̄`` instead of at the entropy ``η̄ + σ_{mix}`` that physical
  mixing produces costs ``δh = - T σ_{mix}``: conserving ``η`` is *entropy-conserving mixing*, and
  the energy it destroys is the temperature times the entropy that real mixing would have created.
  It is first order in the irreversibility, and it is always a loss.

  *Dry air.* Two parcels at ``\bar{T} ± ΔT / 2``: ``h = cᵖ T`` and ``η = cᵖ \ln T`` give
  ``h_{ηη} = T / cᵖ`` and ``\mathrm{var}(η) = (cᵖ ΔT / 2\bar{T})^2``, so

  ```math
  δh = - \frac{cᵖ \, ΔT^2}{8 \bar{T}} ,
  ```

  which is ``- 10`` J kg⁻¹, or ``- 0.01`` K, for ``ΔT = 5`` K at 300 K. Equivalently, the
  entropy-conserving mixture sits at the geometric mean temperature ``\sqrt{T_1 T_2} ≤ \bar{T}``.

  *Moist air.* Two clear parcels at the same temperature with vapor ``q_1`` and ``q_2``: the
  physical mixture keeps its temperature, since ``h = cᵖᵐ T`` is linear in ``qᵛ``, and produces the
  entropy of mixing the vapor, ``σ_{mix} = Rᵛ \big[ \overline{qᵛ \ln qᵛ} - \bar{q}ᵛ \ln \bar{q}ᵛ \big] ≈ Rᵛ Δq^2 / (8 \bar{q}ᵛ)``
  to leading order, from the ``- Rᵛ qᵛ \ln pᵛ`` term of ``η``. Entropy-conserving mixing of 10
  and 20 g kg⁻¹ at 300 K therefore cools the mixture by ``T σ_{mix} ≈ 120`` J kg⁻¹, about 0.1 K —
  an order of magnitude more than the dry case, and larger than the ``θˡⁱ`` error, which is
  second order in the differences and does not involve ``Rᵛ T``.

In radiative-convective equilibrium the irreversible entropy production that balances the
radiative sink is about ``0.06`` W m⁻² K⁻¹ — ``15`` W m⁻² when multiplied by a typical temperature —
and more than half of it is the diffusion of vapor and irreversible phase change
[Pauluis and Held (2002)](@cite PauluisHeld2002a). A model that mixes ``η`` conservatively
forgoes the part of that production that occurs by mixing, and the energy it loses is of that
order: the same size as the expansion error of the old ``θˡⁱ`` definition that the exact
definition removed.

Mixing *between* pressure levels adds a second error for every state function, discussed on the
[Thermodynamics page](@ref theta-li-unconserved): a down-gradient exchange between levels is
physically an isentropic displacement followed by isobaric mixing, which costs kinetic energy in
a stable layer, and transporting ``θˡⁱ`` or ``η`` down its gradient reaches approximately the
right thermodynamic state while spending none. The energy variables are immune: ``∫ ρ s`` and
``∫ ρ E`` are conserved by any conservative transport, between levels or not.

## Comparison

| | ``θˡⁱ`` | ``η`` | ``s`` | ``E`` |
|:---|:---|:---|:---|:---|
| Dynamical core | anelastic, compressible | compressible | anelastic (compressible with approximate pressure work) | compressible |
| Exact under expansion | no: one local source ``∝`` condensate, exact on the anelastic core, hydrostatic estimate on the compressible core | yes | no: buoyancy flux ``-ρᵣ w b``, exact and local on the anelastic core | yes, inside the flux |
| Exact under phase change | yes, all transitions | at saturation only; non-equilibrium needs ``-\sum \mathcal{G}^β \dot{S}^β / T`` | yes, all transitions | yes, all transitions |
| Energy-conserving under mixing | nearly: ``-\tfrac{1}{4} Δ(\hat{c}ᵖ \hat{Π}) Δθˡⁱ`` per event; creates energy between levels | no: loses ``T σ_{mix}`` at first order | yes | yes |
| Extra sources required | expansion; heating ``/(\hat{c}ᵖ \hat{Π})``; local sedimentation content; (optional) drag, dissipation | every irreversible process: phase change off saturation, drag, dissipation, mixing production; heating ``/T``; sedimentation entropy flux | buoyancy flux; (optional) drag, dissipation | none; sedimentation and surface fluxes in flux form |
| Inversion for ``T`` | closed form, one power; iterative on the compressible core | closed form, one exponential at fixed ``ρ`` with prognostic water; iterative with saturation adjustment | closed form, linear | closed form, linear, after subtracting ``K``, ``g z`` and the latent offset |
| `Float32` concerns | per-step increments below one ulp (all variables) | as ``θˡⁱ``; magnitude ``10^3`` with the triple-point origin; one-bit cancellation in the acoustic pressure | as ``θˡⁱ``; magnitude ``3 × 10^5`` | as ``θˡⁱ``, with the least margin aloft; cancellation of ``g z`` in the inversion and the acoustic pressure; mitigated by `reference_temperature` and a reference subtraction |

## Which variable

For energy conservation the answer is clear, and it depends on the core. On the anelastic core
the static energy ``s`` is conserved under every conservative operation, its one source is the
buoyancy flux that the momentum equation already computes, and its inversion is a division; it is
the energy variable of the anelastic equations in the sense of [Pauluis (2008)](@cite Pauluis2008).
On the compressible core the total energy ``E`` is the conserved quantity itself: expansion, phase
change, dissipation and precipitation drag all stay inside it, and the budget closes to round-off
by construction, at the price of a large-number cancellation that `Float32` must be protected
from. On the compressible core ``s`` needs the pressure tendency that ``E`` avoids; on the
anelastic core the conserved energy is ``s + K`` rather than ``E``, because the pressure work has
been linearized into the buoyancy flux and the pressure is no longer a prognostic variable.

For entropy conservation and the treatment of mixing, there are honest arguments for ``θˡⁱ``
and for ``η``, and they pull in different directions.

The argument for ``η`` is that it is the variable whose conservation *is* the reversible limit.
Adiabatic expansion and saturated phase change leave it unchanged with no source on either core;
every irreversible process appears as an explicit, sign-definite production, so that a model's
second law can be audited term by term; and on the compressible core it removes the pressure
tendency that ``θˡⁱ`` must estimate. The argument against it is what happens when the sources are
not all supplied, which in a discrete model they never are: numerical mixing of ``η`` is
entropy-conserving mixing, which is unphysical — real mixing produces entropy — and the energy it
destroys is first order in the entropy that mixing should have produced. Every omitted production
term, including the subgrid one unless the closure supplies it, is an energy loss of the same
kind, and the omission is silent.

The argument for ``θˡⁱ`` is that its source burden is concentrated where it can be met exactly.
Phase change needs nothing, in or out of equilibrium, so no microphysics scheme has to know what
the thermodynamic variable is; expansion is one local source, exact on the anelastic core; and
in clear air it carries the same information as ``η`` while mixing with an error that is second
order in the differences and does not involve ``Rᵛ T``. Its errors are the bilinear mixing term,
the spurious potential energy of mixing between levels, and, on the compressible core, the
hydrostatic estimate of the pressure tendency — all small, and all measured. The argument against
it is that none of them can be removed from the implicit mixing of the advection scheme, and that
``θˡⁱ`` conserves neither energy nor entropy exactly, only approximately both.

## Results: RCE and Bryan–Fritsch comparisons (pending)

!!! warning "Placeholder"
    This section will report the comparisons as they are completed. The planned diagnostics are
    listed here so that the comparison is defined before the numbers are in.

The comparison runs the same radiative-convective equilibrium configuration with each
thermodynamic variable on each core that supports it, and reports:

- **The energy residual and its decomposition.** The residual is the rate of change of the
  column total energy less the boundary fluxes (radiation, surface fluxes, the enthalpy of
  precipitation). It is split online into the dissipative part of the advection scheme's
  thermodynamic flux, ``S_{num}``; the subgrid flux, ``D_{sgs}``; the mechanical term
  ``D_{mech} = ∫ p \, \boldsymbol{\nabla \cdot u} - \mathrm{d} ∫ ρ (K + g z) / \mathrm{d} t - P_{drag}``,
  which measures the kinetic energy lost without heating; and the drag heating of precipitation,
  ``P_{drag}``. For ``E`` the residual must be round-off; for the state functions it measures the
  missing sources. With the exact ``θˡⁱ`` on the compressible core the residual is ``-2.7`` W m⁻²
  over days 10–20.

- **The entropy budget and the implied irreversible production.** ``η(T, p, q)`` is diagnosed in
  every run, and the material production implied by its budget,
  ``\mathrm{d} ∫ ρ η / \mathrm{d} t - ∫ ρ Q / T - \text{surface entropy flux}``, is compared with
  the explicit physical sources (drag, dissipation, vapor diffusion, irreversible phase change).
  The difference is the numerical production, reported globally and as the fraction of the
  domain where it is negative. An ``η`` run shows zero numerical production by construction and
  pays for it in the energy residual; ``θˡⁱ`` and ``E`` runs show positive production where the
  advection scheme mixes.

- **Dry-versus-moist consistency**, after [Bryan and Fritsch (2002)](@cite BryanFritsch2002): a
  dry thermal in a neutral environment and a saturated thermal in a moist-neutral environment
  of uniform equivalent potential temperature and total water. A formulation that is exact
  under both expansion and phase change preserves the neutrality of the undisturbed moist
  environment and reproduces the benchmark fields; the departures of each variable measure its
  expansion and phase-change errors directly, without radiation, surface fluxes or precipitation.
