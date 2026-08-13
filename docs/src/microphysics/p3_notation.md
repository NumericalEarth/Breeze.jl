# [P3 Notation](@id p3_notation)

The [Notation and conventions](@ref) appendix reserves symbols for the dynamics and
thermodynamics. P3 needs more symbols than that table can absorb without
collisions, so its notation is *scoped*: the symbols defined here hold throughout
the microphysics pages and the `PredictedParticleProperties` module, and the
appendix table holds everywhere else.

The symbols this scope claims, and what each already means outside it — a dash
where the appendix reserves nothing and the letter is P3's alone:

| symbol | here | elsewhere in Breeze |
| ------ | ---- | ------------------- |
| ``N``  | number per unit volume [m⁻³] | grid size (`Nx`), acoustic substep count |
| ``F``  | mass fraction of an ice component, ``F^f`` and ``F^l`` | forcing |
| ``b``  | rime volume per unit mass [m³/kg] | buoyancy |
| ``D``  | particle diameter [m] | — |
| ``A``  | particle projected area [m²] | — |
| ``C``  | particle capacitance [m] | surface transfer coefficients ``Cᴰ``, ``Cᵀ``, ``Cᵛ`` |
| ``V``  | terminal velocity of a single particle [m/s] | — |
| ``μ``  | gamma-PSD shape parameter, never written without a species label (``μ^{cl}``, `μᶜˡ`) | a bare `μ` in kernel code is the microphysical-field tuple, and ``μ`` is *not* the dynamic viscosity here — that is ``η`` |

## Conventions

**``N`` counts per volume, ``n`` counts per mass.** An uppercase ``N`` is a number
density [m⁻³], a lowercase ``n`` is a number mixing ratio [kg⁻¹], and the two are
related by the air density, ``N^x = ρ\, n^x``. The prognostic fields are
``ρn^x``, the process rates are written per unit mass, and the lookup tables and
collection kernels take ``N^x``. The same split applies to mass: ``q^x`` is a mass
fraction [kg/kg] and ``ρq^x`` a partial density [kg/m³].

**Species are superscripts.** Following the appendix convention for phase
identifiers, the species label rides in the superscript — ``q^{cl}``, ``n^r``,
``ρq^i`` — never in the subscript. The labels are `cl` (cloud liquid), `r` (rain),
`i` (dry ice), `f` (rime, i.e. frozen accretion), `wi` (liquid coating on ice),
`v` (vapor), and `a` (aerosol). Subscripts are reserved for process names,
thresholds, and indices. Where a symbol carries both, the code form puts the
species superscript immediately after the letter and the subscript last, so the
rain PSD intercept ``N_0^r`` is `Nʳ₀` — the species stays adjacent to the letter it
labels, as in `λʳ`.

**Saturation is ``^+``.** A saturation value carries a `+` in the superscript, as in
the appendix: ``q^{v+l}`` and ``q^{v+i}`` are the saturation mass fractions over
planar liquid and ice, and ``p^{v+}`` the saturation vapor pressure. Departures
from saturation get their own symbols: ``\mathscr{S}^l`` and ``\mathscr{S}^i``
(`𝒮`) are the supersaturation ratios ``p^v / p^{v+} - 1``, and
``s^{v+l} = q^v - q^{v+l}`` is the liquid supersaturation in mass-fraction form,
which is what the optional prognostic ``ρs^{v+l}`` (`ρsᵛ⁺ˡ`) carries. Nothing in these
pages spells "sat" or "s" as a subscript to mean saturation.

**Free parameters are ``\mathbb{C}``.** Empirically fitted constants do not each
consume a letter. They are collected in ``\mathbb{C}^X``, labelled by the relation
``X`` they belong to and numbered in the order they appear in it, so the
fall-speed power law is ``V(D) = \mathbb{C}^V_1 D^{\mathbb{C}^V_2}`` and the
KK2000 autoconversion fit is ``\mathbb{C}^\text{aut}``. Quantities that vary with
the state are *not* free parameters and keep their own symbols: the mass–diameter
coefficients ``α`` and ``β`` change from one size regime to the next, and the
aggregation efficiency ``E^{ii}(T)`` is a function of temperature.

**Rates are dotted, tendencies are ``G``.** A dot marks a process rate per unit
mass of air: ``\dot{q}`` for mass [kg kg⁻¹ s⁻¹], ``\dot{n}`` for number
[kg⁻¹ s⁻¹], ``\dot{b}`` for rime volume [m³ kg⁻¹ s⁻¹]. The subscript names the
process and the superscript names the species the rate acts on, when the same
process acts on more than one — ``\dot{q}^{cl}_\text{rim}`` is the riming of cloud
water and ``\dot{q}^{r}_\text{rim}`` the riming of rain. The microphysical source
term assembled from those rates for a prognostic field ``ρX`` is ``G_{ρX}``,
matching the appendix use of ``G`` for a tendency.

## Prognostic State

The default configuration carries eight densities; the optional groups bring the
maximum to eleven. See [Prognostic Variables and Tendencies](@ref p3_prognostics).

| math symbol | code | description |
| ----------- | ---- | ----------- |
| ``ρq^{cl}`` | `ρqᶜˡ` | Cloud liquid mass density [kg/m³] |
| ``ρn^{cl}`` | `ρnᶜˡ` | Cloud droplet number density [m⁻³]; only with aerosol activation |
| ``ρq^r``    | `ρqʳ`  | Rain mass density [kg/m³] |
| ``ρn^r``    | `ρnʳ`  | Rain number density [m⁻³] |
| ``ρq^i``    | `ρqⁱ`  | Dry ice mass density [kg/m³] (rime plus deposited mass) |
| ``ρn^i``    | `ρnⁱ`  | Ice number density [m⁻³] |
| ``ρq^f``    | `ρqᶠ`  | Rime mass density [kg/m³] |
| ``ρb^f``    | `ρbᶠ`  | Rime volume density [m³/m³] |
| ``ρq^{wi}`` | `ρqʷⁱ` | Liquid coating on ice, mass density [kg/m³] |
| ``ρq^v``    | `ρqᵛ`  | Water vapor density [kg/m³]; the host-coupled moisture variable |
| ``ρs^{v+l}`` | `ρsᵛ⁺ˡ` | Liquid supersaturation density [kg/m³]; only with `predict_supersaturation` |
| ``ρn^a``    | `ρnᵃ`  | Unactivated aerosol number density [m⁻³]; only with aerosol activation |

## Size Distribution

Each species follows a gamma distribution in maximum dimension ``D``.

| math symbol | code | property name | description |
| ----------- | ---- | ------------- | ----------- |
| ``N'(D)``   |      |               | Number concentration per unit diameter, ``N'(D) = N_0 D^μ e^{-λD}`` [m⁻⁴] |
| ``N_0``     | `N₀` |               | Intercept of the gamma distribution [m⁻⁴⁻μ]; a scale factor, not a concentration. Species-labelled as `Nʳ₀` where the rate needs the rain PSD explicitly |
| ``μ^{cl}``, ``μ^r`` | `μᶜˡ`, `μʳ` | `CloudDropletProperties.shape_parameter`, `RainProperties.shape_parameter` | Shape parameter [-]; ``μ^{cl}`` is diagnosed from ``N^{cl}``, ``μ^r = 0`` at runtime |
| ``μ^i``     | `μⁱ` | | Ice shape parameter [-]; an axis of the ice lookup tables rather than a stored field |
| ``λ^{cl}``, ``λ^r`` | `λᶜˡ`, `λʳ` | | Slope parameter [1/m] |
| ``λ^i``     |      | `IceLambdaLimiter` | Ice slope parameter [1/m], bounded by the mean-size limiter |
| ``M_k``     |      |               | ``k``-th moment of the distribution, ``M_k = N_0\,Γ(k+μ+1)/λ^{k+μ+1}`` |
| ``\bar{D}`` |      |               | Mean diameter, ``M_1/M_0`` [m] |
| ``\bar{m}`` |      |               | Mean particle mass, ``(ρq^i + ρq^{wi})/ρn^i`` [kg] |

## Ice Particle Properties

| math symbol | code | property name | description |
| ----------- | ---- | ------------- | ----------- |
| ``F^f``     | `Fᶠ` | | Rime mass fraction of dry ice [-], ``F^f = ρq^f / ρq^i`` |
| ``F^l``     | `Fˡ` | | Liquid fraction of total ice mass [-], ``F^l = ρq^{wi}/(ρq^i + ρq^{wi})`` |
| ``ρ^f``     | `ρᶠ` | `IceProperties.minimum_rime_density`, `maximum_rime_density` | Rime density [kg/m³], ``ρ^f = ρq^f / ρb^f``, bounded to [50, 900] |
| ``ρ^{gr}``  |      | | Graupel density [kg/m³], ``ρ^{gr} = F^f ρ^f + (1-F^f) ρ^d`` |
| ``ρ^i``     |      | | Bulk ice density [kg/m³], 900, used by the mass–diameter relations |
| ``ρ^i_\text{pure}`` | | `ProcessRateParameters.pure_ice_density` | Density of solid ice [kg/m³], 917, used for reflectivity and melt densification |
| ``m(D)``    |      | | Particle mass, ``m(D) = α D^β`` on each size regime [kg] |
| ``α``, ``β``|      | | Mass–diameter coefficient [kg/m^β] and exponent [-] of the active regime |
| ``A(D)``    |      | | Particle projected area [m²], ``A(D) = \mathbb{C}^A_1 D^{\mathbb{C}^A_2}`` for aggregates |
| ``C(D)``    |      | | Particle capacitance for vapor diffusion [m] |
| ``V(D)``    |      | `RainProperties.fall_speed_coefficient`, `fall_speed_exponent` | Terminal velocity [m/s]; ``V(D) = \mathbb{C}^V_1 D^{\mathbb{C}^V_2}`` for rain, a Best-number formulation for ice |
| ``D^{th}``  |      | | Threshold between small spherical ice and vapor-grown aggregates [m] |
| ``D^{gr}``  |      | | Threshold between aggregates and graupel [m] |
| ``D^{cr}``  |      | | Threshold between graupel and partially rimed ice [m] |

## Bulk and Integral Quantities

| math symbol | code | property name | description |
| ----------- | ---- | ------------- | ----------- |
| ``V_n``, ``V_m`` | | | Number- and mass-weighted mean fall speeds [m/s] |
| ``\mathcal{K}`` | | | A PSD-integrated collection kernel, ``\int A(D) V(D) N'(D)\,dD`` |
| ``E^{ci}`` | `Eᶜⁱ` | `cloud_ice_collection_efficiency` | Ice–cloud droplet collection efficiency [-] |
| ``E^{ri}`` | `Eʳⁱ` | `rain_ice_collection_efficiency` | Ice–rain collection efficiency [-] |
| ``E^{ii}(T)`` | | | Ice–ice aggregation efficiency [-], a function of temperature and ``F^f`` |
| ``f^{ve}`` | `fᵛᵉ` | | Ventilation factor for vapor diffusion [-], ``\mathbb{C}^\text{vent}_1 + \mathbb{C}^\text{vent}_2 \text{Re}^{1/2}\text{Sc}^{1/3}`` |
| ``Q_\text{norm}`` | | | Normalized ice mass, the mean particle mass ``\bar{m}`` [kg]; the first lookup-table axis |
| ``\mathcal{F}_X`` | | | Sedimentation flux of ``ρX`` [kg m⁻² s⁻¹ or m⁻² s⁻¹] |

## Air Properties

Diagnosed once per cell by `air_transport_properties` (`transport_properties.jl`)
and passed to every rate that needs them.

| math symbol | code | description |
| ----------- | ---- | ----------- |
| ``D^v``  | `Dᵛ` | Vapor diffusivity in air [m²/s] |
| ``K^a``  | `Kᵃ` | Thermal conductivity of air [W/m/K] |
| ``η``    | `η`  | Dynamic viscosity of air [Pa s], from Sutherland's law |
| ``ν``    | `ν`  | Kinematic viscosity of air [m²/s], ``ν = η/ρ`` |
| ``\text{Sc}`` | | Schmidt number, ``ν / D^v`` |
| ``\text{Re}`` | | Reynolds number, ``V D / ν`` |
| ``ρ_\text{corr}`` | | Air-density fall-speed correction, ``(ρ_s/ρ)^{0.54}`` |

Thermodynamic constants keep their appendix symbols — ``\mathcal{L}^l`` and
``\mathcal{L}^i`` for the latent heats of condensation and deposition, ``c^{pd}``
for the dry-air heat capacity, ``R^v`` for the vapor gas constant. The one
addition is ``\mathcal{L}^\text{fus} = \mathcal{L}^i - \mathcal{L}^l``, the latent
heat of fusion, which the melting and wet-growth heat balances need. It is *not*
written ``\mathcal{L}^f``, since `f` labels rime here.

## Process Rates

All rates are per unit mass of air. Where a superscript is absent, the process
acts on only one species.

| process | mass | number | volume |
| ------- | ---- | ------ | ------ |
| Condensation / evaporation | ``\dot{q}^{cl}_\text{cond}``, ``\dot{q}^{r}_\text{cond}``, ``\dot{q}^{wi}_\text{cond}``, ``\dot{q}^{r}_\text{evap}``, ``\dot{q}^{wi}_\text{evap}`` | ``\dot{n}^{r}_\text{evap}`` | |
| CCN activation | ``\dot{q}_\text{act}`` | ``\dot{n}_\text{act}`` | |
| Autoconversion | ``\dot{q}_\text{aut}`` | ``\dot{n}^{cl}_\text{aut}``, ``\dot{n}^{r}_\text{aut}`` | |
| Accretion | ``\dot{q}_\text{acc}`` | | |
| Self-collection, breakup | | ``\dot{n}^{cl}_\text{slf}``, ``\dot{n}^{r}_\text{slf}``, ``\dot{n}^{r}_\text{brk}`` | |
| Riming | ``\dot{q}^{cl}_\text{rim}``, ``\dot{q}^{r}_\text{rim}`` | ``\dot{n}^{cl}_\text{rim}``, ``\dot{n}^{r}_\text{rim}`` | |
| Above-freezing collection | ``\dot{q}^{cl}_\text{col}``, ``\dot{q}^{r}_\text{col}`` | ``\dot{n}^{cl}_\text{col}``, ``\dot{n}^{r}_\text{col}`` | |
| Deposition / sublimation | ``\dot{q}_\text{dep}``, ``\dot{q}_\text{sub}`` | ``\dot{n}_\text{sub}`` | |
| Ice nucleation | ``\dot{q}_\text{nuc}`` | ``\dot{n}_\text{nuc}`` | |
| Immersion freezing | ``\dot{q}^{cl}_\text{frz}``, ``\dot{q}^{r}_\text{frz}`` | ``\dot{n}^{cl}_\text{frz}``, ``\dot{n}^{r}_\text{frz}`` | |
| Homogeneous freezing | ``\dot{q}^{cl}_\text{hom}``, ``\dot{q}^{r}_\text{hom}`` | ``\dot{n}^{cl}_\text{hom}``, ``\dot{n}^{r}_\text{hom}`` | |
| Hallett–Mossop splintering | | ``\dot{n}_\text{HM}`` | |
| Aggregation | | ``\dot{n}_\text{agg}`` | |
| Melting | ``\dot{q}_{\text{mlt},p}``, ``\dot{q}_{\text{mlt},f}`` | ``\dot{n}_\text{mlt}`` | |
| Shedding | ``\dot{q}_\text{shed}`` | ``\dot{n}_\text{shed}`` | |
| Wet growth | ``\dot{q}^{cl}_\text{wet}``, ``\dot{q}^{r}_\text{wet}``, ``\dot{q}_\text{wsh}``, ``\dot{q}_\text{wdn}`` | ``\dot{n}_\text{wsh}`` | ``\dot{b}_\text{wdn}`` |
| Refreezing | ``\dot{q}_\text{refr}`` | | |
| Melt densification | | | ``\dot{b}_\text{dens}`` |
| Whole-particle clipping | ``\dot{q}^i_\text{clip}``, ``\dot{q}^f_\text{clip}`` | | ``\dot{b}_\text{clip}`` |
| PSD number correction | | ``\dot{n}^{cl}_\text{corr}``, ``\dot{n}^{r}_\text{corr}``, ``\dot{n}^{i}_\text{corr}`` | |
| Ice number cap | | ``\dot{n}_\text{cap}`` | |

## Timescales and Thresholds

| math symbol | property name | description |
| ----------- | ------------- | ----------- |
| ``τ_\text{sink}`` | `sink_limiting_timescale` | Relaxation time for every sink limiter [s], default 10 |
| ``τ_\text{nuc}``  | | Cooper nucleation relaxation time [s], 10 |
| ``τ_\text{act}``  | `AerosolActivation.activation_timescale` | Droplet activation relaxation time [s], default 1 |
| ``τ_\text{hom}``  | `homogeneous_freezing_timescale` | Homogeneous freezing relaxation time [s] |
| ``N^i_\text{max}`` | `maximum_ice_number_density` | Global ice number cap [m⁻³], ``2 \times 10^6`` |
| ``T_0``           | | Freezing point, 273.15 K |
