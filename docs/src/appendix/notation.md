# Notation and conventions

This appendix establishes a common notation across the documentation and source code.
Each entry lists a mathematical symbol and the Unicode form commonly used in
the codebase, along with a common "property name", and a description.
The property names may take a verbose "English form" or concise "mathematical form" corresponding
to the given Unicode symbol. As properties, mathematical names are usually used
mathematical form is invoked for the elements of a `NamedTuple`.
Mathematical symbols are shown with inline math, while the Unicode column shows the exact glyphs used in code.

The table below reserves symbols for the dynamics and thermodynamics — the equations
of motion, the thermodynamic state, and the numerics that step them — so that new
notation in those can be introduced without colliding with what is already there.
A parameterization dense enough to exhaust the alphabet on its own does not belong
in it: its notation is scoped to its own pages instead, where reusing a letter
reserved here is a local decision rather than a conflict. The
[P3 notation](@ref p3_notation) section does that for the Predicted Particle Properties
scheme.

A few notes about the following table:

* `TC` stands for [`ThermodynamicConstants`](@ref)
* `AM` stands for [`AtmosphereModel`](@ref)
* `RS` stands for [`ReferenceState`](@ref Breeze.Thermodynamics.ReferenceState)
* Note that there are independent concepts of "reference". For example, [`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) involves
  a "reference state", which is an adiabatic, hydrostatic solution to the equations of motion. But there is also an
  "energy reference temperature" and "reference latent heat", which are thermodynamic constants required to define
  the internal energy of moist atmospheric constituents.
* Mapping to AM fields: `ρs` corresponds to `static_energy_density(model)`, and the moisture density is accessed via `model.moisture_density`.

The following table also uses a few conventions that suffuse the source code and which are internalized by wise developers:

* `constants` refers to an instance of `ThermodynamicConstants()`
* `q` refers to an instance of  [`MoistureMassFractions`](@ref Breeze.Thermodynamics.MoistureMassFractions)
* "Reference" quantities use a subscript ``r`` (e.g., ``p_r``, ``\rho_r``).
* A subscript plus marks the *clamped* form of a quantity, ``x₊``: either ``\max(0, x)`` or ``x`` floored at the scheme's minimum (e.g. ``nⁱ₊``, ``qʳ₊`` in P3). The superscript ``⁺`` is reserved for saturation (``qᵛ⁺``).
* Phase or mixture identifiers (``d``, ``v``, ``m``, and ``t`` for *total*) appear as superscripts (e.g., ``Rᵈ``, ``cᵖᵐ``, ``qᵗ``, ``ρᵗ``), matching usage in the codebase (e.g., `Rᵈ`, `cᵖᵐ`).
* The superscript ``s`` is reserved for *surface* (e.g. ``pˢ``, ``θˢ``, ``ρˢ``: values at a column's bottom face). Snow therefore takes ``sn`` (e.g. ``qˢⁿ``, ``ρqˢⁿ``, ``𝕎ˢⁿ``). Other multi-letter superscripts keep their own spellings: ``st`` for *standard* (``pˢᵗ``), ``sw`` for *shortwave* (``τˢʷ``), and ``th`` for *thermodynamic state* (``ρᵗʰ``), distinct from ``t`` for *total*.
* Momentum and the thermodynamic variable are stored *continuity-density-weighted* (``ρu = ρᵈ u``, and the thermodynamic density ``ρᵡ = ρᵈ χ`` — i.e. ``ρθ = ρᵈ θ`` or ``ρs``). The continuity density is `continuity_density(dynamics)`: the reference density ``ρᵣ`` for [`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics), and the prognostic dry-air density ``ρᵈ`` for [`CompressibleDynamics`](@ref Breeze.CompressibleEquations.CompressibleDynamics). Velocity and ``θ`` are recovered by dividing by the continuity density.
* Water/moisture is stored as *partial densities* (`ρqᵛ`, `ρqˡ`, `ρqⁱ`, …; mass per volume) and recovered as **mass fractions** by dividing by the *total* air density ``ρ = ρᵈ + ρᵗ`` (``qˣ = ρˣ/ρ``), where ``ρᵗ = ρqᵛᵉ + Σ ρqᶜ`` is the total condensate density (all phases of the condensable species). The thermodynamics works in mass fractions throughout. The total density (`total_density(dynamics)`) is diagnosed on the compressible core; on the anelastic core it is the reference density ``ρᵣ``, which the anelastic approximation substitutes for ``ρ``. It also weights scalar and water advection, hydrostatics, and radiation.
* Relations of the local thermodynamic state, such as the saturation specific humidity ``qᵛ⁺ = pᵛ⁺ / (ρᵗʰ Rᵛ T)`` and the anelastic buoyancy, use the *thermodynamic state density* ``ρᵗʰ`` (`thermodynamic_state_density`, or `density(𝒰, constants)` for a state ``𝒰``): the equation of state evaluated with the local temperature and moisture. It is ``ρᵗʰ = pᵣ / (Rᵐ T)`` on the anelastic core, which differs from ``ρᵣ`` wherever ``T`` or ``q`` depart from the reference state, and ``ρᵗʰ = ρ`` on the compressible core. What distinguishes ``ρᵗʰ`` is the state it is evaluated at, not its mathematical form: many densities satisfy *some* equation of state (the reference density ``ρᵣ`` is the equation of state at the reference temperature and moisture), while ``ρᵗʰ`` is the one at the *local* temperature and moisture.

| math symbol                         | code   | property name                       | description                                                                    |
| ----------------------------------- | ------ | ----------------------------------- | ------------------------------------------------------------------------------ |
| ``\rho``                            | `ρ`    | `AM.density`                        | Total air density, ``ρ = ρᵈ + ρᵗ``, `total_density(dynamics)`: diagnosed on the compressible core; the reference density ``ρᵣ`` on the anelastic core |
| ``ρᵗʰ``                             | `ρᵗʰ`  | `thermodynamic_state_density`       | Thermodynamic state density, the equation of state with the local ``T`` and ``q``: ``ρᵗʰ = pᵣ / (Rᵐ T)`` on the anelastic core, ``ρᵗʰ = ρ`` on the compressible core. Saturation is evaluated at ``ρᵗʰ``, while ``ρ`` carries mass and weights tendencies |
| ``ρᵈ``                              | `ρᵈ`   | `AM.dynamics.dry_density`           | Dry-air density; the prognostic continuity density of [`CompressibleDynamics`](@ref Breeze.CompressibleEquations.CompressibleDynamics) (total ``ρ = ρᵈ + ρᵗ``) |
| ``ρᵗ``                              | `ρᵗ`   |                                     | Total condensate density, ``ρᵗ = ρqᵛᵉ + Σ ρqᶜ`` (vapor/equilibrium moisture plus all condensed species); diagnosed via `total_condensate_density`, summed over [`condensate_field_names`](@ref Breeze.AtmosphereModels.condensate_field_names) |
| ``\alpha``                          | `α`    |                                     | Specific volume, ``α = 1/ρ``                                                   |
| ``\boldsymbol{u} = (u,v,w)``        | `u, v, w` | `AM.velocities`                  | Velocity components in (x, y, z) or (east, north, up)                          |
| ``\boldsymbol{ρu} = (ρu, ρv, ρw)``  | `ρu, ρv, ρw` | `AM.momentum`                 | Momentum components                                                            |
| ``\tilde{w}``                  | `w̃`    | `AM.dynamics.contravariant_vertical_velocity` | Contravariant vertical velocity (grid-relative, normal to ``r``-surfaces) |
| ``\rho \tilde{w}``             | `ρw̃`   | `AM.dynamics.contravariant_vertical_momentum` | Contravariant vertical momentum                                          |
| ``r``                          | `r`    | `rnode(i, j, k, grid, ℓz)`          | Reference (computational) vertical coordinate of a terrain-following grid; the physical height is ``z(x, y, r)`` (`znode`), matching Oceananigans' `r`/`z` convention |
| ``s``                               | `s`    | `static_energy(model)`              | (Liquid-ice) moist static energy, ``s = cᵖᵐ T + g z - ℒˡ₀ qˡ - ℒⁱ₀ qⁱ``; ``e`` is reserved for turbulent kinetic energy |
| ``ρ s``                             | `ρs`   | `static_energy_density(model)`      | Static energy density, the prognostic thermodynamic variable of `StaticEnergyFormulation` |
| ``E``                               | `E`    | `total_energy(model)`               | Total energy, ``E = s + (u^2 + v^2 + w^2)/2``. `E` is also the formulation-agnostic *energy* key: a flux or forcing supplied under `E` (or `ρE`) is applied to whichever thermodynamic variable the formulation evolves |
| ``ρ E``                             | `ρE`   |                                     | Total energy density; the `boundary_conditions` and `forcing` key for an energy input, routed onto ``ρᵡ`` |
| ``ρᵡ``                              | `ρᵡ`   | `thermodynamic_density(formulation)` | Thermodynamic density: the generic continuity-density-weighted prognostic thermodynamic variable, ``ρᵡ = ρᵈ χ`` — concretely ``ρθ`` for the potential-temperature formulation or ``ρs`` for static energy. The intensive variable is recovered as ``χ = ρᵡ / ρᵈ`` |
| ``T``                               | `T`    | `AM.temperature`                    | Temperature                                                                    |
| ``T⁺``                              | `T⁺`   | `DewpointTemperature(model)`        | Dewpoint temperature                                                           |
| ``p``                               | `p`    | `AM.pressure`                       | Pressure                                                                       |
| ``b``                               | `b`    |                                     | Buoyancy                                                                       |
| ``q^{ve}``                           | `qᵛᵉ`  |                                     | Scheme-dependent specific moisture: vapor (non-equilibrium) or equilibrium moisture (saturation adjustment) |
| ``ρ q^{ve}``                         | `ρqᵛᵉ` | `AM.moisture_density`               | Scheme-dependent moisture density: ``ρqᵛ`` or ``ρqᵉ``                          |
| ``ρ qᵉ``                            | `ρqᵉ`  | `AM.moisture_density`               | Equilibrium moisture density (saturation adjustment schemes)                   |
| ``ρ qᵛ``                            | `ρqᵛ`  | `AM.moisture_density`               | Vapor density (non-equilibrium schemes)                                        |
| ``ρ qᵗ``                            | `ρqᵗ`  |                                     | Total moisture density; the `boundary_conditions` and `forcing` key for a water input (specific alias `qᵗ` for forcings), routed onto ``ρqᵛᵉ`` whatever the microphysics calls it |
| ``qᵛ``                              | `qᵛ`   | `AM.microphysical_fields.qᵛ`        | Vapor mass fraction, a.k.a "specific humidity"                                 |
| ``qᵛ⁺ˡ``                            | `qᵛ⁺ˡ` |                                     | Saturation vapor mass fraction over liquid water (kg/kg)                       |
| ``δqᵛ⁺ˡ``                           | `δqᵛ⁺ˡ` | `AM.microphysical_fields.δqᵛ⁺ˡ`     | Vapor excess over liquid, ``δqᵛ⁺ˡ = qᵛ − qᵛ⁺ˡ`` (kg/kg); P3's optional prognostic |
| ``δqᵛ⁺ⁱ``                           | `δqᵛ⁺ⁱ` |                                     | Vapor excess over ice, ``δqᵛ⁺ⁱ = qᵛ − qᵛ⁺ⁱ`` (kg/kg)                           |
| ``ρδqᵛ⁺ˡ``                          | `ρδqᵛ⁺ˡ` | `AM.microphysical_fields.ρδqᵛ⁺ˡ`    | Density-weighted vapor excess over liquid (kg/m³); may be negative             |
| ``qˡ``                              | `qˡ`   | `AM.microphysical_fields.qˡ`        | Liquid mass fraction                                                           |
| ``qⁱ``                              | `qⁱ`   | `AM.microphysical_fields.qⁱ`        | Ice mass fraction                                                              |
| ``qᶜˡ``                             | `qᶜˡ`  | `AM.microphysical_fields.qᶜˡ`       | Cloud liquid mass fraction                                                     |
| ``qᶜⁱ``                             | `qᶜⁱ`  | `AM.microphysical_fields.qᶜⁱ`       | Cloud ice mass fraction                                                        |
| ``q^f``                             | `qᶠ`   | `AM.microphysical_fields.qᶠ`        | Rime mass fraction (P3); a portion of ``qⁱ``, not additional mass              |
| ``b^f``                             | `bᶠ`   | `AM.microphysical_fields.bᶠ`        | Rime volume per unit mass (P3), m³/kg                                          |
| ``q^{wi}``                          | `qʷⁱ`  | `AM.microphysical_fields.qʷⁱ`       | Liquid mass fraction carried on ice particles (P3); counted in ``qˡ``, but sediments with the ice |
| ``qʳ``                              | `qʳ`   |                                     | Rain mass fraction                                                             |
| ``qˢⁿ``                             | `qˢⁿ`  |                                     | Snow mass fraction                                                             |
| ``\hat{q}^x``                       | `q̂ˣ`   |                                     | Unit composition: mass fractions with all mass in constituent ``x`` (``\hat{q}^d`` is dry air) |
| ``Δq``                              | `Δq`   |                                     | Composition increment (a difference of mass-fraction vectors), e.g. per unit sedimenting mass |
| ``ρqᵛ``                             | `ρqᵛ`  |                                     | Vapor density                                                                  |
| ``ρqˡ``                             | `ρqˡ`  |                                     | Liquid density                                                                 |
| ``ρqⁱ``                             | `ρqⁱ`  |                                     | Ice density                                                                    |
| ``ρqᶜˡ``                            | `ρqᶜˡ` |                                     | Cloud liquid density                                                           |
| ``ρqᶜⁱ``                            | `ρqᶜⁱ` |                                     | Cloud ice density                                                              |
| ``ρqʳ``                             | `ρqʳ`  | `AM.microphysical_fields.ρqʳ`       | Rain density                                                                   |
| ``ρqˢⁿ``                            | `ρqˢⁿ` | `AM.microphysical_fields.ρqˢⁿ`      | Snow density                                                                   |
| ``n^{cl}``                          | `nᶜˡ`  | `AM.microphysical_fields.nᶜˡ`       | Cloud droplet number per unit mass (1/kg)                                      |
| ``n^r``                             | `nʳ`   | `AM.microphysical_fields.nʳ`        | Rain drop number per unit mass (1/kg)                                          |
| ``n^a``                             | `nᵃ`   | `AM.microphysical_fields.nᵃ`        | Aerosol number per unit mass (1/kg)                                            |
| ``\rho n^{cl}``                     | `ρnᶜˡ` | `AM.microphysical_fields.ρnᶜˡ`      | Cloud droplet number density (1/m³)                                            |
| ``\rho n^r``                        | `ρnʳ`  | `AM.microphysical_fields.ρnʳ`       | Rain drop number density (1/m³)                                                |
| ``\rho n^a``                        | `ρnᵃ`  | `AM.microphysical_fields.ρnᵃ`       | Aerosol number density (1/m³)                                                  |
| ``N^{cl}``                          | `Nᶜˡ`  |                                     | Volumetric cloud droplet number density, ``Nᶜˡ = ρ nᶜˡ`` (1/m³)               |
| ``N^r``                             | `Nʳ`   |                                     | Volumetric rain drop number density, ``Nʳ = ρ nʳ`` (1/m³)                      |
| ``N^a``                             | `Nᵃ`   |                                     | Volumetric aerosol number density, ``Nᵃ = ρ nᵃ`` (1/m³)                        |
| ``n^{ccn}``                         | `nᶜᶜⁿ` |                                     | Cloud condensation nuclei activated per unit mass (1/kg); the activation rate ``∂nᶜᶜⁿ/∂t`` is a source of ``nᶜˡ`` |
| ``N^{ccn}``                         | `Nᶜᶜⁿ` |                                     | Volumetric CCN number density, ``Nᶜᶜⁿ = ρ nᶜᶜⁿ`` (1/m³)                        |
| ``q^{ccn}``                         | `qᶜᶜⁿ` |                                     | Mass fraction condensed onto newly activated CCN (kg/kg)                       |
| ``\mathbb{W}^{cl}``                 | `𝕎ᶜˡ`  |                                     | Terminal velocity of cloud liquid (scalar, positive downward)                  |
| ``\mathbb{W}^{ci}``                 | `𝕎ᶜⁱ`  |                                     | Terminal velocity of cloud ice (scalar, positive downward)                     |
| ``\mathbb{W}^r``                    | `𝕎ʳ`   |                                     | Terminal velocity of rain (scalar, positive downward)                          |
| ``\mathbb{W}^{sn}``                 | `𝕎ˢⁿ`  |                                     | Terminal velocity of snow (scalar, positive downward)                          |
| ``\mathbb{W}^i``                    | `𝕎ⁱ`   |                                     | Terminal velocity of ice (scalar, positive downward); ``i`` is dry ice, distinct from cloud ice ``ci`` |
| ``\mathbb{W}^{nx}``                 | `𝕎ⁿˣ`  |                                     | Number-weighted terminal velocity of species ``x`` (`𝕎ⁿᶜˡ`, `𝕎ⁿʳ`, `𝕎ⁿⁱ`); a bare species label is the mass-weighted mean, and the weighting marker is a superscript preceding the species |
| ``w^x``                             | `wˣ`   |                                     | Signed vertical advection velocity of species ``x``, ``wˣ = -𝕎ˣ`` (`wᶜˡ`, `wⁿᶜˡ`, `wʳ`, `wⁿʳ`, `wⁱ`, `wⁿⁱ`) |
| ``qᵛ⁺``                             | `qᵛ⁺`  |                                     | Saturation specific humidity over a surface                                    |
| ``qᵛ⁺ˡ``                            | `qᵛ⁺ˡ` |                                     | Saturation specific humidity over a planar liquid surface                      |
| ``qᵛ⁺ⁱ``                            | `qᵛ⁺ⁱ` |                                     | Saturation specific humidity over a planar ice surface                         |
| ``pᵛ``                              | `pᵛ`   |                                     | Vapor pressure (partial pressure of water vapor), ``pᵛ = ρ qᵛ Rᵛ T``           |
| ``pᵛ⁺``                             | `pᵛ⁺`  |                                     | Saturation vapor pressure                                                      |
| ``\mathscr{H}``                     | `ℋ`    | `RelativeHumidity(model)`           | Relative humidity, ``ℋ = pᵛ / pᵛ⁺``                                            |
| ``\mathscr{S}``                     | `𝒮`    | `supersaturation(T, ρ, q, c, surf)`, `Supersaturation(model)` | Supersaturation, ``𝒮 = pᵛ / pᵛ⁺ - 1``            |
| ``ξ``                               | `ξ`    | `psychrometric_correction`           | Psychrometric correction, ``ξ = 1 + ℒ² qᵛ⁺ / (cᵖ Rᵛ T²)``; ``ξˡ`` and ``ξⁱ`` name the liquid and ice phase |
| ``δqˡ``, ``δqⁱ``                    | `δqˡ`, `δqⁱ` |                               | Saturation-adjustment increments, ``δq = (qᵛ - qᵛ⁺) / ξ`` for the liquid and ice phase |
| ``δqᵛ⁺ˡ``                           | `δqᵛ⁺ˡ` |                                    | Vapor excess over liquid, ``δqᵛ⁺ˡ = qᵛ - qᵛ⁺ˡ``; its grid mean is ``μ`` in the subgrid condensation (kg/kg) |
| ``Γ``                               | `Γ`    | `cloud_condensation_correlation`, `subgrid_vapor_excess_correlation` | Subgrid supersaturation correlation added to the grid-mean vapor excess ``μ`` in P3 cloud condensation and evaporation: ``Γ = σ G(s̄/σ) - s̄ ≥ 0``, the Gaussian truncation about the grid-mean saturation excess of total water ``s̄ = μ + ξˡ qᶜˡ``, so that ``μ + Γ = ⟨s⁺⟩ - ξˡ qᶜˡ``; with plumes, ``Γ = Σᵢ aᵢ [σᵢ G(s̃ᵢ/σᵢ) - s̃ᵢ]`` (kg/kg) |
| ``G``                               | `G`    |                                     | ``G(Q) = φ(Q) + Q Φ(Q)``, with ``φ`` and ``Φ`` the standard normal density and distribution function |
| ``σ``                               | `σ`    | `supersaturation_statistics(...).σ` | Subgrid standard deviation of ``δqᵛ⁺ˡ`` (per plume ``σᵢ``, with weights ``aᵢ`` and means ``μᵢ``) (kg/kg) |
| ``f⁺``                              | `f⁺`   | `saturated_fraction(supersaturation_statistics(...))` | Saturated fraction of a cell, the probability that the vapor excess is positive; the weight of `SaturationWeightedStaticStability()` |
| ``χ``                               | `χ`    | `placed_cloud_liquid`               | Fraction of the cloud liquid in the saturated part of the subgrid distribution (`AllProcesses` of `SubgridCondensation`): the steady state of the subsaturated liquid between exchange at ``ωᵐ`` and evaporation, ``χ = (ωˡ |⟨s⁻⟩| + f⁺ (1 - f⁺) ωᵐ qᶜˡ) / (ωˡ |⟨s⁻⟩| + (1 - f⁺) ωᵐ qᶜˡ)``, at most ``⟨s⁺⟩ / (ξˡ qᶜˡ)`` |
| ``ωᵐ``                              | `ωᵐ`   | `supersaturation_statistics` of `TKEBasedTurbulenceClosure` | Exchange rate of air between the saturated and subsaturated parts of the subgrid distribution, ``√e / (ℂᵐ ℓ)`` (s⁻¹) |
| ``ωᶜ``                              | `ωᶜ`   | `phase_relaxation_rate`             | Phase relaxation rate: the coefficient of the vapor excess in the condensation rate, ``∂ₜqᶜ = ωᶜ δqᵛ⁺``; ``ωᶜ = ωˡ + ωⁱ``, 0 without phase change, ``∞`` for saturation adjustment (s⁻¹) |
| ``ωˡ``, ``ωⁱ``                      | `ωˡ`, `ωⁱ` | `p3_phase_relaxation_rates`     | Cloud-liquid and ice parts of ``ωᶜ`` in P3: ``ωˡ = 4π Dᵛ Nᶜˡ r̄ / ξˡ`` with ``r̄`` the mean droplet radius, ``ωⁱ = ℂᵈⁱᶠᶠ₁ εⁱ / ξⁱ + εʷⁱ / ξˡ`` from the ice vapor relaxation coefficients (s⁻¹) |
| ``g``                               | `g`    | `TC.gravitational_acceleration`     | Gravitational acceleration                                                     |
| ``c^{ac}``                          | `cᵃᶜ`  |                                     | Acoustic sound speed, ``cᵃᶜ = \sqrt{γ Rᵈ T}``                                  |
| ``\mathbb{C}_{X,i}``               | `ℂˣᵢ`  | descriptive parameter property      | The ``i``-th calibratable empirical coefficient in relation ``X``; source uses modifier letters because Unicode lacks general subscript letters; state, physical constants, case inputs, switches, and numerical safeguards do not receive ``\mathbb{C}`` |
| ``\mathcal{R}``                     | `ℛ`    | `TC.molar_gas_constant`             | Universal (molar) gas constant                                                 |
| ``Tᵗʳ``                             | `Tᵗʳ`  | `TC.triple_point_temperature`       | Temperature at the vapor-liquid-ice triple point                               |
| ``pᵗʳ``                             | `pᵗʳ`  | `TC.triple_point_pressure`          | Pressure at the vapor-liquid-ice triple point                                  |
| ``mᵈ``                              | `mᵈ`   | `TC.dry_air.molar_mass`             | Molar mass of dry air                                                          |
| ``mᵛ``                              | `mᵛ`   | `TC.vapor.molar_mass`               | Molar mass of vapor                                                            |
| ``Rᵈ``                              | `Rᵈ`   | `dry_air_gas_constant(constants)`   | Dry air gas constant (``Rᵈ = \mathcal{R} / mᵈ``)                               |
| ``Rᵛ``                              | `Rᵛ`   | `vapor_gas_constant(constants)`     | Water vapor gas constant (``Rᵛ = \mathcal{R} / mᵛ``)                           |
| ``Rᵐ``                              | `Rᵐ`   | `mixture_gas_constant(q, constants)` | Mixture gas constant, function of ``q``                                       |
| ``cᵖᵈ``                             | `cᵖᵈ`  | `TC.dry_air.heat_capacity`          | Heat capacity of dry air at constant pressure                                  |
| ``cᵖᵛ``                             | `cᵖᵛ`  | `TC.vapor.heat_capacity`            | Heat capacity of vapor at constant pressure                                    |
| ``cˡ``                              | `cˡ`   | `TC.liquid.heat_capacity`           | Heat capacity of the liquid phase (incompressible)                             |
| ``cⁱ``                              | `cⁱ`   | `TC.ice.heat_capacity`              | Heat capacity of the ice phase (incompressible)                                |
| ``\rho^L``                          | `ρᴸ`   | `TC.liquid.density`                 | Intrinsic density of liquid water                                                        |
| ``\rho^I``                          | `ρᴵ`   | `TC.ice.density`                    | Intrinsic density of ice                                                                 |
| ``cᵖᵐ``                             | `cᵖᵐ`  | `mixture_heat_capacity(q, constants)` | Mixture heat capacity at constant pressure                                   |
| ``cᵛᵐ``                             | `cᵛᵐ`  |                                     | Mixture heat capacity at constant volume, ``cᵛᵐ = cᵖᵐ - Rᵐ``                   |
| ``\hat{c}ᵖ``                        | `ĉᵖ`   | `vapor_equivalent_heat_capacity(q, constants)` | Vapor-equivalent heat capacity, ``ĉᵖ = qᵈ cᵖᵈ + qᵗ cᵖᵛ``: the heat capacity if all water were vapor; unchanged by phase change |
| ``\hat{R}``                         | `R̂`    | `vapor_equivalent_gas_constant(q, constants)` | Vapor-equivalent gas constant, ``R̂ = qᵈ Rᵈ + qᵗ Rᵛ``; equals ``Rᵐ`` in unsaturated air |
| ``\hat{κ}``                         | `κ̂`    | `exner_exponent(VaporEquivalentExponent(), q, constants)` | Vapor-equivalent Exner exponent, ``κ̂ = R̂ / ĉᵖ`` (the default exponent of ``θˡⁱ``); ``κᵈ = Rᵈ / cᵖᵈ`` is the `DryExponent`. In code, `κ̂` also names the exponent choice (`VaporEquivalentExponent()` or `DryExponent()`) passed to functions. Distinct from the von Kármán constant ``κ`` |
| ``Tᵣ``                              | `Tᵣ`   | `TC.energy_reference_temperature`   | Reference temperature for internal energy relations and latent heat            |
| ``\mathcal{L}^l_r``                 | `ℒˡᵣ`  | `TC.liquid.reference_latent_heat`   | Latent heat of condensation at the energy reference temperature                |
| ``\mathcal{L}^i_r``                 | `ℒⁱᵣ`  | `TC.ice.reference_latent_heat`      | Latent heat of deposition at the energy reference temperature                  |
| ``ℒˡ₀, ℒⁱ₀``                       | `ℒˡ₀`, `ℒⁱ₀` | `absolute_zero_latent_heat(TC, phase)` | Latent-heat intercepts at zero kelvin, used with absolute-temperature sensible energy |
| ``H^c``                             | `Hᶜ`   | `condensate_latent_energy(q, TC)`   | Latent energy of the condensate, ``Hᶜ = ℒˡ₀ qˡ + ℒⁱ₀ qⁱ``, so that ``H = cᵖᵐ T - Hᶜ`` |
| ``\mathcal{L}^l(T)``                | `ℒˡ`   | `liquid_latent_heat(T, constants)`  | Temperature-dependent latent heat of condensation                              |
| ``\mathcal{L}^i(T)``                | `ℒⁱ`   | `ice_latent_heat(T, constants)`     | Temperature-dependent latent heat of deposition                                |
| ``\mathcal{L}^{fr}(T)``              | `ℒᶠʳ`  | `fusion_latent_heat(constants, T)`  | Temperature-dependent latent heat of fusion (freezing), ``ℒᶠʳ = ℒⁱ - ℒˡ``; the superscript `fr` abbreviates freezing, since `f` labels rime (`qᶠ`, `bᶠ`) |
| ``θ₀``                              | `θ₀`   | `RS.potential_temperature`          | (Constant) reference potential temperature for the anelastic formulation       |
| ``p₀``                              | `p₀`   | `RS.base_pressure`                  | Reference pressure at ``z = 0``: the datum the reference profiles are anchored to |
| ``pˢ``                              | `pˢ`   | `RS.surface_pressure`               | Reference pressure at a column's bottom face (the terrain surface over terrain), ``p₀`` reduced to that height |
| ``p^{st}``                          | `pˢᵗ`  | `RS.standard_pressure`              | Standard pressure for potential temperature (default 10⁵ Pa)                   |
| ``ρᵣ``                              | `ρᵣ`   | `RS.density`                        | Density of a dry reference state for the anelastic formulation                 |
| ``αᵣ``                              | `αᵣ`   |                                     | Specific volume of a dry reference state, ``αᵣ = Rᵈ θ₀ / pᵣ``                  |
| ``pᵣ``                              | `pᵣ`   | `RS.pressure`                       | Pressure of a dry adiabatic reference pressure for the anelastic formulation   |
| ``\Pi``                             | `Π`    | `exner_function(𝒰, constants)`      | Exner function of the liquid-ice potential temperature, ``Π = (p / pˢᵗ)^κ``, with ``κ = κ̂`` (`VaporEquivalentExponent`, default) or ``κᵈ`` (`DryExponent`); ``p = pᵣ`` on the anelastic core |
| ``\hat{Π}``                         | `Π̂`    |                                     | Vapor-equivalent Exner function, ``Π̂ = (p / pˢᵗ)^{κ̂}`` |
| ``θᵛ``                              | `θᵛ`   |                                     | Virtual potential temperature                                                  |
| ``θᵉ``                              | `θᵉ`   |                                     | Equivalent potential temperature                                               |
| ``θˡⁱ``                             | `θˡⁱ`  | `LiquidIcePotentialTemperature(model)` | Liquid-ice potential temperature, ``θˡⁱ = H / (ĉᵖ Π)``; ``θˡⁱ = T / Π`` in unsaturated air |
| ``H``                               | `H`    |                                     | Liquid-ice enthalpy, ``H = cᵖᵐ T - ℒˡ₀ qˡ - ℒⁱ₀ qⁱ = ĉᵖ Π θˡⁱ``; invariant under phase change at fixed ``p`` and ``qᵗ`` |
| ``θᵇ``                              | `θᵇ`   |                                     | Stability-equivalent potential temperature (for moist Brunt-Väisälä)           |
| ``θ``                               | `θ`    |                                     | Shorthand for liquid-ice potential temperature (used in `set!`) |
| ``\Delta t``                        | `Δt`   | `Simulation.Δt`                     | Time step.                                                                     |
| ``\boldsymbol{\tau}``               | `τ`    |                                     | Kinematic subgrid/viscous stress tensor (per unit mass)                        |
| ``\boldsymbol{\mathcal{T}}``        | `𝒯`    |                                     | Dynamic stress tensor used in anelastic momentum, ``\mathcal{T} = ρᵣ τ``       |
| ``\boldsymbol{J}``                  | `J`    |                                     | Dynamic diffusive flux for scalars                                             |
| ``τˣ``                              | `τˣ`   |                                     | Surface momentum flux (``x``-component), N/m²                                  |
| ``τʸ``                              | `τʸ`   |                                     | Surface momentum flux (``y``-component), N/m²                                  |
| ``\mathcal{Q}^T``                   | `𝒬ᵀ`   |                                     | Surface sensible heat flux, ``\mathcal{Q}^T = cᵖᵐ Jᵀ``                         |
| ``\mathcal{Q}^v``                   | `𝒬ᵛ`   |                                     | Surface latent heat flux, ``\mathcal{Q}^v = \mathcal{L}^l Jᵛ``                 |
| ``Jᵀ``                              | `Jᵀ`   |                                     | Surface temperature flux, kg K/m²s                                             |
| ``Jᶿ``                              | `Jᶿ`   |                                     | Surface potential-temperature flux, ``Jᶿ = 𝒬ᵀ / (ĉᵖ Π)``, kg K/m²s            |
| ``Jᵛ``                              | `Jᵛ`   |                                     | Surface moisture flux, kg/m²s                                                  |
| ``Cᴰ``                              | `Cᴰ`   |                                     | Surface drag coefficient                                                       |
| ``Cᵀ``                              | `Cᵀ`   |                                     | Surface sensible heat transfer coefficient (Stanton number)                    |
| ``Cᵛ``                              | `Cᵛ`   |                                     | Surface vapor transfer coefficient (Dalton number)                             |
| ``Uᵍ``                              | `Uᵍ`   | `gustiness`                         | Gustiness: the unresolved wind added to the resolved wind in the bulk surface fluxes, m/s; ``Uᵍ = \max(c \sqrt{e}, Uᵍ_{min})`` for `TKEGustiness` |
| ``Ũ``                               | `Ũ`    |                                     | Effective wind speed of the bulk surface fluxes and their transfer coefficients, ``Ũ = \sqrt{u² + v² + Uᵍ²}``, m/s |
| ``\ell``                            | `ℓ`    |                                     | A **length scale**, m; the superscript says which one. Bare ``ℓ`` is used locally where only one length scale is in play (the primary mixing length in `TKEBasedTurbulenceClosure`, the divergence-damping scale in `CompressibleEquations`) |
| ``\ell^b``                          | `ℓᵇ`   |                                     | Buoyancy penetration depth, ``ℓᵇ = \sqrt{e} / N``, how far an eddy with kinetic energy ``e`` penetrates air of buoyancy frequency ``N``; infinite where ``N² ≤ 0``. Every mixing-length formulation is built from it and the wall length ``ℂˢ z``: the local minimum ``\min(ℂˢ z, ℓᵇ)``, its interior envelope with slope ``ℂⁱ``, capped separately by ``ℂˢ z`` (the default), or the Bougeault–Lacarrère parcel lengths |
| ``\ell^u, \ell^c, \ell^e, \ell^D``  | `ℓᵘ`, `ℓᶜ`, `ℓᵉ`, `ℓᴰ` |                    | Mixing lengths for momentum, scalars, TKE and dissipation: ``ℓᵘ = 𝕊ᵘ ℓ``, ``ℓᶜ = 𝕊ᶜ ℓ``, ``ℓᵉ = 𝕊ᵉ ℓ``, ``ℓᴰ = ℓ / 𝕊ᴰ`` |
| ``\mathbb{S}^u, \mathbb{S}^c, \mathbb{S}^e, \mathbb{S}^D``              | `𝕊ᵘ`, `𝕊ᶜ`, `𝕊ᵉ`, `𝕊ᴰ` |                    | Stability functions of `TKEBasedTurbulenceClosure`; constants ``ℂᵘ, ℂᶜ, ℂᵉ, ℂᴰ`` in `ConstantStabilityFunctions` |
| ``\mathbb{S}^v_1, \mathbb{S}^v_2, \mathbb{S}^v_3, ρ^v`` | `𝕊ᵛ₁`, `𝕊ᵛ₂`, `𝕊ᵛ₃`, `ρᵛ` | `DoubleGaussian(; variance_stability_functions = VarianceStabilityFunctions())` | Stability functions of the local scalar second moments of `TKEBasedTurbulenceClosure`: ``θˡⁱ′² = 𝕊ᵛ₁ ℓ² (∂_z θˡⁱ)²``, ``qᵗ′² = 𝕊ᵛ₂ ℓ² (∂_z qᵗ)²``, ``θˡⁱ′qᵗ′ = 𝕊ᵛ₃ ℓ² ∂_z θˡⁱ ∂_z qᵗ``, ``𝕊ᵛ₃ = ρᵛ \sqrt{𝕊ᵛ₁ 𝕊ᵛ₂}``, with ``ρᵛ ∈ [-1, 1]`` the correlation magnitude of θˡⁱ and qᵗ; functions of ``v`` (`TurbulentRichardsonStabilityFunction`). Without them, ``𝕊ᵛ_i = (2ℂᵛ_i/ℂᵠ) 𝕊ᶜ/𝕊ᴰ`` |
| ``ℂᵛ₁^{-,0,+}, ℂᵛ₂^{-,0,+}, ℂᵛ₃^{-,m,+}`` | `ℂᵛ₁⁻`, `ℂᵛ₁⁰`, `ℂᵛ₁⁺`, `ℂᵛ₂⁻`, …, `ℂᵛ₃ᵐ`, `ℂᵛ₃⁺` | `closure_parameters(closure)` | Coefficients of `VarianceStabilityFunctions`: strongly unstable, neutral and strongly stable values of ``𝕊ᵛ₁`` and ``𝕊ᵛ₂`` (positive), and the Bernstein control values of ``ρᵛ`` (within ``[-1, 1]``) |
| ``\ell^r``                          | `ℓʳ`   | `roughness_length`                  | Surface roughness length for momentum, m. Superscript ``r`` elsewhere denotes *rain* (``qʳ``); the two never appear together |
| ``\ell^{rh}``                       | `ℓʳʰ`  | `scalar_roughness_length`           | Surface roughness length for heat and moisture; defaults to ``ℓʳ/7.3`` |
| ``\kappa``                          | `κ`    |                                     | von Kármán constant. Never used for a diffusivity in Breeze's own code — the Oceananigans accessors literally named `κᶠᶜᶜ`, `κᶜᶠᶜ`, `κᶜᶜᶠ` are the one exception, and they are thin one-liners |
| ``K^u``                             | `Kᵘ`   |                                     | Eddy diffusivity for momentum, ``K^u = ℓ^u \sqrt{e}``, m² s⁻¹. Oceananigans' scalar-diffusivity closures spell their own field `νₑ`, which Breeze reads but does not define |
| ``K^c``                             | `Kᶜ`   |                                     | Eddy diffusivity for scalars, ``K^c = ℓ^c \sqrt{e}``, m² s⁻¹                    |
| ``K^e``                             | `Kᵉ`   |                                     | Eddy diffusivity for TKE, ``K^e = ℓ^e \sqrt{e}``, m² s⁻¹                        |
| ``e``                               | `e`    |                                     | Subgrid turbulent kinetic energy, m² s⁻²                                       |
| ``eᵐⁱⁿ`` | `eᵐⁱⁿ` | `minimum_tke` | Floor on ``e`` inside ``\sqrt{e}`` of the diffusivities and length scales, and of the buoyancy sink rate ``ωᴮ = -B⁻ / \max(e, eᵐⁱⁿ)``, m² s⁻² |
| ``ωᴮ`` | `ωᴮ` | | Implicit buoyancy sink rate of the TKE equation, ``-B⁻ / \max(e, eᵐⁱⁿ)`` with ``B⁻`` the negative part of the buoyancy flux, s⁻¹ |
| ``\rho e``                          | `ρe`   |                                     | Prognostic TKE density, the tracer `TKEBasedTurbulenceClosure` adds to the model |
| ``\mathrm{Pr}``                     | `Pr`   |                                     | Turbulent Prandtl number, ``\mathrm{Pr} = K^u/K^c = \mathbb{S}^u/\mathbb{S}^c``                 |
| ``Ri``                              | `Ri`   |                                     | Gradient Richardson number, ``Ri = N²/S²``; the rational stability functions use ``N² / (S² + S²ᵐⁱⁿ)`` |
| ``ℂˢ, ℂⁱ`` | `ℂˢ`, `ℂⁱ` | | Wall-length coefficient and independent interior envelope slope. Coefficients of `TKEBasedTurbulenceClosure` are written ``ℂ``, reserving ``C`` for bulk transfer coefficients such as the surface drag ``Cᴰ`` |
| ``ℂᵘ, ℂᶜ, ℂˡ₁₋₅, ℂᵛ₁₋₃, ℂʷ₁₋₁₁, ℂᵝ₁₋₄, ℂᴰ₁₋₅, ℂᴾ, ℂᵍ₁₋₄`` | `ℂᵘ`, `ℂᶜ`, `ℂˡ₁`, …, `ℂᵍ₄` | `closure_parameters(closure)` | Coefficients of the components of `SimplifiedHigherOrderClosure`, each a field of the component that uses it: eddy diffusivities of momentum and scalars (``ℂᵘ, ℂᶜ``), mixing length (``ℂˡ``), second-moment variances (``ℂᵛ``), vertical velocity variance and third moment (``ℂʷ``), stability damping of the return-to-isotropy time scale (``ℂᵝ``), dissipation (``ℂᴰ``), the share ``ℂᴾ`` of the shear production that feeds the TKE (the rest heats the air) and the assumed double-Gaussian PDF (``ℂᵍ``). The docstrings of the components (`BlendedMixingLength`, `ColumnStabilityDamping`, `DowngradientSecondMoments`, `AlgebraicThirdMoment`, `DoubleGaussianPDF`) map each to its name in E3SM `shoc.F90`, or to the literal it replaces. ``ℂˡ₅`` is dimensional (`DIMENSIONAL_COEFFICIENTS`). Bounds, tolerances and switch thresholds (for example `maximum_length`, `maximum_time_scale`, `maximum_skewness`, `plume_fraction_min`) are fields too, listed by `closure_scalars(closure)`, and are not written ``ℂ``. Here ``ℂᵘ, ℂᶜ`` multiply the return-to-isotropy time scale and ``e``: the stability functions are ``𝕊ᵘ = ℂᵘ τ \sqrt{e} / ℓ``, ``𝕊ᶜ = ℂᶜ τ \sqrt{e} / ℓ`` |
| ``αᴺ = τ² N²`` | `αᴺ` | — | Stratification number of the third-moment diagnosis of `AlgebraicThirdMoment` (`buoy_sgs2` in E3SM `shoc.F90`; the notation of `TKEDissipationVerticalDiffusivity`), with ``τ`` the return-to-isotropy time scale; bounded there by the closure's `stratification_number_bound` (`UnboundedStratificationNumber`, or `RationalSaturatedStratificationNumber`: ``αᴺ/(1 + αᴺ/αᴺᵐᵃˣ)`` for ``αᴺ > 0`` with ``αᴺᵐᵃˣ = γ/(a₁ + a₃)``, ``γ`` = `pole_fraction`) |
| ``ℂᵗ`` | `ℂᵗ` | `StratificationLimitedMomentTimeScale(; ℂᵗ)` | Time scale of the moments of a `DoubleGaussian` in eddy turnover times ``ℓ / \sqrt{e}`` |
| ``ℂᴺ`` | `ℂᴺ` | `StratificationLimitedMomentTimeScale(; ℂᴺ)` | Largest product ``τ N`` of the time scale of the moments and the buoyancy frequency (``τ = τ₀ / (1 + (τ₀ N⁺/ℂᴺ)ᵖ)^{1/p}``, so ``αᴺ = τ² N² < (ℂᴺ)²``) |
| ``ℂʳ`` | `ℂʳ` | `DoubleGaussian(; ℂʳ)` | Maximum flux correlation of the subgrid distribution of `TKEBasedTurbulenceClosure`: the variances ``θˡⁱ′², qᵗ′²`` are raised by a smooth maximum so that ``|w′a′| / \sqrt{w′² a′²} ≲ ℂʳ``, and ``θˡⁱ′qᵗ′`` scaled with them; requires ``0 < (ℂʳ)² < 1 - ℂᵍ₁``, above which the double-Gaussian plumes have no width. `nothing` (the default) for no bound |
| ``ℂᵠ`` | `ℂᵠ` | `DissipationTimeScale(; ℂᵠ)` | Ratio of the TKE dissipation time ``e/ε = ℓ/(𝕊ᴰ \sqrt{e})`` to the scalar-variance dissipation time ``τᵠ`` of the subgrid distribution of `TKEBasedTurbulenceClosure`; the second moments use ``τ = 2τᵠ`` (production = dissipation), so ``ρ² = 3 ℂᵠ 𝕊ᶜ 𝕊ᴰ / (4 ℂᵛ ℂʷ₁)``, which requires ``ℂᵠ < 4 ℂᵛ ℂʷ₁ (1 - ℂᵍ₁) / (3 \max 𝕊ᶜ 𝕊ᴰ)``. With `EddyTurnoverTimeScale`, ``τᵠ = ℓ/(ℂᵠ \sqrt{e})`` and ``𝕊ᴰ`` drops out of both |
| ``ℂᵐ`` | `ℂᵐ` | `DoubleGaussian(; ℂᵐ)` | Mixing time of air between the saturated and subsaturated parts of the subgrid distribution in eddy turnover times ``ℓ/√e``; default 1 |
| ``ℂʷᵉ, ℂʷᵋ, ℂʷᴸ, ℂʷʳ, ℂʷᵇ, ℂʷˡ, ℂʷᵍ, ℂʷᴺ, ℂʷᵀ`` | `ℂʷᵉ`, `ℂʷᵋ`, `ℂʷᴸ`, `ℂʷʳ`, `ℂʷᵇ`, `ℂʷˡ`, `ℂʷᵍ`, `ℂʷᴺ`, `ℂʷᵀ` | `TKEBudgetThirdMoment(; ℂʷᵉ, ...)` | Coefficients of the third moment of vertical velocity from the vertical integral of the TKE budget, ``w′³ = r (ℂʷᵉ Fᵉ + ℂʷᵋ Fᵋ) + τ₃ (\sqrt{e} (ℂʷᵇ B⁺ + ℂʷˡ (1 - aᶜ) Bˡ⁺) - ℂʷᵍ w′² ∂_z w′²) / (1 + ℂʷᴺ δ \mathrm{p}(αᴺ/δ))`` with ``\mathrm{p}(x) = (x + \sqrt{x² + 1})/2``, ``τ₃ = ℓ/\sqrt{e}`` and ``αᴺ = τ₃² N²``: of the budget fluxes with the sink shapes ``e`` and ``ε`` (``ℂʷᵉ, ℂʷᵋ``), the exchange length of the sink amplitude in mixing lengths (``ℂʷᴸ``, `Inf` for one amplitude per column), the amplitude of the dependence of the ratio ``r`` on the stratification number (``ℂʷʳ``), of the buoyant source, of the latent source (``aᶜ`` the cloud fraction and ``Bˡ = (b_{qˡ} - b_{qᵛ}) w′qˡ′`` the part of the buoyancy flux carried by the flux of liquid, both of the distribution without skewness) and of the gradient production of the damped local third moment (``ℂʷᵇ, ℂʷˡ, ℂʷᵍ``), of the stratification number in its damping (``ℂʷᴺ``; the width ``δ`` is the field `stratification_damping_width`), and the share of the production the budget flux transports in the TKE equation (``ℂʷᵀ``) |
| ``ℂᑫ, ℂᑫᴸ, ℂʷᑫ`` | `ℂᑫ`, `ℂᑫᴸ`, `ℂʷᑫ` | `HeatingBudgetFlux(; ℂᑫ, ...)` | Coefficients of the budget flux of diabatic heating, ``(ρJ)_{k+½} - (ρJ)_{k-½} = ρ_k Δz_k (𝒬_k - s_k)``, ``Jᑫ = ℂᑫ \max(J, 0)``: the share of the destabilizing flux applied to ``θˡⁱ`` and converted to TKE, in ``[0, 1]`` (``ℂᑫ``), the exchange length of the redistributed heating ``s`` in mixing lengths (``ℂᑫᴸ``), and the coefficient of the conversion of the flux driven by cooling in the damped local third moment of `TKEBudgetThirdMoment` (``ℂʷᑫ``) |
| ``c = e/u_\star²`` | `e_u★²` | `RationalStabilityFunction(SurfaceLayerSimilarity(); e_u★²)` | Neutral surface-layer ratio of TKE to friction velocity squared, from which the neutral stability functions follow (``𝕊ᵘ⁰ = κ/(ℂˢ\sqrt{c})``, ``𝕊ᴰ⁰ = ℂˢ/(κ c^{3/2})``) |
| ``Pr⁰, Pr⁻, Pr⁺`` | `Pr⁰`, `Pr⁻`, `Pr⁺` | `RationalStabilityFunction(SurfaceLayerSimilarity(); Pr⁰, Pr⁻, Pr⁺)` | Turbulent Prandtl number at neutral, in the unstable limit and in the stable limit |
| ``Ri_c`` | `Ri_critical` | `RationalStabilityFunction(SurfaceLayerSimilarity(); Ri_critical)` | Gradient Richardson number of the steady state of the TKE equation in stratified shear with ``ℓ = \sqrt{e}/N`` |
| ``δ`` | `width` | `DoubleGaussian(; clipping)` | Smoothing length of a bound of the components of a `DoubleGaussian` with `SmoothClipping`, as a fraction of the scale of the bounded quantity: ``\max(a, b) → (a + b + \sqrt{(a - b)² + δ²}) / 2`` |
| ``ℓᵐ`` | `ℓᵐ` | | Stored turbulence mixing length (`closure_fields.ℓᵐ`) |
| ``ℓˢ`` | `ℓˢ` | | Stratification-only envelope before capping at the surface length |
| ``Ri⁻, Ri⁺`` | `Ri⁻`, `Ri⁺` | | Unstable and stable transition scales of the rational stability functions |
| ``S²ᵐⁱⁿ`` | `S²ᵐⁱⁿ` | `minimum_shear_squared` | Shear floor of the Richardson number of the rational stability functions, ``Ri = N² / (S² + S²ᵐⁱⁿ)``, s⁻² |
| ``p`` | `p` | `exponent` | Exponent of the `HarmonicBlend` of the lengths that bound the mixing length, ``(ℓ₁^{-p} + ℓ₂^{-p})^{-1/p}`` (elsewhere ``p`` is pressure) |
| ``α_N, α_S`` | `αᴺ`, `αˢ` | | Stratification number ``τ² N²`` and shear number ``τ² (S² + S²ᵐⁱⁿ)`` of turbulence with the time scale ``τ = ℓ / \sqrt{e}`` (`TurbulentRichardsonStabilityFunction`) |
| ``Riᵗ`` | `Riᵗ` | `turbulent_richardson_number(sf, αᴺ, αˢ)` | Turbulent Richardson number of `TurbulentRichardsonStabilityFunction`, a smooth blend of ``Riᵗ^± = α_N / (ℂᴿⁱ^± α_S + ℂᵅ)`` by the sign of ``α_N``: ``Ri / ℂᴿⁱ^±`` under strong shear, ``α_N / ℂᵅ`` without shear |
| ``ℂᴿⁱ⁻, ℂᴿⁱ⁺, ℂᵅ`` | `ℂᴿⁱ⁻`, `ℂᴿⁱ⁺`, `ℂᵅ` | `ℂᴿⁱ⁻`, `ℂᴿⁱ⁺`, `ℂᵅ` | Richardson numbers at which ``Riᵗ = ∓1`` under strong shear (unstable, stable side) and stratification number at which ``Riᵗ = 1`` without shear |
| ``v`` | `v` | | ``Riᵗ / \sqrt{1 + Riᵗ²} ∈ (-1, 1)``, the argument of the quadratic in ``\ln 𝕊`` of `TurbulentRichardsonStabilityFunction` |
| ``N²ᵈ, N²ˢ`` | `N²ᵈ`, `N²ˢ` | | Dry and saturated squared buoyancy frequencies, which `SaturationWeightedStaticStability` blends with the weight ``aᴺ`` |
| ``aᴺ`` | `aᴺ` | `saturated_stability_weightᶜᶜᶠ` | Effective saturated fraction: the weight of ``N²ˢ`` in `SaturationWeightedStaticStability`, ``N² = (1 - aᴺ) N²ᵈ + aᴺ N²ˢ``, from its `fraction`; for `CondensateFluxFraction`, ``aᴺ = w′qᶜ′ / w′s′`` of the subgrid distribution |
| ``a^{\uparrow}, a^{\downarrow}`` | `a↑`, `a↓` | | Weights of the updraft and downdraft components of the `DoubleGaussian` subgrid distribution, ``a↓ = 1 - a↑`` |
| ``aᶜ`` | `aᶜ` | | Cloud fraction (cloudy volume fraction) of the subgrid distribution |
| ``a⁺`` | `a⁺` | | Supersaturated fraction of the subgrid distribution |
| ``aʷ`` | `aʷ` | | Flux-weighted saturated fraction: the share of the flux of the saturation excess carried by saturated air |
| ``N²ᵇ, N²ᶜ`` | `N²ᵇ`, `N²ᶜ` | | Stratification of a displacement without phase change, with the condensate ``qᶜ`` of the microphysics (``N²ᵇ = Σ b_χ ∂_z χ``) and of the conserved variables ``θˡⁱ``, ``qᵗ`` (``N²ᶜ``) |
| ``q̄ˡ`` | `q̄ˡ` | | Mean liquid of the subgrid distribution; `MeanLiquidStratification` reads ``N² = N²ᵇ + (b_qˡ - b_qᵛ) ∂_z (q̄ˡ - qᶜ)`` |
| ``b_χ`` | `b_θ`, `b_qᵛ`, `b_qˡ`, `b_qⁱ` | | Derivatives ``∂b/∂χ`` of the buoyancy with respect to the mixed variables ``χ ∈ \{θˡⁱ, qᵛ, qˡ, qⁱ\}``, from the model's thermodynamics |
| ``S``                               | `S`    |                                     | Vertical shear magnitude, ``S² = (∂_z u)² + (∂_z v)²``, s⁻¹                     |
| ``N^2``                             | `N²`   |                                     | Squared Brunt–Väisälä frequency, s⁻². (Bare ``N`` is the acoustic substep count, below) |
| ``h^{bl}``                          | `hᵇˡ`  |                                     | Boundary-layer depth, m; a diagnostic (the height of the capping inversion in convective conditions) |
| ``T_0``                             | `T₀`   |                                     | Sea surface temperature                                                        |
| ``qᵛ₀``                             | `qᵛ₀`  |                                     | Saturation specific humidity at sea surface                                    |
| ``\mathscr{I}``                     | `ℐ`    |                                     | Radiative flux (intensity), W/m²                                               |
| ``F_{\mathscr{I}}``                 | `Fℐ`   |                                     | Radiative flux divergence (heating rate), K/s                                  |
| ``τˡʷ``                             | `τˡʷ`  |                                     | Atmosphere optical thickness for longwave                                      |
| ``τˢʷ``                             | `τˢʷ`  |                                     | Atmosphere optical thickness for shortwave                                     |
| ``N_A``                             | `ℕᴬ`   |                                     | Avogadro's number, molecules per mole                                          |
| ``\mathcal{U}``                     | `𝒰`    |                                     | Thermodynamic state struct (e.g., `StaticEnergyState`)                         |
| ``\mathcal{M}``                     | `ℳ`    |                                     | Microphysical state struct (e.g., `WarmPhaseOneMomentState`)                   |
| ``\mathcal{S}``                     | `𝒮`    |                                     | Subgrid statistics of a cell that the turbulence closure supplies to the microphysics (`DiracStatistics`; `ClosureCloudStatistics` is retired from P3). The subgrid distribution of the vapor excess is instead `supersaturation_statistics` |
| ``θ^L``                             | `θᴸ`   | `AcousticSubstepper.linearization_potential_temperature` | Potential temperature of the linearization state, ``θ^L = (ρθ)^L / ρ^L``. The cached field holds ``θ^L + Λ^H``, what the loop's flux of its pressure carries |
| ``C^L``                             | `Cᴸ`   | `AcousticSubstepper.linearization_pressure_coefficient` | Linearized pressure coefficient, the slope ``∂p/∂(ρθ)`` of the equation of state at the linearization state (the stage-entry state under `AcousticRungeKutta3`, the outer-step-start state under `MultirateInfinitesimalStep`), ``C^L = γ^e p^L / (ρθ)^L`` |
| ``ζ``                               | `ζ`    | `adjusted_partition_response`       | Ratio of the slope ``∂p/∂(ρθ)`` of the saturation-adjusted pressure to the slope at the held partition, at fixed densities |
| ``γ^e``                             | `γᵉ`   |                                     | Effective heat-capacity ratio of the linearized equation of state, ``γ^e = (θ / T) ∂T/∂θ`` |
| ``θ ∂T/∂θ``                         | `θ∂T∂θ` | `temperature_and_sensitivity(𝒰, constants)` | Temperature sensitivity at fixed density and composition, propagated through the selected inversion |
| ``\tilde{T}``                       | `T̃`    |                                     | Condensate-free root of the density-based inversion of ``θˡⁱ`` (the root with ``H^c = 0``) |
| ``G^n``                             | `Gⁿ`   |                                     | Tendency fields at time step ``n``                                             |
| ``G^s``                             | `Gˢ`   |                                     | Slow tendencies (excludes fast pressure gradient and buoyancy)                 |
| ``N``                               | `N`    | `AcousticSubstepper.substeps`       | Acoustic substeps per outer time step, or adaptive substep count before stage partitioning |
| ``N_τ``                             | `Nτ`   |                                     | Acoustic substeps in one stage of the outer scheme                               |
| ``\Delta \tau``                     | `Δτ`   |                                     | Acoustic substep size, ``Δτ = Δt / N`` for proportional substep distribution   |
| ``ρ'``                              | `ρ′`   | `AcousticSubstepper.density_perturbation` | Density perturbation about the stage-entry state                       |
| ``(ρθ)'``                           | `ρθ′`  | `AcousticSubstepper.density_potential_temperature_perturbation` | Thermodynamic-density perturbation about the stage-entry state; with moisture the field holds ``(ρθ)' + W``, the loop's pressure in units of ``C^L`` |
| ``(ρu)', (ρv)', (ρw)'``             | `ρu′, ρv′, ρw′` | `AcousticSubstepper.momentum_perturbation` | Momentum perturbations about the stage-entry state              |
| ``D_τ``                             | `Dτ`   |                                     | Klemp-Skamarock-Ha divergence-damping proxy, ``D_τ = ((ρθ)'_τ - (ρθ)'_{τ-\Delta τ}) / θ^L`` |
| ``\bar{u}, \bar{v}, \bar{w}``       | `ū, v̄, w̄` | `AcousticSubstepper.time_averaged_velocities` | Time-averaged velocities for non-acoustic scalar advection |
| ``Λ^H``                             | `Λᴴ`   | `AcousticSubstepper.fixed_composition_sensitivity` | Fixed-composition sensitivity of the pressure to the dry density, ``Λ^H = θ H^c / (cᵖᵐ T - H^c)``, so that ``∂p/∂ρᵈ = C^L Λ^H`` at fixed composition and thermodynamic density; zero without condensate |
| ``Λ_a``                             | `λᵛ, λˡ, λⁱ` | `moisture_pressure_sensitivities`   | Thermodynamic density that a unit of the moisture density ``ρq_a`` holds at fixed pressure (``a`` is vapor, liquid or ice); ``Λ^d = Λ^H + Σ_a Λ_a ρq_a / ρᵈ`` is the same for the dry density |
| ``W``                               | `W`    |                                     | Moisture content of the pressure the acoustic loop integrates, ``W = - Σ_a Λ_a (ρq_a)' + Λ^d ρᵈ'``, so that ``p' = C^L ((ρθ)' + W)``; `W⁰` is the content of a stage's start state relative to its entry state. (``𝕎`` is a terminal velocity; ``W_{ij}`` below is unrelated) |
| ``G^W``                             | `Gᵂ`   |                                     | Slow tendency of ``W``, ``G^W = - Σ_a Λ_a G^s_{ρq_a} + (Σ_a Λ_a ρq_a / ρᵈ) G^t_{ρᵈ} + Λ^H G^s_{ρᵈ}``, added to the thermodynamic tendency that forces the loop's pressure |
| ``G^t_{ρᵈ}``                        | `Gᵗρᵈ` | `AcousticSubstepper.transport_density_tendency` | Transport density tendency: the dry air's share of the divergence of the mass flux the moisture tendencies were built with |
| ``ς``                               | `ς`    |                                     | Scale of the slow path of the loop's pressure where saturation adjustment adjusts the moisture partition, ``ς = ζ \bar{C} / C^L``, with ``\bar{C}`` the slope ``∂p/∂(ρθ)`` at the held partition; one elsewhere. `AcousticSubstepper.scaled_thermodynamic_tendency` holds ``ς (G_{ρᵡ} + G^W)`` |
| ``U^n``                             | `Uⁿ`   | `MultirateInfinitesimalStep.Uⁿ`     | Prognostic state at the start of the outer time step |
| ``U^0``                             | `U⁰`   | `AcousticRungeKutta3.U⁰`            | State that a stage's acoustic substep loop starts from: ``U^n`` in every Wicker–Skamarock stage, ``U^0 = U^n + Z_i`` in stage ``i`` of a multirate infinitesimal step (where it is not stored: `MultirateInfinitesimalStep.U⁰` holds what the start states are formed from) |
| ``Y_i``                             | `Yᵢ`   |                                     | State at the end of stage ``i`` of a multirate infinitesimal step; ``Y_0 = U^n`` and ``Y_s = U^{n+1}`` for ``s`` stages |
| ``ΔY_i``                            | `ΔY`   |                                     | Stage increment, ``ΔY_i = Y_i - U^n`` |
| ``S_i``                             | `Sᵢ`   |                                     | Slow tendency that stage ``i`` adds, evaluated at ``Y_{i-1}``. (Bare ``S`` is the vertical shear magnitude) |
| ``α_{ij}, β_{ij}, γ_{ij}``          | `α, β, γ` | `MultirateInfinitesimalStepTableau.α`, `.β`, `.γ` | Lower-triangular coefficients of a multirate infinitesimal step: stage ``i`` starts from ``U^n + Σ_j α_{ij} ΔY_j`` and is forced with ``(1/d_i) [Σ_j β_{ij} S_j + Σ_j γ_{ij} ΔY_j / Δt]``. (Bare ``α`` is the specific volume; ``β₁, β₂, β₃`` are the stage fractions of `AcousticRungeKutta3`) |
| ``d_i``                             | `d`    | `MultirateInfinitesimalStepTableau.d` | Stage interval fraction, ``d_i = Σ_j β_{ij}``: the acoustic loop of stage ``i`` covers ``d_i Δt``. ``Σ_i d_i`` is the length of acoustic integration per outer step, in units of ``Δt`` |
| ``c_i``                             | `c`    | `MultirateInfinitesimalStepTableau.c` | Stage abscissa: stage ``i`` ends at ``t^n + c_i Δt``, with ``c_i = d_i + Σ_j (α_{ij} + γ_{ij}) c_j`` |
| ``Z_i``                             | `Zᵢ`   | `MultirateInfinitesimalStep.accumulators` | Offset of the start state of stage ``i`` from ``U^n``, ``Z_i = Σ_j α_{ij} ΔY_j`` |
| ``C``                               | `C`    | `MultirateInfinitesimalStepTableau.strong_stability_coefficient` | Strong-stability coefficient of the slow method of a multirate infinitesimal step (its radius of absolute monotonicity): for ``C > 0`` every stage of a slow variable is a convex combination of forward-Euler steps no longer than ``Δt / C`` |
| ``Δt / C``                          |        | `stage_time_step(timestepper, clock)` | Forward-Euler interval of a multirate infinitesimal step, over which microphysical sinks are budgeted and the scalar mass-flux correction is limited; ``Δt`` where ``C = 0`` |
| ``F_i``                             | `Fᵢ`   |                                     | Mass flux of the acoustic loop of stage ``i`` (the stage-entry total density times the loop's transport velocity), which moves the dry air over ``d_i Δt``. In the stage-assembly kernels `Fᵢ` is instead the forcing accumulated from earlier stages, ``(1/d_i) Σ_{j<i} (β_{ij} S_j + γ_{ij} ΔY_j / Δt)`` |
| ``W_{ij}``                          | `W`    | `ScalarMassFluxCorrection.weights`  | Weights of the scalar mass-flux correction, ``W = β^{-1} \mathrm{diag}(d) - I`` |
| ``R_i``                             | `Rᵢ`   | `ScalarMassFluxCorrection.mass_fluxes` | Correction mass flux of stage ``i``, ``R_i = Σ_{k ≤ i} W_{ik} F_k``, for which the recombined scalar fluxes carry the stage's own mass flux: ``Σ_j β_{ij} (F_j + R_j) = d_i F_i`` |
| ``φ``                               | `φ`    | `ScalarMassFluxCorrection.limiter`  | Fraction of a cell's corrective outflow ``R^+`` that the cell can supply over a forward-Euler step, ``0 ≤ φ ≤ 1``; one where the mass fraction is uniform |
