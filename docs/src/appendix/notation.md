# Notation and conventions

This appendix establishes a common notation across the documentation and source code.
Each entry lists a mathematical symbol and the Unicode form commonly used in
the codebase, along with a common "property name", and a description.
The property names may take a verbose "English form" or concise "mathematical form" corresponding
to the given Unicode symbol. As properties, mathematical names are usually used
mathematical form is invoked for the elements of a `NamedTuple`.
Mathematical symbols are shown with inline math, while the Unicode column shows the exact glyphs used in code.

A few notes about the following table:

* `TC` stands for [`ThermodynamicConstants`](@ref)
* `AM` stands for [`AtmosphereModel`](@ref)
* `RS` stands for [`ReferenceState`](@ref Breeze.Thermodynamics.ReferenceState)
* Note that there are independent concepts of "reference". For example, [`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics) involves
  a "reference state", which is an adiabatic, hydrostatic solution to the equations of motion. But there is also an
  "energy reference temperature" and "reference latent heat", which are thermodynamic constants required to define
  the internal energy of moist atmospheric constituents.
* Mapping to AM fields: `ρe` corresponds to `energy_density(model)`, and the moisture density is accessed via `model.moisture_density`.

The following table also uses a few conventions that suffuse the source code and which are internalized by wise developers:

* `constants` refers to an instance of `ThermodynamicConstants()`
* `q` refers to an instance of  [`MoistureMassFractions`](@ref Breeze.Thermodynamics.MoistureMassFractions)
* "Reference" quantities use a subscript ``r`` (e.g., ``p_r``, ``\rho_r``).
* Phase or mixture identifiers (``d``, ``v``, ``m``) appear as superscripts (e.g., ``Rᵈ``, ``cᵖᵐ``), matching usage in the codebase (e.g., `Rᵈ`, `cᵖᵐ`).
* Conservative variables are stored in ρᵣ-weighted form in the code (e.g., `ρu`, `ρv`, `ρw`, `ρe`, `ρqᵉ` or `ρqᵛ`).

| math symbol                         | code   | property name                       | description                                                                    |
| ----------------------------------- | ------ | ----------------------------------- | ------------------------------------------------------------------------------ |
| ``\rho``                            | `ρ`    | `AM.density`                        | Density, ``ρ = pᵣ / Rᵐ T`` for anelastic                                       |
| ``\alpha``                          | `α`    |                                     | Specific volume, ``α = 1/ρ``                                                   |
| ``\boldsymbol{u} = (u,v,w)``        | `u, v, w` | `AM.velocities`                  | Velocity components in (x, y, z) or (east, north, up)                          |
| ``\boldsymbol{ρu} = (ρu, ρv, ρw)``  | `ρu, ρv, ρw` | `AM.momentum`                 | Momentum components                                                            |
| ``ρ e``                             | `ρe`   | `AM.energy_density`                 | Energy density                                                                 |
| ``T``                               | `T`    | `AM.temperature`                    | Temperature                                                                    |
| ``T⁺``                              | `T⁺`   | `DewpointTemperature(model)`        | Dewpoint temperature                                                           |
| ``p``                               | `p`    | `AM.pressure`                       | Pressure                                                                       |
| ``b``                               | `b`    |                                     | Buoyancy                                                                       |
| ``ρ qᵉ``                            | `ρqᵉ`  | `AM.moisture_density`               | Equilibrium moisture density (saturation adjustment schemes)                   |
| ``ρ qᵛ``                            | `ρqᵛ`  | `AM.moisture_density`               | Vapor density (non-equilibrium schemes)                                        |
| ``qᵛ``                              | `qᵛ`   | `AM.microphysical_fields.qᵛ`        | Vapor mass fraction, a.k.a "specific humidity"                                 |
| ``qˡ``                              | `qˡ`   | `AM.microphysical_fields.qˡ`        | Liquid mass fraction                                                           |
| ``qⁱ``                              | `qⁱ`   | `AM.microphysical_fields.qⁱ`        | Ice mass fraction                                                              |
| ``qᶜˡ``                             | `qᶜˡ`  | `AM.microphysical_fields.qᶜˡ`       | Cloud liquid mass fraction                                                     |
| ``qᶜⁱ``                             | `qᶜⁱ`  | `AM.microphysical_fields.qᶜⁱ`       | Cloud ice mass fraction                                                        |
| ``qʳ``                              | `qʳ`   |                                     | Rain mass fraction                                                             |
| ``qˢ``                              | `qˢ`   |                                     | Snow mass fraction                                                             |
| ``ρqᵛ``                             | `ρqᵛ`  |                                     | Vapor density                                                                  |
| ``ρqˡ``                             | `ρqˡ`  |                                     | Liquid density                                                                 |
| ``ρqⁱ``                             | `ρqⁱ`  |                                     | Ice density                                                                    |
| ``ρqᶜˡ``                            | `ρqᶜˡ` |                                     | Cloud liquid density                                                           |
| ``ρqᶜⁱ``                            | `ρqᶜⁱ` |                                     | Cloud ice density                                                              |
| ``ρqʳ``                             | `ρqʳ`  | `AM.microphysical_fields.ρqʳ`       | Rain density                                                                   |
| ``ρqˢ``                             | `ρqˢ`  | `AM.microphysical_fields.ρqˢ`       | Snow density                                                                   |
| ``n^{cl}``                          | `nᶜˡ`  | `AM.microphysical_fields.nᶜˡ`       | Cloud droplet number per unit mass (1/kg)                                      |
| ``n^r``                             | `nʳ`   | `AM.microphysical_fields.nʳ`        | Rain drop number per unit mass (1/kg)                                          |
| ``n^a``                             | `nᵃ`   | `AM.microphysical_fields.nᵃ`        | Aerosol number per unit mass (1/kg)                                            |
| ``\rho n^{cl}``                     | `ρnᶜˡ` | `AM.microphysical_fields.ρnᶜˡ`      | Cloud droplet number density (1/m³), prognostic                                |
| ``\rho n^r``                        | `ρnʳ`  | `AM.microphysical_fields.ρnʳ`       | Rain drop number density (1/m³), prognostic                                    |
| ``\rho n^a``                        | `ρnᵃ`  | `AM.microphysical_fields.ρnᵃ`       | Aerosol number density (1/m³), prognostic                                      |
| ``N^{cl}``                          | `Nᶜˡ`  |                                     | Volumetric cloud droplet number density, ``Nᶜˡ = ρ nᶜˡ`` (1/m³)               |
| ``N^r``                             | `Nʳ`   |                                     | Volumetric rain drop number density, ``Nʳ = ρ nʳ`` (1/m³)                      |
| ``N^a``                             | `Nᵃ`   |                                     | Volumetric aerosol number density, ``Nᵃ = ρ nᵃ`` (1/m³)                        |
| ``\mathbb{W}^{cl}``                 | `𝕎ᶜˡ`  |                                     | Terminal velocity of cloud liquid (scalar, positive downward)                  |
| ``\mathbb{W}^{ci}``                 | `𝕎ᶜⁱ`  |                                     | Terminal velocity of cloud ice (scalar, positive downward)                     |
| ``\mathbb{W}^r``                    | `𝕎ʳ`   |                                     | Terminal velocity of rain (scalar, positive downward)                          |
| ``\mathbb{W}^s``                    | `𝕎ˢ`   |                                     | Terminal velocity of snow (scalar, positive downward)                          |
| ``qᵛ⁺``                             | `qᵛ⁺`  |                                     | Saturation specific humidity over a surface                                    |
| ``qᵛ⁺ˡ``                            | `qᵛ⁺ˡ` |                                     | Saturation specific humidity over a planar liquid surface                      |
| ``qᵛ⁺ⁱ``                            | `qᵛ⁺ⁱ` |                                     | Saturation specific humidity over a planar ice surface                         |
| ``pᵛ``                              | `pᵛ`   |                                     | Vapor pressure (partial pressure of water vapor), ``pᵛ = ρ qᵛ Rᵛ T``           |
| ``pᵛ⁺``                             | `pᵛ⁺`  |                                     | Saturation vapor pressure                                                      |
| ``\mathscr{H}``                     | `ℋ`    | `RelativeHumidity(model)`           | Relative humidity, ``ℋ = pᵛ / pᵛ⁺``                                            |
| ``\mathscr{S}``                     | `𝒮`    | `supersaturation(T, ρ, q, c, surf)` | Supersaturation, ``𝒮 = pᵛ / pᵛ⁺ - 1``                                          |
| ``g``                               | `g`    | `TC.gravitational_acceleration`     | Gravitational acceleration                                                     |
| ``\mathbb{C}^{ac}``                 | `ℂᵃᶜ`  |                                     | Acoustic sound speed, ``ℂᵃᶜ = \sqrt{γ Rᵈ T}``                                  |
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
| ``Tᵣ``                              | `Tᵣ`   | `TC.energy_reference_temperature`   | Reference temperature for internal energy relations and latent heat            |
| ``\mathcal{L}^l_r``                 | `ℒˡᵣ`  | `TC.liquid.reference_latent_heat`   | Latent heat of condensation at the energy reference temperature                |
| ``\mathcal{L}^i_r``                 | `ℒⁱᵣ`  | `TC.ice.reference_latent_heat`      | Latent heat of deposition at the energy reference temperature                  |
| ``\mathcal{L}^l(T)``                | `ℒˡ`   | `liquid_latent_heat(T, constants)`  | Temperature-dependent latent heat of condensation                              |
| ``\mathcal{L}^i(T)``                | `ℒⁱ`   | `ice_latent_heat(T, constants)`     | Temperature-dependent latent heat of deposition                                |
| ``θ₀``                              | `θ₀`   | `RS.potential_temperature`          | (Constant) reference potential temperature for the anelastic formulation       |
| ``p₀``                              | `p₀`   | `RS.surface_pressure`               | Surface reference pressure                                              |
| ``p^{st}``                          | `pˢᵗ`  | `RS.standard_pressure`              | Standard pressure for potential temperature (default 10⁵ Pa)                   |
| ``ρᵣ``                              | `ρᵣ`   | `RS.density`                        | Density of a dry reference state for the anelastic formulation                 |
| ``αᵣ``                              | `αᵣ`   |                                     | Specific volume of a dry reference state, ``αᵣ = Rᵈ θ₀ / pᵣ``                  |
| ``pᵣ``                              | `pᵣ`   | `RS.pressure`                       | Pressure of a dry adiabatic reference pressure for the anelastic formulation   |
| ``\Pi``                             | `Π`    |                                     | Exner function, ``Π = (pᵣ / pˢᵗ)^{Rᵐ / cᵖᵐ}``                                  |
| ``θᵛ``                              | `θᵛ`   |                                     | Virtual potential temperature                                                  |
| ``θᵉ``                              | `θᵉ`   |                                     | Equivalent potential temperature                                               |
| ``θˡⁱ``                             | `θˡⁱ`  |                                     | Liquid-ice potential temperature                                               |
| ``θᵇ``                              | `θᵇ`   |                                     | Stability-equivalent potential temperature (for moist Brunt-Väisälä)           |
| ``θ``                               | `θ`    |                                     | Shorthand for liquid-ice potential temperature (used in [`set!`](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Fields.set!)) |
| ``\Delta t``                        | `Δt`   | `Simulation.Δt`                     | Time step.                                                                     |
| ``\boldsymbol{\tau}``               | `τ`    |                                     | Kinematic subgrid/viscous stress tensor (per unit mass)                        |
| ``\boldsymbol{\mathcal{T}}``        | `𝒯`    |                                     | Dynamic stress tensor used in anelastic momentum, ``\mathcal{T} = ρᵣ τ``       |
| ``\boldsymbol{J}``                  | `J`    |                                     | Dynamic diffusive flux for scalars                                             |
| ``τˣ``                              | `τˣ`   |                                     | Surface momentum flux (``x``-component), N/m²                                  |
| ``τʸ``                              | `τʸ`   |                                     | Surface momentum flux (``y``-component), N/m²                                  |
| ``\mathcal{Q}^T``                   | `𝒬ᵀ`   |                                     | Surface sensible heat flux, ``\mathcal{Q}^T = cᵖᵐ Jᵀ``                         |
| ``\mathcal{Q}^v``                   | `𝒬ᵛ`   |                                     | Surface latent heat flux, ``\mathcal{Q}^v = \mathcal{L}^l Jᵛ``                 |
| ``Jᵀ``                              | `Jᵀ`   |                                     | Surface temperature flux, kg K/m²s                                             |
| ``Jᵛ``                              | `Jᵛ`   |                                     | Surface moisture flux, kg/m²s                                                  |
| ``Cᴰ``                              | `Cᴰ`   |                                     | Surface drag coefficient                                                       |
| ``Cᵀ``                              | `Cᵀ`   |                                     | Surface sensible heat transfer coefficient (Stanton number)                    |
| ``Cᵛ``                              | `Cᵛ`   |                                     | Surface vapor transfer coefficient (Dalton number)                             |
| ``\ell``                            | `ℓ`    |                                     | Surface roughness length, m                                                    |
| ``T_0``                             | `T₀`   |                                     | Sea surface temperature                                                        |
| ``qᵛ₀``                             | `qᵛ₀`  |                                     | Saturation specific humidity at sea surface                                    |
| ``\mathscr{I}``                     | `ℐ`    |                                     | Radiative flux (intensity), W/m²                                               |
| ``F_{\mathscr{I}}``                 | `Fℐ`   |                                     | Radiative flux divergence (heating rate), K/s                                  |
| ``τˡʷ``                             | `τˡʷ`  |                                     | Atmosphere optical thickness for longwave                                      |
| ``τˢʷ``                             | `τˢʷ`  |                                     | Atmosphere optical thickness for shortwave                                     |
| ``N_A``                             | `ℕᴬ`   |                                     | Avogadro's number, molecules per mole                                          |
| ``\mathcal{U}``                     | `𝒰`    |                                     | Thermodynamic state struct (e.g., `StaticEnergyState`)                         |
| ``\mathcal{M}``                     | `ℳ`    |                                     | Microphysical state struct (e.g., `WarmPhaseOneMomentState`)                   |
| ``\psi``                            | `ψ`    | `AcousticSubstepper.pressure_coefficient` | Pressure coefficient for acoustic substepping, ``ψ = Rᵐ T``             |
| ``{\mathbb{C}^{ac}}^2``             | `ℂᵃᶜ²` |                                           | Acoustic sound speed squared, ``ℂᵃᶜ² = γᵐ ψ = γᵐ Rᵐ T``                 |
| ``G^n``                             | `Gⁿ`   |                                     | Tendency fields at time step ``n``                                             |
| ``G^s``                             | `Gˢ`   |                                     | Slow tendencies (excludes fast pressure gradient and buoyancy)                 |
| ``N_s``                             | `Ns`   | `AcousticSubstepper.Ns`             | Number of acoustic substeps per full time step                                 |
| ``\Delta \tau``                     | `Δτ`   |                                     | Acoustic substep time step, ``Δτ = Δt / Ns``                                   |
| ``\kappa^d``                        | `κᵈ`   | `AcousticSubstepper.κᵈ`             | Divergence damping coefficient for acoustic substepping                        |
| ``\rho_r``                          | `ρᵣ`   | `AcousticSubstepper.ρᵣ`             | Reference density for divergence damping (start of acoustic loop)              |
| ``w^{avg}``                         | `averaging_weight` |                              | Time-averaging weight for velocity fields in acoustic substepping              |
| ``\bar{u}, \bar{v}, \bar{w}``       | `ū, v̄, w̄` |                                 | Time-averaged velocities for scalar advection                                  |
