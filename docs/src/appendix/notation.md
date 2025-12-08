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
* `RS` stands for [`ReferenceState`](@ref Breeze.AtmosphereModels.ReferenceState)
* Note that there are independent concepts of "reference". For example, [`AnelasticFormulation`](@ref) involves
  a "reference state", which is an adiabatic, hydrostatic solution to the equations of motion. But there is also an
  "energy reference temperature" and "reference latent heat", which are thermodynamic constants required to define
  the internal energy of moist atmospheric constituents.
* Mapping to AM fields: `ρe` corresponds to `energy_density(model)`, `ρqᵗ` to `model.moisture_density`, and `qᵗ` to `model.specific_moisture`.

The following table also uses a few conventions that suffuse the source code and which are internalized by wise developers:

* `constants` refers to an instance of `ThermodynamicConstants()`
* `q` refers to an instance of  [`MoistureMassFractions`](@ref Breeze.Thermodynamics.MoistureMassFractions)
* "Reference" quantities use a subscript ``r`` (e.g., ``p_r``, ``\rho_r``).
* Phase or mixture identifiers (``d``, ``v``, ``m``) appear as superscripts (e.g., ``Rᵈ``, ``cᵖᵐ``), matching usage in the codebase (e.g., `Rᵈ`, `cᵖᵐ`).
* Conservative variables are stored in ρᵣ-weighted form in the code (e.g., `ρu`, `ρv`, `ρw`, `ρe`, `ρqᵗ`).

| math symbol                           | code          | property name                      | description |
| --- | --- | --- | --- |
| ``\rho``                              | `ρ`           | `AM.density`                       | Density, ``ρ = pᵣ / Rᵐ T`` for anelastic |
| ``\alpha``                            | `α`           |                                    | Specific volume, ``α = 1/ρ``|
| ``\boldsymbol{u} = (u,v,w)``          | `u, v, w`     | `AM.velocities`                    | Velocity components in (x, y, z) or (east, north, up) |
| ``\boldsymbol{ρu} = (ρu, ρv, ρw )``   | `ρu, ρv, ρw`  | `AM.momentum`                      | Momentum components |
| ``ρ e``                               | `ρe`          | `AM.energy_density`                | Energy density |
| ``T``                                 | `T`           | `AM.temperature`                   | Temperature |
| ``p``                                 | `p`           | `AM.pressure`                      | Pressure |
| ``b``                                 | `b`           |                                    | Buoyancy |
| ``ρ qᵗ``                              | `ρqᵗ`         | `AM.moisture_density`              | Total moisture density |
| ``qᵗ``                                | `qᵗ`          | `AM.specific_moisture`             | Total specific moisture (the sum of vapor, liquid, and ice mass fractions) |
| ``qᵛ``                                | `qᵛ`          | `AM.microphysical_fields.qᵛ`       | Vapor mass fraction, a.k.a "specific humidity" |
| ``qˡ``                                | `qˡ`          | `AM.microphysical_fields.qˡ`       | Liquid mass fraction |
| ``qⁱ``                                | `qⁱ`          | `AM.microphysical_fields.qⁱ`       | Ice mass fraction |
| ``qᶜⁱ``                               | `qᶜˡ`         | `AM.microphysical_fields.qᶜˡ`      | Cloud liquid mass fraction |
| ``q^{ci}``                            | `qᶜⁱ`         | `AM.microphysical_fields.qᶜⁱ`      | Cloud ice mass fraction |
| ``q^{r}``                             | `qʳ`          |                                    | Rain mass fraction |
| ``q^{s}``                             | `qˢ`          |                                    | Snow mass fraction |
| ``ρq^{v}``                            | `ρqᵛ`         |                                    | Vapor density |
| ``ρqˡ``                               | `ρqˡ`         |                                    | Liquid density |
| ``ρqⁱ``                               | `ρqⁱ`         |                                    | Ice density |
| ``ρqᶜˡ``                              | `ρqᶜˡ`        |                                    | Cloud liquid density |
| ``ρqᶜⁱ``                              | `ρqᶜⁱ`        |                                    | Cloud ice density |
| ``ρqʳ``                               | `ρqʳ`         |  `AM.microphysical_fields.ρqʳ`     | Rain density |
| ``ρqˢ``                               | `ρqˢ`         |  `AM.microphysical_fields.ρqˢ`     | Snow density |
| ``qᵛ⁺``                               | `qᵛ⁺`         |                                    | Saturation specific humidity over a surface |
| ``qᵛ⁺ˡ``                              | `qᵛ⁺ˡ`        |                                    | Saturation specific humidity over a planar liquid surface |
| ``qᵛ⁺ⁱ``                              | `qᵛ⁺ⁱ`        |                                    | Saturation specific humidity over a planar ice surface |
| ``g``                                 | `g`           | `TC.gravitational_acceleration`    | Gravitational acceleration |
| ``\mathcal{R}``                       | `ℛ`           | `TC.molar_gas_constant`            | Universal (molar) gas constant |
| ``Tᵗʳ``                               | `Tᵗʳ`         | `TC.triple_point_temperature`      | Temperature at the vapor-liquid-ice triple point |
| ``pᵗʳ``                               | `pᵗʳ`         | `TC.triple_point_pressure`         | Pressure at the vapor-liquid-ice triple point |
| ``mᵈ``                                | `mᵈ`          | `TC.dry_air.molar_mass`            | Molar mass of dry air |
| ``mᵛ``                                | `mᵛ`          | `TC.vapor.molar_mass`              | Molar mass of vapor |
| ``Rᵈ``                                | `Rᵈ`          | `dry_air_gas_constant(constants)`     | Dry air gas constant (``Rᵈ = \mathcal{R} / mᵈ``) |
| ``Rᵛ``                                | `Rᵛ`          | `vapor_gas_constant(constants)`       | Water vapor gas constant (``Rᵛ = \mathcal{R} / mᵛ``) |
| ``Rᵐ``                                | `Rᵐ`          | `mixture_gas_constant(q, constants)`  | Mixture gas constant, function of ``q`` |
| ``cᵖᵈ``                               | `cᵖᵈ`         | `TC.dry_air.heat_capacity`         | Heat capacity of dry air at constant pressure |
| ``cᵖᵛ``                               | `cᵖᵛ`         | `TC.vapor.heat_capacity`           | Heat capacity of vapor at constant pressure |
| ``cˡ``                                | `cˡ`          | `TC.liquid.heat_capacity`          | Heat capacity of the liquid phase (incompressible) |
| ``cⁱ``                                | `cⁱ`          | `TC.ice.heat_capacity`             | Heat capacity of the ice phase (incompressible) |
| ``cᵖᵐ``                               | `cᵖᵐ`         | `mixture_heat_capacity(q, constants)` | Mixture heat capacity at constant pressure |
| ``Tᵣ``                                | `Tᵣ`          | `TC.energy_reference_temperature`  | Reference temperature for internal energy relations and latent heat |
| ``\mathcal{L}^l_r``                   | `ℒˡᵣ`         | `TC.liquid.reference_latent_heat`  | Latent heat of condensation at the energy reference temperature |
| ``\mathcal{L}^i_r``                   | `ℒⁱᵣ`         | `TC.ice.reference_latent_heat`     | Latent heat of deposition at the energy reference temperature |
| ``\mathcal{L}^l(T)``                  | `ℒˡ`          | `liquid_latent_heat(T, constants)` | Temperature-dependent latent heat of condensation |
| ``\mathcal{L}^i(T)``                  | `ℒⁱ`          | `ice_latent_heat(T, constants)`    | Temperature-dependent latent heat of deposition |
| ``θ₀``                                | `θ₀`          | `RS.potential_temperature`         | (Constant) reference potential temperature for the anelastic formulation |
| ``p₀``                                | `p₀`          | `RS.base_pressure`                 | Base (surface) reference pressure |
| ``ρᵣ``                                | `ρᵣ`          | `RS.density`                       | Density of a dry reference state for the anelastic formulation |
| ``αᵣ``                                | `αᵣ`          |                                    | Specific volume of a dry reference state, ``αᵣ = Rᵈ θ₀ / pᵣ`` |
| ``p_r``                               | `pᵣ`          | `RS.pressure`                      | Pressure of a dry adiabatic reference pressure for the anelastic formulation |
| ``\Pi``                               | `Π`           |                                    | Exner function, ``Π = (pᵣ / p₀)^{Rᵐ / cᵖᵐ}`` |
| ``θᵈ``                                | `θᵈ`          |                                    | Dry potential temperature |
| ``θᵛ``                                | `θᵛ`          |                                    | Virtual potential temperature |
| ``θᵉ``                                | `θᵉ`          |                                    | Equivalent potential temperature |
| ``θˡⁱ``                               | `θˡⁱ`         |                                    | Liquid-ice potential temperature |
| ``θ``                                 | `θ`           |                                    | Shorthand for liquid-ice potential temperature (used in `set!`) |
| ``\Delta t``                          | `Δt`          | `Simulation.Δt`                    | Time step |
| ``\boldsymbol{\tau}``                 | `τ`           |                                    | Kinematic subgrid/viscous stress tensor (per unit mass) |
| ``\boldsymbol{\mathcal{T}}``          | `𝒯`           |                                   | Dynamic stress tensor used in anelastic momentum, ``\mathcal{T} = ρᵣ τ`` |
| ``\boldsymbol{J}``                    | `J`           |                                    | Dynamic diffusive flux for scalars |
