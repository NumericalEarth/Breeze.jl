# Notation

This appendix establishes a common notation across the documentation and source.
Each entry lists a mathematical symbol, the Unicode form commonly used in
the codebase, and a brief description. Mathematical symbols are shown with
inline math, while the Unicode column shows the exact glyphs used in code.

Guide:
* `TC ≡ ThermodynamicConstants`
* `AM ≡ AtmosphereModel`
* `thermo = ThermodynamicConstants()`
* `q isa MoistureMassFractions`

| mathematical symbol           | unicode       | property name                     | description |
| ---                           | ---           | ---                               | --- |
| ``\rho``                      | `ρ`           | `AM.density`                      | Density, ``ρ = p_r / R^m T`` for anelastic |
| ``\boldsymbol{u} = (u,v,w)``  | `u, v, w`     | `AM.velocities`                   | Velocity components (x, y, z) or (east, north, up) |
| ``ρ \, u,\; ρ \,v,\; ρ \, w`` | `ρu, ρv, ρw`  | `AM.momentum`                     | Momentum components |
| ``T``                         | `T`           | `AM.temperature`                  | Temperature |
| ``\alpha``                    | `α`           |                                   | Specific volume, ``α = 1/ρ`` or ``α = R^{m} T / p_r`` in anelastic |
| ``ρ e``                       | `ρe`          | `AM.energy_density`               | Moist static energy density (conservative variable) |
| ``e``                         | `e`           | `AM.specific_energy`              | Moist static energy per unit mass, ``e = c^{pd} \, θ`` |
| ``ρ q^{t}``                   | `ρqᵗ`         | `AM.moisture_density`             | Total moisture density (conservative water variable) |
| ``q^{t}``                     | `qᵗ`          | `AM.specific_moisture`            | Total specific moisture (the sum of vapor, liquid, and ice mass fractions) |
| ``q^{v}``                     | `qᵛ`          | `AM.microphysical_fields.qᵛ`      | Vapor mass fraction, a.k.a "specific humidity" |
| ``q^{\ell}``                  | `qˡ`          | `AM.microphysical_fields.qˡ`      | Liquid mass fraction |
| ``q^{i}``                     | `qⁱ`          | `AM.microphysical_fields.qⁱ`      | Ice mass fraction |
| ``q^{cl}``                    | `qᶜˡ`         | `AM.microphysical_fields.qᶜˡ`     | Cloud liquid mass fraction |
| ``q^{ci}``                    | `qᶜⁱ`         | `AM.microphysical_fields.qᶜⁱ`     | Cloud ice mass fraction |
| ``q^{r}``                     | `qʳ`          | `AM.microphysical_fields.qʳ`      | Rain mass fraction |
| ``q^{s}``                     | `qˢ`          | `AM.microphysical_fields.qʳ`      | Snow mass fraction |
| ``ρq^{v}``                    | `ρqᵛ`         |                                   | Vapor density |
| ``ρq^{\ell}``                 | `ρqˡ`         |                                   | Liquid density |
| ``ρq^{i}``                    | `ρqⁱ`         |                                   | Ice density |
| ``ρq^{cl}``                   | `ρqᶜˡ`        |                                   | Cloud liquid density |
| ``ρq^{ci}``                   | `ρqᶜⁱ`        |                                   | Cloud ice density |
| ``ρq^{r}``                    | `ρqʳ`         |                                   | Rain density |
| ``ρq^{s}``                    | `ρqˢ`         |                                   | Snow density |
| ``q^{v+}``                    | `qᵛ⁺`         |                                   | Saturation specific humidity over a surface |
| ``q^{v+}``                    | `qᵛ⁺`         |                                   | Saturation specific humidity over a surface |
| ``q^{v+l}``                   | `qᵛ⁺ˡ`        |                                   | Saturation specific humidity over a planar liquid surface |
| ``q^{v+i}``                   | `qᵛ⁺ⁱ`        |                                   | Saturation specific humidity over a planar ice surface |
| ``g``                         | `g`           | `TC.gravitational_acceleration`   | Gravitational acceleration |
| ``\mathcal{R}``               | `ℛ`           | `TC.molar_gas_constant`           | Universal (molar) gas constant |
| ``T^{tr}``                    | `Tᵗʳ`         | `TC.triple_point_temperature`     | Temperature at the vapor-liquid-ice triple point |
| ``p^{tr}``                    | `pᵗʳ`         | `TC.triple_point_pressure`        | Pressure at the vapor-liquid-ice triple point |
| ``m^d``                       | `mᵈ`          | `TC.dry_air.molar_mass`           | Molar mass of dry air |
| ``m^v``                       | `mᵛ`          | `TC.vapor.molar_mass`             | Molar mass of vapor |
| ``R^{d}``                     | `Rᵈ`          | `dry_air_gas_constant(thermo)`    | Dry air gas constant (``R^d = \mathcal{R} / m^d``) |
| ``R^{v}``                     | `Rᵛ`          | `vapor_gas_constant(thermo)`      | Water vapor gas constant (``R^v = \mathcal{R} / m^v``) |
| ``R^{m}``                     | `Rᵐ`          | `mixture_gas_constant(q, thermo)` | Mixture gas constant, function of ``q`` |
| ``c^{pd}``                    | `cᵖᵈ`         | `TC.dry_air.heat_capacity`        | Heat capacity of dry air at constant pressure |
| ``c^{pv}``                    | `cᵖᵛ`         | `TC.vapor.heat_capacity`          | Heat capacity of vapor at constant pressure |
| ``c^l``                       | `cˡ`          | `TC.liquid.heat_capacity`         | Heat capacity of the liquid phase (incompressible) |
| ``c^i``                       | `cⁱ`          | `TC.ice.heat_capacity`            | Heat capacity of the ice phase (incompressible) |
| ``c^{pm}``                    | `cᵖᵐ`         | `mixture_heat_capacity(q, thermo)`| Mixture heat capacity at constant pressure |
| ``\theta``                    | `θ`           |                                   | Potential temperature |
| ``\theta_0``                  | `θ₀`          |                                   | Reference potential temperature (constant) |
| ``\Pi``                       | `Π`           |                                   | Exner function, ``\Pi = (p_r/p_0)^{R^{m}/c^{pm}}`` |
| ``p``                         | `p`           |                                   | Pressure |
| ``p_0``                       | `p₀`          | `ReferenceState.base_pressure`    | Base (surface) reference pressure |
| ``\rho_r(z)``                 | `ρᵣ`          | `ReferenceState.density`          | Density of a dry reference state at height ``z`` |
| ``\alpha_{r}``                | `αᵣ`          |                                   | Specific volume of a dry reference state, ``α_{r} = R^d θ_0 / p_r`` |
| ``p_r(z)``                    | `pᵣ`          | `ReferenceState.pressure`         | Hydrostatic reference pressure at height ``z`` |
| ``\mathcal{L}^{l}``           | `ℒˡ`          |                                   | Latent heat of vaporization |
| ``b``                         | `b`           |                                   | Buoyancy |
| ``p_n``                       | `pₙ`          | `AM.nonhydrostatic_pressure`      | Nonhydrostatic pressure (projection/correction potential) |
| ``\Delta t``                  | `Δt`          | `Simulation.Δt`                   | Time step |
| ``\boldsymbol{\tau}``         | `τ`           |                                   | Kinematic subgrid/viscous stress tensor (per unit mass) |
| ``\boldsymbol{\mathcal{T}}``  | `𝒯`           |                                   | Dynamic stress tensor used in anelastic momentum, ``\mathcal{T} = ρᵣ \, \tau`` |
| ``\boldsymbol{J}``            | `J`           |                                   | Dynamic diffusive flux for scalars |

Notes:
- Reference-state quantities use a subscript ``r`` (e.g., ``p_r``, ``\rho_r``), following the Thermodynamics docs and code.
- Phase or mixture identifiers (``d``, ``v``, ``m``) appear as superscripts (e.g., ``R^{d}``, ``c^{pm}``), matching usage in the codebase (e.g., `Rᵈ`, `cᵖᵐ`).
- Conservative variables are stored in ρᵣ-weighted form in the code (e.g., `ρu`, `ρv`, `ρw`, `ρe`, `ρqᵗ`).
- Mapping to AM fields: `ρe` corresponds to `model.energy_density`, `ρqᵗ` to `model.moisture_density`, and `qᵗ` to `model.specific_moisture`.
