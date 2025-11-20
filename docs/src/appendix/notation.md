# Notation

This appendix establishes a common notation across the documentation and source.
Each entry lists a mathematical symbol, the Unicode form commonly used in
the codebase, and a brief description. Mathematical symbols are shown with
inline math, while the Unicode column shows the exact glyphs used in code.

| mathematical symbol | unicode | description |
| --- | --- | --- |
| ``\rho``                      | `ρ`           | Density |
| ``\boldsymbol{u} = (u,v,w)``  | `u, v, w`     | Velocity components (x, y, z) |
| ``ρ \, u,\; ρ \,v,\; ρ \, w`` | `ρu, ρv, ρw`  | Momentum components |
| ``T``                         | `T`           | Temperature |
| ``\alpha``                    | `α`           | Specific volume, ``α = 1/ρ`` or ``α = R^{m} T / p_r`` in anelastic |
| ``ρ e``                       | `ρe`          | Moist static energy density (conservative variable) |
| ``e``                         | `e`           | Moist static energy per unit mass, ``e = c^{pd} \, θ`` |
| ``ρ q^{t}``                   | `ρqᵗ`         | Total moisture density (conservative water variable) |
| ``q^{t}``                     | `qᵗ`          | Total moisture mass fraction (vapor + liquid + ice) |
| ``q^{v}``                     | `qᵛ`          | Vapor specific humidity |
| ``q^{\ell}``                  | `qˡ`          | Total liquid mass fraction |
| ``q^{i}``                     | `qⁱ`          | Total ice mass fraction |
| ``q^{cl}``                    | `qᶜˡ`         | Cloud liquid mass fraction |
| ``q^{ci}``                    | `qᶜⁱ`         | Cloud ice mass fraction |
| ``q^{r}``                     | `qʳ`          | Rain mass fraction |
| ``q^{s}``                     | `qˢ`          | Snow mass fraction |
| ``q^{v+}``                    | `qᵛ⁺`         | Saturation specific humidity over a surface |
| ``q^{v+l}``                   | `qᵛ⁺ˡ`        | Saturation specific humidity over a planar liquid surface |
| ``q^{v+i}``                   | `qᵛ⁺ⁱ`        | Saturation specific humidity over a planar ice surface |
| ``g``                         | `g`           | Gravitational acceleration |
| ``\mathcal{R}``               | `ℛ`           | Universal (molar) gas constant |
| ``R^{d}``                     | `Rᵈ`          | Dry air gas constant |
| ``R^{v}``                     | `Rᵛ`          | Water vapor gas constant |
| ``R^{m}``                     | `Rᵐ`          | Mixture gas constant, function of ``q`` |
| ``c^{pd}``                    | `cᵖᵈ`         | Heat capacity of dry air at constant pressure |
| ``c^{pv}``                    | `cᵖᵛ`         | Heat capacity of vapor at constant pressure |
| ``c^l``                       | `cˡ`          | Heat capacity of the liquid phase (incompressible) |
| ``c^i``                       | `cⁱ`          | Heat capacity of the ice phase (incompressible) |
| ``c^{pm}``                    | `cᵖᵐ`         | Mixture heat capacity at constant pressure |
| ``\theta``                    | `θ`           | Potential temperature |
| ``\theta_0``                  | `θ₀`          | Reference potential temperature (constant) |
| ``\Pi``                       | `Π`           | Exner function, ``\Pi = (p_r/p_0)^{R^{m}/c^{pm}}`` |
| ``p``                         | `p`           | Pressure |
| ``p_0``                       | `p₀`          | Base (surface) reference pressure |
| ``\rho_r(z)``                 | `ρᵣ`          | Density of a dry reference state at height ``z`` |
| ``\alpha_{r}``                | `αᵣ`          | Specific volume of a dry reference state, ``α_{r} = R^d θ_0 / p_r`` |
| ``p_r(z)``                    | `pᵣ`          | Hydrostatic reference pressure at height ``z`` |
| ``\mathcal{L}^{l}``           | `ℒˡ`          | Latent heat of vaporization |
| ``b``                         | `b`           | Buoyancy |
| ``p_h'``                      | `pₕ′`         | Hydrostatic pressure anomaly, ``∂_z p_h' = - ρ_r b`` |
| ``p_n``                       | `pₙ`          | Nonhydrostatic pressure (projection/correction potential) |
| ``\Delta t``                  | `Δt`          | Time step |
| ``\Delta z``                  | `Δz`          | Vertical grid spacing |
| ``\boldsymbol{\tau}``         | `τ`           | Kinematic subgrid/viscous stress tensor (per unit mass) |
| ``\boldsymbol{\mathcal{T}}``  | `𝒯`           | Dynamic stress tensor used in anelastic momentum, ``\mathcal{T} = ρᵣ \, \tau`` |
| ``\boldsymbol{J}``            | `J`           | Dynamic diffusive flux for scalars |

Notes:
- Reference-state quantities use a subscript ``r`` (e.g., ``p_r``, ``\rho_r``), following the Thermodynamics docs and code.
- Phase or mixture identifiers (``d``, ``v``, ``m``) appear as superscripts (e.g., ``R^{d}``, ``c^{pm}``), matching usage in the codebase (e.g., `Rᵈ`, `cᵖᵐ`).
- Conservative variables are stored in ρᵣ-weighted form in the code (e.g., `ρu`, `ρv`, `ρw`, `ρe`, `ρqᵗ`).
- Mapping to AtmosphereModel fields: `ρe` corresponds to `model.energy_density`, `ρqᵗ` to `model.moisture_density`, and `qᵗ` to `model.specific_moisture`.
