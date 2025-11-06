# Notation

This appendix establishes a common notation across the documentation and source.
Each entry lists a mathematical symbol, the Unicode form commonly used in
the codebase, and a brief description. Mathematical symbols are shown with
inline math, while the Unicode column shows the exact glyphs used in code.

| mathematical symbol | unicode | description |
| --- | --- | --- |
| ``p`` | `p` | Pressure |
| ``p_r(z)`` | `pᵣ` | Hydrostatic reference pressure at height ``z`` |
| ``p_0`` | `p₀` | Base (surface) reference pressure |
| ``p_h'`` | `pₕ′` | Hydrostatic pressure anomaly, ``∂_z p_h' = - ρ_r b`` |
| ``p_n`` | `pₙ` | Nonhydrostatic pressure (projection/correction potential) |
| ``\rho`` | `ρ` | Density |
| ``\rho_r(z)`` | `ρᵣ` | Reference density at height ``z`` |
| ``\boldsymbol{u} = (u,v,w)`` | `u, v, w` | Velocity components (x, y, z) |
| ``\rho_r\,u,\; \rho_r\,v,\; \rho_r\,w`` | `ρu, ρv, ρw` | Momentum components (ρᵣ-weighted velocities) |
| ``g`` | `g` | Gravitational acceleration |
| ``T`` | `T` | Temperature |
| ``\theta`` | `θ` | Potential temperature |
| ``\theta_r`` | `θᵣ` | Reference potential temperature (constant) |
| ``\alpha`` | `α` | Specific volume, ``α = 1/ρ`` or ``α = R^{m} T / p_r`` in anelastic |
| ``\alpha_{r}`` | `αᵣ` | Reference specific volume, ``α_{r} = R^{d} θ_r / p_r`` |
| ``\Pi`` | `Π` | Exner function, ``\Pi = (p_r/p_0)^{R^{m}/c^{pm}}`` |
| ``R^{d}`` | `Rᵈ` | Dry air gas constant |
| ``R^{v}`` | `Rᵛ` | Water vapor gas constant |
| ``R^{m}`` | `Rᵐ` | Mixture gas constant, function of ``q`` |
| ``c^{pd}`` | `cᵖᵈ` | Heat capacity of dry air at constant pressure |
| ``c^{pv}`` | `cᵖᵛ` | Heat capacity of vapor at constant pressure |
| ``c^{pm}`` | `cᵖᵐ` | Mixture heat capacity at constant pressure |
| ``q`` | `q` | Specific humidity (vapor mass fraction) |
| ``q^{t}`` | `qᵗ` | Total specific humidity (vapor + condensates) |
| ``q^{v}`` | `qᵛ` | Water vapor specific humidity |
| ``q^{\ell}`` | `qˡ` | Liquid water specific humidity |
| ``q^{i}`` | `qi` | Ice specific humidity |
| ``q^{v+}`` | `qᵛ⁺` | Saturation specific humidity over liquid/ice (context-dependent) |
| ``\mathcal{L}^{v}`` | `ℒᵛ` | Latent heat of vaporization |
| ``\mathcal{R}`` | `ℛ` | Universal (molar) gas constant |
| ``b`` | `b` | Buoyancy, ``b = g (α - α_{r}) / α_{r}`` |
| ``\Delta t`` | `Δt` | Time step |
| ``\Delta z`` | `Δz` | Vertical grid spacing |
 

Notes:
- Reference-state quantities use a subscript ``r`` (e.g., ``p_r``, ``\rho_r``), following the Thermodynamics docs and code.
- Phase or mixture identifiers (``d``, ``v``, ``m``) appear as superscripts (e.g., ``R^{d}``, ``c^{pm}``), matching usage in the codebase (e.g., `Rᵈ`, `cᵖᵐ`).
- Conservative variables are typically stored in ρᵣ-weighted form in the code (e.g., `ρu`, `ρv`, `ρw`, `ρe`, `ρqᵗ`).
