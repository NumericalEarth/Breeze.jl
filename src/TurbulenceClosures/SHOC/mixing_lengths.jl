#####
##### Slot: mixing length
#####
##### A mixing length component implements
#####
#####     mixing_length_column!(ml, cp, c, i, j, grid, Nz, tke_weighted_height, state, host, diagnostics, scratch)
#####
##### called once per column after the sweep that floors the turbulent kinetic energy and fills
##### `brunt` (it returns whatever per-column value the level function needs), and
#####
#####     mixing_length(ml, cp, c, column, i, j, k, grid, tke_k, brunt_k, zt_k, host_scale, state, scratch)
#####
##### called level by level, which returns the length ℓ of level `k`. `cp` is the clipping policy
##### and `c` the `SHOCConstants`.
#####

"""
$(TYPEDEF)

The mixing length of [Bogenschutz and Krueger (2013)](@cite Bogenschutz2013) as E3SM `shoc.F90`
computes it (`compute_shoc_mix_shoc_length`): a blend of the distance to the surface `κ z`, an
asymptotic length `l∞` (the fraction `ℂˡ₄` of the height weighted by `√e` over the column) and,
in stable air, the buoyancy length `√e / N`,

    ℓ = (ℂˡ₂ / ℂˡ₁) [1 / (ℂˡ₅ √e κ z) + 1 / (ℂˡ₅ √e l∞) + ℂˡ₃ N²⁺ / e]^(-1/2)

bounded by `minlen`, `maxlen` and the horizontal mesh size.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂˡ₁` | `length_fac`     | 0.5    | divides the mixing length |
| `ℂˡ₂` | literal `2.8284` | 2.8284 | prefactor of the mixing length (`2√2`, truncated as in the Fortran) |
| `ℂˡ₃` | literal `0.01`   | 0.01   | weight of the stratification term `N² / e` |
| `ℂˡ₄` | literal `0.1`    | 0.1    | the asymptotic length `l∞` as a fraction of the TKE-weighted mean height |
| `ℂˡ₅` | `tscale`         | 400    | eddy turnover time scale, **in seconds** (the one dimensional coefficient of SHOC) |

$(TYPEDFIELDS)
"""
struct SHOCMixingLength{FT} <: AbstractSHOCComponent
    "Mixing length divisor (`length_fac`)"
    ℂˡ₁ :: FT
    "Mixing length prefactor (literal `2.8284`)"
    ℂˡ₂ :: FT
    "Weight of the stratification term of the mixing length (literal `0.01`)"
    ℂˡ₃ :: FT
    "Asymptotic length as a fraction of the TKE-weighted mean height (literal `0.1`)"
    ℂˡ₄ :: FT
    "Eddy turnover time scale of the mixing length, s (`tscale`)"
    ℂˡ₅ :: FT
    "Maximum mixing length, m"
    maxlen :: FT
    "Minimum mixing length, m"
    minlen :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `SHOCMixingLength{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword
argument.
"""
SHOCMixingLength(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(SHOCMixingLength, FT,
                            (ℂˡ₁ = 0.5, ℂˡ₂ = 2.8284, ℂˡ₃ = 0.01, ℂˡ₄ = 0.1, ℂˡ₅ = 400, maxlen = 20000, minlen = 20), kw)

Adapt.@adapt_structure SHOCMixingLength

# The asymptotic length l∞ of the column
@inline mixing_length_column!(ml::SHOCMixingLength, cp, c, i, j, grid, Nz, tke_weighted_height,
                              state, host, diagnostics, scratch) = ml.ℂˡ₄ * tke_weighted_height

@inline function mixing_length(ml::SHOCMixingLength, cp, c, l_inf, i, j, k, grid, tke_k, brunt_k, zt_k,
                               host_scale, state, scratch)
    FT = typeof(tke_k)
    tscale = ml.ℂˡ₅ # turnover time scale [s]
    brunt_scale = convert(FT, 1e-6) # scale of the squared buoyancy frequency for the clipping policy [s⁻²]

    tkes = sqrt(tke_k)
    brunt2 = clip_negative(cp, brunt_k, brunt_scale)

    mix = clip_min(cp, ml.maxlen, (ml.ℂˡ₂ * sqrt(1 / ((1 / (tscale * tkes * c.vk * zt_k))
                         + (1 / (tscale * tkes * l_inf)) + ml.ℂˡ₃ * (brunt2 / tke_k)))) / ml.ℂˡ₁, ml.minlen)
    mix = clip_min(cp, ml.maxlen, mix, ml.minlen)
    mix = clip_max(cp, ml.minlen, mix, ml.minlen)
    mix = clip_min(cp, host_scale, mix, ml.minlen)

    return mix
end
