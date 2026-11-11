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

A mixing length that blends three lengths by the sum of their inverses: the distance to the
surface `κ z`, an asymptotic length `l∞` (the fraction `ℂˡ₄` of the height weighted by `√e` over
the column) and, in stable air, the buoyancy length `√e / N`,

    ℓ = (ℂˡ₂ / ℂˡ₁) [1 / (ℂˡ₅ √e κ z) + 1 / (ℂˡ₅ √e l∞) + ℂˡ₃ N²⁺ / e]^(-1/2)

bounded by `minlen`, `maxlen` and the horizontal mesh size. Each level is a function of that
level and of one number of the column, `l∞`.

This is the length of [Bogenschutz and Krueger (2013)](@cite Bogenschutz2013) as E3SM `shoc.F90`
computes it (`compute_shoc_mix_shoc_length`, `check_length_scale_shoc_length`), and the mixing
default mixing length of `SimplifiedHigherOrderClosure`.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂˡ₁` | `length_fac`     | 0.5    | divides the mixing length |
| `ℂˡ₂` | literal `2.8284` | 2.8284 | prefactor of the mixing length (`2√2`, truncated as in the Fortran) |
| `ℂˡ₃` | literal `0.01`   | 0.01   | weight of the stratification term `N² / e` |
| `ℂˡ₄` | literal `0.1`    | 0.1    | the asymptotic length `l∞` as a fraction of the TKE-weighted mean height |
| `ℂˡ₅` | `tscale`         | 400    | eddy turnover time scale, **in seconds** (the one dimensional coefficient of SHOC) |

$(TYPEDFIELDS)
"""
struct BlendedMixingLength{FT} <: AbstractSHOCComponent
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

Return `BlendedMixingLength{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword
argument.
"""
BlendedMixingLength(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(BlendedMixingLength, FT,
                            (ℂˡ₁ = 0.5, ℂˡ₂ = 2.8284, ℂˡ₃ = 0.01, ℂˡ₄ = 0.1, ℂˡ₅ = 400, maxlen = 20000, minlen = 20), kw)

Adapt.@adapt_structure BlendedMixingLength

# The asymptotic length l∞ of the column
@inline mixing_length_column!(ml::BlendedMixingLength, cp, c, i, j, grid, Nz, tke_weighted_height,
                              state, host, diagnostics, scratch) = ml.ℂˡ₄ * tke_weighted_height

@inline function mixing_length(ml::BlendedMixingLength, cp, c, l_inf, i, j, k, grid, tke_k, brunt_k, zt_k,
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

#####
##### The gradient-limited mixing length of the TKE closure, reused
#####
#####     ℓ(z) = min(ℂˢ z, min over z′ of [ℓᵇ(z′) + ℂⁱ |z - z′|]),    ℓᵇ = √e / N where N² > 0
#####
##### `GradientLimitedMixingLength` is not a function of one level: it is the lower envelope, over
##### the column, of the buoyancy penetration depth √e / N of every face carried away at the slope
##### ℂⁱ, bounded by the wall length ℂˢ z. Level 1 is the lowest level here as in the grid (SHOC's
##### arrays are not flipped in Breeze; E3SM counts from the top). Breeze's `fill_mixing_length!` computes the envelope of
##### a column in two sweeps and `mixing_lengthᶜᶜᶜ` reconstructs the length of a level from it;
##### both are called here as they are, with
#####
#####   * a `MixingLengthContext` in the place of the TKE closure, which gives them the mixing
#####     length and the floor of `e` they read;
#####   * SHOC's turbulent kinetic energy, with the level above the top set to the top level;
#####   * the stratification N² at the interfaces from SHOC's own virtual potential temperature
#####     (the gradient across each interior interface; zero at the surface and at the top), so that
#####     it is consistent with `brunt` and with the state SHOC has just loaded;
#####   * three scratch fields at interfaces: N², the envelope ℓˢ and the capped length ℓ.
#####
##### The length has no eddy turnover time scale: every coefficient of a SHOC with it is
##### non-dimensional. Its bounds are the hard `min` of the TKE closure in every clipping policy
##### (continuous, with kinks), and it is infinite in the envelope where the air is not stable.
#####

"""
$(TYPEDEF)

What Breeze's mixing length functions read of a closure: the mixing length and the floor of the
turbulent kinetic energy inside the turbulent velocity `√e`.
"""
struct MixingLengthContext{ML, FT}
    mixing_length :: ML
    minimum_tke :: FT
end

Adapt.@adapt_structure MixingLengthContext

component_fields(grid, ::GradientLimitedMixingLength) =
    (gradient_limited_length = ZFaceField(grid),
     stratification_envelope = ZFaceField(grid),
     interface_stratification = ZFaceField(grid))

# The envelope of the column, from the turbulent kinetic energy floored by the sweep before
@inline function mixing_length_column!(ml::GradientLimitedMixingLength, cp, c, i, j, grid, Nz, tke_weighted_height,
                                       state, host, diagnostics, scratch)
    tke, thv, zt = state.tke, host.thv, host.zt_grid
    N² = scratch.interface_stratification

    @inbounds begin
        # The reconstruction of √e at the top face reads the level above the top
        tke[i, j, Nz+1] = tke[i, j, Nz]

        N²[i, j, 1] = 0
        thv_l = thv[i, j, 1]
        zt_l = zt[i, j, 1]
        for k in 2:Nz
            thv_u = thv[i, j, k]
            zt_u = zt[i, j, k]
            N²[i, j, k] = (2 * c.ggr / (thv_u + thv_l)) * (thv_u - thv_l) / (zt_u - zt_l)
            thv_l, zt_l = thv_u, zt_u
        end
        N²[i, j, Nz+1] = 0
    end

    context = MixingLengthContext(ml, c.mintke)
    fill_mixing_length!(scratch.gradient_limited_length, scratch.stratification_envelope, i, j, grid, ml, context, tke, N²)

    return context
end

@inline function mixing_length(ml::GradientLimitedMixingLength, cp, c, context, i, j, k, grid, tke_k, brunt_k, zt_k,
                               host_scale, state, scratch)
    envelope = (ℓˢ = scratch.stratification_envelope, N² = scratch.interface_stratification)
    mix = mixing_lengthᶜᶜᶜ(i, j, k, grid, context, state.tke, envelope)
    return min(host_scale, mix)
end
