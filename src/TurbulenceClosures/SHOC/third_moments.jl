#####
##### Slot: third moment of vertical velocity (compute_diag_third_shoc_moment and its helpers)
#####
##### A third-moment component implements
#####
#####     third_moment(tm, constants, τ_zi, N²_zi, w_sec_zi, thetal_zi, thedz, thedz2,
#####                  Δthl_sec, Δwthl_sec, Δw_sec, Δtke, wthl) -> w3
#####     bound_third_moment(tm, cp, w_sec_zi, w3) -> w3
#####
##### at an interior interface: the diagnosis, and the bound that keeps the skewness realizable.
#####

"""
$(TYPEDEF)

Beyond the bound, the third moment is reset to the constant 0.02 m³ s⁻³, as in E3SM `shoc.F90`
(`clipping_diag_third_shoc_moments`). The third moment jumps where it reaches the bound.
"""
struct ResetBound <: AbstractSHOCComponent end

"""
$(TYPEDEF)

The diagnostic third moment of vertical velocity of Canuto et al. (2001) as
E3SM `shoc.F90` computes it (`compute_diag_third_shoc_moment`), bounded in magnitude by
`w3clip √(2 w'w'³)` in the manner of `bound`.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂʷ₂` | `c_diag_3rd_mom` | 7 | the coefficient `c` of the diagnostic third moment |

$(TYPEDFIELDS)
"""
struct CanutoThirdMoment{B, FT} <: AbstractSHOCComponent
    "How the third moment is kept within its bound"
    bound :: B
    "Coefficient of the diagnostic third moment of vertical velocity (`c_diag_3rd_mom`)"
    ℂʷ₂ :: FT
    "Bound on the third moment over `√(2 w'w'³)`"
    w3clip :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a `CanutoThirdMoment` with the values of E3SM `shoc.F90`, bounded as `bound` prescribes.
"""
CanutoThirdMoment(FT::DataType = Oceananigans.defaults.FloatType; bound = ResetBound(), ℂʷ₂ = 7, w3clip = 1.2) =
    CanutoThirdMoment(bound, convert(FT, ℂʷ₂), convert(FT, w3clip))

Adapt.@adapt_structure CanutoThirdMoment

Base.summary(tm::CanutoThirdMoment{B, FT}) where {B, FT} = string("CanutoThirdMoment{", FT, "} with ", summary(tm.bound))

# The third moment at an interior interface from the interface values of isotropy, Brunt-Vaisala
# frequency, w_sec and thetal, the inverse thicknesses `thedz` and `thedz2`, and the differences
# of the moments across the interface; `wthl` is wthl_sec at the interface.
@inline function third_moment(tm::CanutoThirdMoment, constants, isotropy_zi, brunt_zi, w_sec_zi, thetal_zi, thedz, thedz2,
                              thl_sec_diff, wthl_sec_diff, wsec_diff, tke_diff, wthl)
    FT = typeof(wthl)
    c = tm.ℂʷ₂

    # fterms_input_for_diag_third_shoc_moment
    iso = isotropy_zi
    isosqrd = iso * iso
    buoy_sgs2 = isosqrd * brunt_zi
    bet2 = constants.ggr / thetal_zi

    f0 = thedz2 * (bet2 * bet2 * bet2) * ((iso * iso) * (iso * iso)) * wthl * thl_sec_diff
    f1 = thedz2 * (bet2 * bet2) * (iso * iso * iso) * (wthl * wthl_sec_diff + convert(FT, 0.5) * w_sec_zi * thl_sec_diff)
    f2 = thedz * bet2 * isosqrd * wthl * wsec_diff + 2 * thedz2 * bet2 * isosqrd * w_sec_zi * wthl_sec_diff
    f3 = thedz2 * bet2 * isosqrd * w_sec_zi * wthl_sec_diff + thedz * bet2 * isosqrd * (wthl * tke_diff)
    f4 = thedz * iso * w_sec_zi * (wsec_diff + tke_diff)
    f5 = thedz * iso * w_sec_zi * wsec_diff

    # omega_terms_diag_third_shoc_moment
    a4 = convert(FT, 2.4) / (3 * c + 5)
    a5 = convert(FT, 0.6) / (c * (3 + 5 * c))
    omega0 = a4 / (1 - a5 * buoy_sgs2)
    omega1 = omega0 / (2 * c)
    omega2 = omega1 * f3 + (convert(FT, 5) / 4) * omega0 * f4

    # x_y_terms_diag_third_shoc_moment
    a0 = (convert(FT, 0.52) * (1 / (c * c))) / (c - 2)
    a1 = convert(FT, 0.87) / (c * c)
    a2 = convert(FT, 0.5) / c
    a3 = convert(FT, 0.6) / (c * (c - 2))
    x0 = (a2 * buoy_sgs2 * (1 - a3 * buoy_sgs2)) / (1 - (a1 + a3) * buoy_sgs2)
    y0 = (2 * a2 * buoy_sgs2 * x0) / (1 - a3 * buoy_sgs2)
    x1 = (a0 * f0 + a1 * f1 + a2 * (1 - a3 * buoy_sgs2) * f2) / (1 - (a1 + a3) * buoy_sgs2)
    y1 = (2 * a2 * (buoy_sgs2 * x1 + (a0 / a1) * f0 + f1)) / (1 - a3 * buoy_sgs2)

    # aa_terms_diag_third_shoc_moment
    aa0 = omega0 * x0 + omega1 * y0
    aa1 = omega0 * x1 + omega1 * y1 + omega2

    # w3_diag_third_shoc_moment
    return (aa1 - convert(FT, 1.2) * x1 - convert(FT, 1.5) * f5) / (c - convert(FT, 1.2) * x0 + aa0)
end

# The largest magnitude of the third moment; `theterm` is w_sec interpolated to the interface
@inline third_moment_bound(tm::CanutoThirdMoment, theterm) = tm.w3clip * sqrt(2 * (theterm * theterm * theterm))

@inline bound_third_moment(tm::CanutoThirdMoment, cp, theterm, w3) =
    bound_third_moment(tm.bound, cp, third_moment_bound(tm, theterm), w3)

# clipping_diag_third_shoc_moments: a value beyond the bound is reset to 0.02, as in the Fortran
@inline function bound_third_moment(::ResetBound, cp, cond, w3)
    FT = typeof(w3)
    tsign = ifelse(w3 < 0, -one(FT), one(FT))
    return ifelse(tsign * w3 > cond, convert(FT, 0.02), w3)
end
