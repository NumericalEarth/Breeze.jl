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

Beyond the bound, the third moment is reset to the constant `w3_reset` (0.02 m³ s⁻³), as in
E3SM `shoc.F90` (`clipping_diag_third_shoc_moments`). The third moment jumps where it reaches the
bound.

$(TYPEDFIELDS)
"""
struct ResetBound{FT} <: AbstractSHOCComponent
    "The value a third moment beyond its bound is reset to, m³ s⁻³ (literal `0.02`)"
    w3_reset :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ResetBound{FT}` with the reset value of E3SM `shoc.F90`.
"""
ResetBound(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ResetBound, FT, (; w3_reset = 0.02), kw)

Adapt.@adapt_structure ResetBound

Base.summary(::ResetBound) = "ResetBound"

"""
$(TYPEDEF)

Beyond the bound `b`, the third moment is the bound: `w3 ← clip_between(w3, -b, b)` through the
clipping policy. Continuous, with a kink at the bound under [`HardClipping`](@ref) and smooth
under [`SmoothClipping`](@ref), whose smoothing length is its width times `b`.
"""
struct ClampBound <: AbstractSHOCComponent end

"""
$(TYPEDEF)

The third moment approaches the bound `b` smoothly: `w3 ← b tanh(w3 / b)`. Infinitely
differentiable whatever the clipping policy; a third moment well inside the bound is changed by
the relative amount `(w3 / b)² / 3`.
"""
struct TanhBound <: AbstractSHOCComponent end

"""
$(TYPEDEF)

The diagnostic third moment of vertical velocity of Canuto et al. (2001) as
E3SM `shoc.F90` computes it (`compute_diag_third_shoc_moment`), bounded in magnitude by
`w3clip √(2 w'w'³)` in the manner of `bound`.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂʷ₂`  | `c_diag_3rd_mom` | 7 | the coefficient `c` of the diagnostic third moment |
| `ℂʷ₃`  | literal `0.52` | 0.52 | `a0 = ℂʷ₃ / (c² (c - 2))` |
| `ℂʷ₄`  | literal `0.87` | 0.87 | `a1 = ℂʷ₄ / c²` |
| `ℂʷ₅`  | literal `0.5`  | 0.5  | `a2 = ℂʷ₅ / c` |
| `ℂʷ₆`  | literal `0.6`  | 0.6  | `a3 = ℂʷ₆ / (c (c - 2))` |
| `ℂʷ₇`  | literal `2.4`  | 2.4  | `a4 = ℂʷ₇ / (3 c + 5)` |
| `ℂʷ₈`  | literal `0.6`  | 0.6  | `a5 = ℂʷ₈ / (c (3 + 5 c))` |
| `ℂʷ₉`  | literal `1.2`  | 1.2  | weight of `X₀` and `X₁` in `w'³ = (A₁ - ℂʷ₉ X₁ - ℂʷ₁₀ f₅) / (c - ℂʷ₉ X₀ + A₀)` |
| `ℂʷ₁₀` | literal `1.5`  | 1.5  | weight of `f₅`, see `ℂʷ₉` |
| `ℂʷ₁₁` | literal `5/4`  | 1.25 | weight of `f₄` in `ω₂ = ω₁ f₃ + ℂʷ₁₁ ω₀ f₄` |

`ℂʷ₃` to `ℂʷ₁₁` are the numbers of the closure of Canuto et al. (2001) that `shoc.F90` writes
as literals; `shoc.F90` exposes only `c`. `w3clip` is a bound, not an empirical coefficient.

$(TYPEDFIELDS)
"""
struct CanutoThirdMoment{B, FT} <: AbstractSHOCComponent
    "How the third moment is kept within its bound"
    bound :: B
    "Coefficient of the diagnostic third moment of vertical velocity (`c_diag_3rd_mom`)"
    ℂʷ₂ :: FT
    "Numerator of `a0` of the diagnostic third moment (literal `0.52`)"
    ℂʷ₃ :: FT
    "Numerator of `a1` (literal `0.87`)"
    ℂʷ₄ :: FT
    "Numerator of `a2` (literal `0.5`)"
    ℂʷ₅ :: FT
    "Numerator of `a3` (literal `0.6`)"
    ℂʷ₆ :: FT
    "Numerator of `a4` (literal `2.4`)"
    ℂʷ₇ :: FT
    "Numerator of `a5` (literal `0.6`)"
    ℂʷ₈ :: FT
    "Weight of `X₀` and `X₁` in the third moment (literal `1.2`)"
    ℂʷ₉ :: FT
    "Weight of `f₅` in the third moment (literal `1.5`)"
    ℂʷ₁₀ :: FT
    "Weight of `f₄` in `ω₂` (literal `5/4`)"
    ℂʷ₁₁ :: FT
    "Bound on the third moment over `√(2 w'w'³)`"
    w3clip :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a `CanutoThirdMoment` with the values of E3SM `shoc.F90`, bounded as `bound` prescribes.
Every scalar field is a keyword argument.
"""
function CanutoThirdMoment(FT::DataType = Oceananigans.defaults.FloatType; bound = ResetBound(Float64), kw...)
    defaults = (ℂʷ₂ = 7, ℂʷ₃ = 0.52, ℂʷ₄ = 0.87, ℂʷ₅ = 0.5, ℂʷ₆ = 0.6, ℂʷ₇ = 2.4, ℂʷ₈ = 0.6, ℂʷ₉ = 1.2, ℂʷ₁₀ = 1.5,
                ℂʷ₁₁ = 1.25, w3clip = 1.2)
    names = keys(defaults)
    all(name -> name ∈ names, keys(kw)) ||
        throw(ArgumentError("$(setdiff(keys(kw), names)) are not scalar fields of CanutoThirdMoment; they are $(names)."))
    scalars = map(value -> convert(FT, value), Tuple(merge(defaults, NamedTuple(kw))))
    return CanutoThirdMoment(convert_eltype(FT, bound), scalars...)
end

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
    a4 = tm.ℂʷ₇ / (3 * c + 5)
    a5 = tm.ℂʷ₈ / (c * (3 + 5 * c))
    omega0 = a4 / (1 - a5 * buoy_sgs2)
    omega1 = omega0 / (2 * c)
    omega2 = omega1 * f3 + tm.ℂʷ₁₁ * omega0 * f4

    # x_y_terms_diag_third_shoc_moment
    a0 = (tm.ℂʷ₃ * (1 / (c * c))) / (c - 2)
    a1 = tm.ℂʷ₄ / (c * c)
    a2 = tm.ℂʷ₅ / c
    a3 = tm.ℂʷ₆ / (c * (c - 2))
    x0 = (a2 * buoy_sgs2 * (1 - a3 * buoy_sgs2)) / (1 - (a1 + a3) * buoy_sgs2)
    y0 = (2 * a2 * buoy_sgs2 * x0) / (1 - a3 * buoy_sgs2)
    x1 = (a0 * f0 + a1 * f1 + a2 * (1 - a3 * buoy_sgs2) * f2) / (1 - (a1 + a3) * buoy_sgs2)
    y1 = (2 * a2 * (buoy_sgs2 * x1 + (a0 / a1) * f0 + f1)) / (1 - a3 * buoy_sgs2)

    # aa_terms_diag_third_shoc_moment
    aa0 = omega0 * x0 + omega1 * y0
    aa1 = omega0 * x1 + omega1 * y1 + omega2

    # w3_diag_third_shoc_moment
    return (aa1 - tm.ℂʷ₉ * x1 - tm.ℂʷ₁₀ * f5) / (c - tm.ℂʷ₉ * x0 + aa0)
end

# The largest magnitude of the third moment; `theterm` is w_sec interpolated to the interface
@inline third_moment_bound(tm::CanutoThirdMoment, theterm) = tm.w3clip * sqrt(2 * (theterm * theterm * theterm))

@inline bound_third_moment(tm::CanutoThirdMoment, cp, theterm, w3) =
    bound_third_moment(tm.bound, cp, third_moment_bound(tm, theterm), w3)

# clipping_diag_third_shoc_moments: a value beyond the bound is reset to a constant, as in the Fortran
@inline function bound_third_moment(bound::ResetBound, cp, cond, w3)
    FT = typeof(w3)
    tsign = ifelse(w3 < 0, -one(FT), one(FT))
    return ifelse(tsign * w3 > cond, bound.w3_reset, w3)
end

# The bound is positive: the interface variance it is made of is floored
@inline bound_third_moment(::ClampBound, cp, cond, w3) = clip_between(cp, w3, -cond, cond, cond)
@inline bound_third_moment(::TanhBound, cp, cond, w3) = cond * tanh(w3 / cond)
