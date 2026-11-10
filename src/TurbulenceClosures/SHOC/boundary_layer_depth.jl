#####
##### Surface layer and boundary layer depth (shoc_diag_obklen, pblintd)
#####

"""
$(TYPEDEF)

The boundary layer depth of Holtslag and Boville (1993), `pblintd`
of E3SM `shoc.F90`: the height at which a bulk Richardson number reaches the critical value `ℂᴴ₁`.
The depth is a diagnostic; it feeds back to the state only in the cold-surface fallback of the
diffusivities.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂᴴ₁` | `ricr`  | 0.3 | critical bulk Richardson number of the boundary layer top |
| `ℂᴴ₂` | `fac`   | 100 | friction velocity factor in the squared wind of the bulk Richardson number |
| `ℂᴴ₃` | `fak`   | 8.5 | surface temperature excess of convective thermals |
| `ℂᴴ₄` | `betam` | 15  | wind-gradient constant of the temperature excess (enters as `ℂᴴ₄ × 0.1`, with `sffrac = 0.1`) |

$(TYPEDFIELDS)
"""
struct HoltslagBovilleDepth{FT} <: AbstractSHOCComponent
    "Critical bulk Richardson number (`ricr`)"
    ℂᴴ₁ :: FT
    "Friction-velocity factor in the bulk Richardson number (`fac`)"
    ℂᴴ₂ :: FT
    "Surface temperature excess constant (`fak`)"
    ℂᴴ₃ :: FT
    "Wind-gradient constant of the surface temperature excess (`betam`)"
    ℂᴴ₄ :: FT
    "Minimum surface friction velocity, m s⁻¹"
    ustar_min :: FT
    "Lowest pressure at which the boundary layer top may be diagnosed, Pa"
    pblmaxp :: FT
    "Lower bound on the squared wind in the bulk Richardson number, m² s⁻²"
    tinyw :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `HoltslagBovilleDepth{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword
argument.
"""
HoltslagBovilleDepth(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(HoltslagBovilleDepth, FT,
                            (ℂᴴ₁ = 0.3, ℂᴴ₂ = 100, ℂᴴ₃ = 8.5, ℂᴴ₄ = 15, ustar_min = 0.01, pblmaxp = 4e4, tinyw = 1e-36), kw)

Adapt.@adapt_structure HoltslagBovilleDepth

# Virtual potential temperature as pblintd_init_pot computes it
@inline function pbl_thv(i, j, k, c, thetal, qw, shoc_ql)
    @inbounds begin
        ql = shoc_ql[i, j, k]
        q = qw[i, j, k] - ql
        th = thetal[i, j, k] + (c.lcond / c.cp) * ql
    end
    return th * (1 + c.eps * q - ql)
end

# pblintd_height: the upward scan for the first level whose bulk Richardson number reaches
# `ricr`, as a masked sweep over all levels. `rino` of the level below is carried as a scalar.
@inline function pbl_height_scan(i, j, Nz, bl, c, zt, pres, thetal, qw, shoc_ql, u_wind, v_wind,
                                 ustar, thv_ref, pblh, check)
    thv_1 = pbl_thv(i, j, 1, c, thetal, qw, shoc_ql)
    rino_below = zero(pblh)
    # The highest level at which the boundary layer top may be diagnosed
    z_pbl_max = @inbounds zt[i, j, 1]

    @inbounds for k in 2:Nz
        in_pbl = pres[i, j, k] >= bl.pblmaxp
        z_pbl_max = ifelse(in_pbl, zt[i, j, k], z_pbl_max)
        Δu = u_wind[i, j, k] - u_wind[i, j, 1]
        Δv = v_wind[i, j, k] - v_wind[i, j, 1]
        vvk = Δu * Δu + Δv * Δv + bl.ℂᴴ₂ * (ustar * ustar)
        vvk = max(vvk, bl.tinyw)
        thv_k = pbl_thv(i, j, k, c, thetal, qw, shoc_ql)
        rino = c.ggr * (thv_k - thv_ref) * (zt[i, j, k] - zt[i, j, 1]) / (thv_1 * vvk)

        cross = check & in_pbl & (rino >= bl.ℂᴴ₁)
        Δrino = ifelse(cross, rino - rino_below, one(rino))
        pblh_k = zt[i, j, k-1] + (bl.ℂᴴ₁ - rino_below) / Δrino * (zt[i, j, k] - zt[i, j, k-1])
        pblh = ifelse(cross, pblh_k, pblh)
        check = check & !cross
        rino_below = rino
    end

    return pblh, check, z_pbl_max
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

`shoc_diag_obklen` followed by `pblintd`: return the planetary boundary layer depth of column
`(i, j)` following Holtslag and Boville (1993). The Fortran `npbl` (the number of levels below
`pblmaxp`) is replaced by the per-column mask `pres ≥ pblmaxp`. None of the bounds here goes
through the clipping policy: the depth is the outcome of a discrete search.
"""
@inline function pbl_height(i, j, Nz, bl::HoltslagBovilleDepth, c, zt, zi, pres, thetal, qw, shoc_ql, shoc_cldfrac,
                            u_wind, v_wind, uw_sfc, vw_sfc, wthl_sfc, wqw_sfc)

    FT = typeof(uw_sfc)
    fak = bl.ℂᴴ₃                               # constant in surface temperature excess (fak)
    binm = bl.ℂᴴ₄ * convert(FT, 0.1)           # betam * sffrac

    # shoc_diag_obklen
    @inbounds begin
        cldliq_sfc = shoc_ql[i, j, 1]
        qv_sfc = qw[i, j, 1] - cldliq_sfc
        th_sfc = thetal[i, j, 1] + (c.lcond / c.cp) * cldliq_sfc
    end
    thv_sfc = th_sfc * (1 + c.eps * qv_sfc - cldliq_sfc)
    ustar = max(zero_safe_sqrt(uw_sfc * uw_sfc + vw_sfc * vw_sfc), bl.ustar_min)
    kbfs = wthl_sfc + c.eps * th_sfc * wqw_sfc
    # sign(1e-10, kbfs), in the form of the C++ (differs from the Fortran only for kbfs = -0.0)
    kbfs_offset = ifelse(kbfs >= 0, convert(FT, 1e-10), -convert(FT, 1e-10))
    obklen = -thv_sfc * (ustar * ustar * ustar) / (c.ggr * c.vk * (kbfs + kbfs_offset))

    # pblintd_init, pblintd_height
    thv_1 = pbl_thv(i, j, 1, c, thetal, qw, shoc_ql)
    pblh = @inbounds zt[i, j, 1]
    pblh, check, z_pbl_max = pbl_height_scan(i, j, Nz, bl, c, zt, pres, thetal, qw, shoc_ql, u_wind, v_wind,
                                             ustar, thv_1, pblh, true)

    # pblintd_surf_temp: an effective surface temperature that accounts for surface fluctuations
    pblh = ifelse(check, z_pbl_max, pblh)
    check = kbfs > 0
    phiminv = cbrt(ifelse(check, 1 - binm * pblh / obklen, one(pblh)))
    tlv = ifelse(check, thv_1 + kbfs * fak / (ustar * phiminv), thv_1)

    # Improve the estimate for unstable conditions with the convective temperature excess
    pblh, check, _ = pbl_height_scan(i, j, Nz, bl, c, zt, pres, thetal, qw, shoc_ql, u_wind, v_wind,
                                     ustar, tlv, pblh, check)

    # pblintd_check_pblh
    pblh = ifelse(check, z_pbl_max, pblh)
    pblh = max(pblh, 700 * ustar)

    # pblintd_cldcheck (as in the Fortran, true for any non-negative cloud fraction)
    @inbounds pblh = ifelse(shoc_cldfrac[i, j, 1] >= 0, max(pblh, zi[i, j, 2] + 50), pblh)

    return pblh
end
