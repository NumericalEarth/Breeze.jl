#####
##### Surface layer and boundary layer depth (shoc_diag_obklen, pblintd)
#####

# Virtual potential temperature as pblintd_init_pot computes it
@inline function pbl_thv(i, j, k, p, thetal, qw, shoc_ql)
    @inbounds begin
        ql = shoc_ql[i, j, k]
        q = qw[i, j, k] - ql
        th = thetal[i, j, k] + (p.lcond / p.cp) * ql
    end
    return th * (1 + p.eps * q - ql)
end

# pblintd_height: the upward scan for the first level whose bulk Richardson number reaches
# `ricr`, as a masked sweep over all levels. `rino` of the level below is carried as a scalar.
@inline function pbl_height_scan(i, j, Nz, p, zt, pres, thetal, qw, shoc_ql, u_wind, v_wind,
                                 ustar, thv_ref, pblh, check)
    thv_1 = pbl_thv(i, j, 1, p, thetal, qw, shoc_ql)
    rino_below = zero(pblh)
    # The highest level at which the boundary layer top may be diagnosed
    z_pbl_max = @inbounds zt[i, j, 1]

    @inbounds for k in 2:Nz
        in_pbl = pres[i, j, k] >= p.pblmaxp
        z_pbl_max = ifelse(in_pbl, zt[i, j, k], z_pbl_max)
        Δu = u_wind[i, j, k] - u_wind[i, j, 1]
        Δv = v_wind[i, j, k] - v_wind[i, j, 1]
        vvk = Δu * Δu + Δv * Δv + p.ℂᴴ₂ * (ustar * ustar)
        vvk = max(vvk, p.tinyw)
        thv_k = pbl_thv(i, j, k, p, thetal, qw, shoc_ql)
        rino = p.ggr * (thv_k - thv_ref) * (zt[i, j, k] - zt[i, j, 1]) / (thv_1 * vvk)

        cross = check & in_pbl & (rino >= p.ℂᴴ₁)
        Δrino = ifelse(cross, rino - rino_below, one(rino))
        pblh_k = zt[i, j, k-1] + (p.ℂᴴ₁ - rino_below) / Δrino * (zt[i, j, k] - zt[i, j, k-1])
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
`pblmaxp`) is replaced by the per-column mask `pres ≥ pblmaxp`.
"""
@inline function pbl_height(i, j, Nz, p, zt, zi, pres, thetal, qw, shoc_ql, shoc_cldfrac, u_wind, v_wind,
                            uw_sfc, vw_sfc, wthl_sfc, wqw_sfc)

    FT = typeof(uw_sfc)
    fak = p.ℂᴴ₃                               # constant in surface temperature excess (fak)
    binm = p.ℂᴴ₄ * convert(FT, 0.1)           # betam * sffrac

    # shoc_diag_obklen
    @inbounds begin
        cldliq_sfc = shoc_ql[i, j, 1]
        qv_sfc = qw[i, j, 1] - cldliq_sfc
        th_sfc = thetal[i, j, 1] + (p.lcond / p.cp) * cldliq_sfc
    end
    thv_sfc = th_sfc * (1 + p.eps * qv_sfc - cldliq_sfc)
    ustar = max(zero_safe_sqrt(uw_sfc * uw_sfc + vw_sfc * vw_sfc), p.ustar_min)
    kbfs = wthl_sfc + p.eps * th_sfc * wqw_sfc
    # sign(1e-10, kbfs), in the form of the C++ (differs from the Fortran only for kbfs = -0.0)
    kbfs_offset = ifelse(kbfs >= 0, convert(FT, 1e-10), -convert(FT, 1e-10))
    obklen = -thv_sfc * (ustar * ustar * ustar) / (p.ggr * p.vk * (kbfs + kbfs_offset))

    # pblintd_init, pblintd_height
    thv_1 = pbl_thv(i, j, 1, p, thetal, qw, shoc_ql)
    pblh = @inbounds zt[i, j, 1]
    pblh, check, z_pbl_max = pbl_height_scan(i, j, Nz, p, zt, pres, thetal, qw, shoc_ql, u_wind, v_wind,
                                             ustar, thv_1, pblh, true)

    # pblintd_surf_temp: an effective surface temperature that accounts for surface fluctuations
    pblh = ifelse(check, z_pbl_max, pblh)
    check = kbfs > 0
    phiminv = cbrt(ifelse(check, 1 - binm * pblh / obklen, one(pblh)))
    tlv = ifelse(check, thv_1 + kbfs * fak / (ustar * phiminv), thv_1)

    # Improve the estimate for unstable conditions with the convective temperature excess
    pblh, check, _ = pbl_height_scan(i, j, Nz, p, zt, pres, thetal, qw, shoc_ql, u_wind, v_wind,
                                     ustar, tlv, pblh, check)

    # pblintd_check_pblh
    pblh = ifelse(check, z_pbl_max, pblh)
    pblh = max(pblh, 700 * ustar)

    # pblintd_cldcheck (as in the Fortran, true for any non-negative cloud fraction)
    @inbounds pblh = ifelse(shoc_cldfrac[i, j, 1] >= 0, max(pblh, zi[i, j, 2] + 50), pblh)

    return pblh
end
