#####
##### The SHOC column physics: `shoc_main` of E3SM shoc.F90 (ba53243f) with `nadv = 1`, written
##### for one column `(i, j)` with every vertical loop inside the calling kernel thread.
#####
##### Conventions
#####
#####   * Arrays are indexed `[i, j, k]` with `k = 1` at the BOTTOM (the Oceananigans convention),
#####     whereas shoc.F90 has `k = 1` at the top. Thermodynamic ("zt") levels are `k = 1:Nz` and
#####     interfaces ("zi") are `k = 1:Nz+1`; interface `k` lies between levels `k-1` and `k`, so
#####     the Fortran level `kF` is `k = Nz + 1 - kF` and the Fortran interface `kF` is
#####     `k = Nz + 2 - kF`. Two-dimensional quantities are indexed `[i, j, 1]`.
#####   * The arithmetic (operation order, association, sweep direction of the sums and of the
#####     Thomas algorithm) is that of the Fortran, so that the results match the reference port
#####     `ShocColumn` to round-off.
#####   * Fortran temporaries that are pure functions of stored arrays (layer thicknesses,
#####     density, every interpolated profile, the shear term) are recomputed where they are
#####     used rather than stored; the only scratch is the tridiagonal factorization.
#####   * No allocation, no `Ref`s, `ifelse` in place of branches (both arguments are finite),
#####     fixed trip counts (the upward search of `pblintd_height` is a masked sweep), and guarded
#####     square roots and divisions, so that the column can be differentiated.
#####

#####
##### Small utilities
#####

# √x whose value is exactly that of `sqrt` for x ≥ 0, with a finite (zero) derivative at x = 0.
@inline zero_safe_sqrt(x) = ifelse(x > 0, sqrt(ifelse(x > 0, x, one(x))), zero(x))

# Fortran `if (y < minthresh) y = minthresh`, which keeps a negative zero where `max` would not.
@inline clip_below(y, minthresh) = ifelse(y < minthresh, minthresh, y)

# shoc.F90 `linear_interp` between an upper node (x1u, y1u) and a lower node (x1l, y1l).
@inline linear_interp(x1u, x1l, y1u, y1l, x2) = y1u + (y1l - y1u) * (x2 - x1u) / (x1l - x1u)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Interpolate the level array `y` to interface `k` (linear extrapolation at the top and bottom
interfaces), clipped below at `minthresh`.
"""
@inline function interpolate_to_interface(i, j, k, Nz, zt, zi, y, minthresh)
    ku = clamp(k, 2, Nz)
    kl = ku - 1
    @inbounds y2 = linear_interp(zt[i, j, ku], zt[i, j, kl], y[i, j, ku], y[i, j, kl], zi[i, j, k])
    return clip_below(y2, minthresh)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Interpolate the interface array `y` to level `k`, clipped below at `minthresh`.
"""
@inline function interpolate_to_level(i, j, k, zt, zi, y, minthresh)
    @inbounds y2 = linear_interp(zi[i, j, k+1], zi[i, j, k], y[i, j, k+1], y[i, j, k], zt[i, j, k])
    return clip_below(y2, minthresh)
end

#####
##### shoc_grid: thicknesses and density, recomputed on demand
#####

# Thickness of level k
@inline dz_zt(i, j, k, zi) = @inbounds zi[i, j, k+1] - zi[i, j, k]

# Distance between the levels adjacent to interface k; the height of the lowest level at k = 1
@inline dz_zi(i, j, k, zt) = @inbounds ifelse(k == 1, zt[i, j, 1], zt[i, j, k] - zt[i, j, max(k-1, 1)])

# Air density of level k
@inline rho_zt(i, j, k, p, zi, pdel) = @inbounds (1 / p.ggr) * (pdel[i, j, k] / dz_zt(i, j, k, zi))

# Air density at interface k
@inline function rho_zi(i, j, k, Nz, p, zt, zi, pdel)
    ku = clamp(k, 2, Nz)
    kl = ku - 1
    @inbounds y2 = linear_interp(zt[i, j, ku], zt[i, j, kl],
                                 rho_zt(i, j, ku, p, zi, pdel), rho_zt(i, j, kl, p, zi, pdel), zi[i, j, k])
    return clip_below(y2, zero(y2))
end

# 1 / dp of level k
@inline rdp_zt(i, j, k, p, zi, pdel) = 1 / (p.ggr * rho_zt(i, j, k, p, zi, pdel) * dz_zt(i, j, k, zi))

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

    @inbounds for k in 2:Nz
        in_pbl = pres[i, j, k] >= p.pblmaxp
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

    return pblh, check
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

    # The highest level at which the boundary layer top may be diagnosed
    z_pbl_max = @inbounds zt[i, j, 1]
    @inbounds for k in 2:Nz
        z_pbl_max = ifelse(pres[i, j, k] >= p.pblmaxp, zt[i, j, k], z_pbl_max)
    end

    # pblintd_init, pblintd_height
    thv_1 = pbl_thv(i, j, 1, p, thetal, qw, shoc_ql)
    pblh = @inbounds zt[i, j, 1]
    pblh, check = pbl_height_scan(i, j, Nz, p, zt, pres, thetal, qw, shoc_ql, u_wind, v_wind,
                                  ustar, thv_1, pblh, true)

    # pblintd_surf_temp: an effective surface temperature that accounts for surface fluctuations
    pblh = ifelse(check, z_pbl_max, pblh)
    check = kbfs > 0
    phiminv = cbrt(ifelse(check, 1 - binm * pblh / obklen, one(pblh)))
    tlv = ifelse(check, thv_1 + kbfs * fak / (ustar * phiminv), thv_1)

    # Improve the estimate for unstable conditions with the convective temperature excess
    pblh, check = pbl_height_scan(i, j, Nz, p, zt, pres, thetal, qw, shoc_ql, u_wind, v_wind,
                                  ustar, tlv, pblh, check)

    # pblintd_check_pblh
    pblh = ifelse(check, z_pbl_max, pblh)
    pblh = max(pblh, 700 * ustar)

    # pblintd_cldcheck (as in the Fortran, true for any non-negative cloud fraction)
    @inbounds pblh = ifelse(shoc_cldfrac[i, j, 1] >= 0, max(pblh, zi[i, j, 2] + 50), pblh)

    return pblh
end

#####
##### Shear production (compute_shr_prod), on interfaces and interpolated to levels
#####

@inline function shear_production_term(i, j, k, Nz, p, zt, u_wind, v_wind)
    ku = clamp(k, 2, Nz)
    kl = ku - 1
    grid_dz = 1 / dz_zi(i, j, ku, zt)
    @inbounds begin
        u_grad = grid_dz * (u_wind[i, j, ku] - u_wind[i, j, kl])
        v_grad = grid_dz * (v_wind[i, j, ku] - v_wind[i, j, kl])
    end
    sterm = p.ℂᴾ * (u_grad * u_grad + v_grad * v_grad)
    return ifelse((k == 1) | (k == Nz + 1), zero(sterm), sterm)
end

@inline function shear_production_term_zt(i, j, k, Nz, p, zt, zi, u_wind, v_wind)
    sterm_u = shear_production_term(i, j, k+1, Nz, p, zt, u_wind, v_wind)
    sterm_l = shear_production_term(i, j, k, Nz, p, zt, u_wind, v_wind)
    @inbounds y2 = linear_interp(zi[i, j, k+1], zi[i, j, k], sterm_u, sterm_l, zt[i, j, k])
    return clip_below(y2, zero(y2))
end

#####
##### Implicit diffusion (vd_shoc_decomp, vd_shoc_solve)
#####

# Superdiagonal `du` (coupling to the level below) and Thomas-factorized subdiagonal `dl` and
# diagonal `d` of the diffusion matrix for the level diffusivity `kv`. The factorization sweeps
# from the top of the column down, as in the Fortran.
@inline function vd_shoc_decomp!(i, j, Nz, dtime, p, zt, zi, pdel, kv, flux, du, dl, d)
    FT = typeof(flux)

    @inbounds for k in Nz:-1:2
        kv_zi = interpolate_to_interface(i, j, k, Nz, zt, zi, kv, zero(FT))
        # tmpi = dt * (g * rho)² / dp at the interface, with dp = g * rho * dz
        tmpi = dtime * (p.ggr * rho_zi(i, j, k, Nz, p, zt, zi, pdel)) / dz_zi(i, j, k, zt)
        du[i, j, k]   = -1 * kv_zi * tmpi * rdp_zt(i, j, k, p, zi, pdel)
        dl[i, j, k-1] = -1 * kv_zi * tmpi * rdp_zt(i, j, k-1, p, zi, pdel)
    end

    @inbounds begin
        du[i, j, 1] = 0
        dl[i, j, Nz] = 0
        d[i, j, Nz] = 1 - du[i, j, Nz]
    end

    @inbounds for k in Nz-1:-1:2
        d[i, j, k] = 1 - du[i, j, k] - dl[i, j, k]
        dl[i, j, k] = dl[i, j, k] / d[i, j, k+1]
        d[i, j, k] = d[i, j, k] - dl[i, j, k] * du[i, j, k+1]
    end

    @inbounds begin
        d[i, j, 1] = 1 - dl[i, j, 1] + flux * dtime * p.ggr * rdp_zt(i, j, 1, p, zi, pdel)
        dl[i, j, 1] = dl[i, j, 1] / d[i, j, 2]
        d[i, j, 1] = d[i, j, 1] - dl[i, j, 1] * du[i, j, 2]
    end

    return nothing
end

@inline function vd_shoc_solve!(i, j, Nz, du, dl, d, var)
    @inbounds for k in Nz-1:-1:1
        var[i, j, k] = var[i, j, k] - dl[i, j, k] * var[i, j, k+1]
    end

    @inbounds var[i, j, 1] = var[i, j, 1] / d[i, j, 1]

    @inbounds for k in 2:Nz
        var[i, j, k] = (var[i, j, k] - du[i, j, k] * var[i, j, k-1]) / d[i, j, k]
    end

    return nothing
end

#####
##### Third moment of vertical velocity (compute_diag_third_shoc_moment and its helpers)
#####

@inline function third_moment(i, j, k, Nz, p, zt, zi, thetal, tke, isotropy, brunt, w_sec, thl_sec, wthl_sec)
    FT = eltype(tke)
    c = p.ℂʷ₂
    largeneg = convert(FT, -99999999.99)

    isotropy_zi = interpolate_to_interface(i, j, k, Nz, zt, zi, isotropy, zero(FT))
    brunt_zi = interpolate_to_interface(i, j, k, Nz, zt, zi, brunt, largeneg)
    w_sec_zi = interpolate_to_interface(i, j, k, Nz, zt, zi, w_sec, (convert(FT, 2) / 3) * p.mintke)
    thetal_zi = interpolate_to_interface(i, j, k, Nz, zt, zi, thetal, zero(FT))

    # fterms_input_for_diag_third_shoc_moment
    thedz = 1 / dz_zi(i, j, k, zt)
    thedz2 = 1 / (dz_zt(i, j, k-1, zi) + dz_zt(i, j, k, zi))
    iso = isotropy_zi
    isosqrd = iso * iso
    buoy_sgs2 = isosqrd * brunt_zi
    bet2 = p.ggr / thetal_zi

    # f0_to_f5_diag_third_shoc_moment
    @inbounds begin
        thl_sec_diff = thl_sec[i, j, k+1] - thl_sec[i, j, k-1]
        wthl_sec_diff = wthl_sec[i, j, k+1] - wthl_sec[i, j, k-1]
        wsec_diff = w_sec[i, j, k] - w_sec[i, j, k-1]
        tke_diff = tke[i, j, k] - tke[i, j, k-1]
        wthl = wthl_sec[i, j, k]
    end

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

# clipping_diag_third_shoc_moments: a value beyond the bound is reset to 0.02, as in the Fortran
@inline function clip_third_moment(i, j, k, Nz, p, zt, zi, w_sec, w3)
    FT = typeof(w3)
    theterm = interpolate_to_interface(i, j, k, Nz, zt, zi, w_sec, (convert(FT, 2) / 3) * p.mintke)
    cond = p.w3clip * sqrt(2 * (theterm * theterm * theterm))
    tsign = ifelse(w3 < 0, -one(FT), one(FT))
    return ifelse(tsign * w3 > cond, convert(FT, 0.02), w3)
end

#####
##### The assumed double-Gaussian PDF (shoc_assumed_pdf and its helpers)
#####

# Saturation vapor pressure over liquid of Murphy and Koop (2005) (wv_sat_scream.F90, MurphyKoop_svp)
@inline function murphy_koop_liquid_svp(t::FT) where FT
    logt = log(t)
    tmp = convert(FT, 54.842763) - (convert(FT, 6763.22) / t) - (convert(FT, 4.210) * logt) + (convert(FT, 0.000367) * t) +
          (tanh(convert(FT, 0.0415) * (t - convert(FT, 218.8))) * (convert(FT, 53.878) - (convert(FT, 1331.22) / t) -
          (convert(FT, 9.44523) * logt) + convert(FT, 0.014025) * t))
    return exp(tmp)
end

# shoc_assumed_pdf_thl_parameters and shoc_assumed_pdf_qw_parameters share one form: the plume
# means and variances of a scalar from its variance `sec`, its flux correlation with w, and the
# skewness `skew` of the scalar (identically zero for thetal, where `dothetal_skew = false`).
@inline function plume_variance(first_plume, x1_1, x1_2, a, skew, sec)
    one_minus = 1 - a * (x1_1 * x1_1) - (1 - a) * (x1_2 * x1_2)
    third = skew - a * (x1_1 * x1_1 * x1_1) - (1 - a) * (x1_2 * x1_2 * x1_2)
    numerator = ifelse(first_plume, 3 * x1_2 * one_minus - third, -3 * x1_1 * one_minus + third)
    denominator = ifelse(first_plume, 3 * a * (x1_2 - x1_1), 3 * (1 - a) * (x1_2 - x1_1))
    # Where the plume means coincide the Fortran forms 0/0 and MAX(0, NaN) drops it; that is zero.
    degenerate = denominator == 0
    ratio = ifelse(degenerate, zero(numerator), numerator / ifelse(degenerate, one(denominator), denominator))
    return min(100, max(0, ratio)) * sec
end

@inline function scalar_plume_parameters(flux, sqrtw2, sqrt_sec, sec, tol, x_first, w1_1, w1_2, Skew_w, a, skewed, p)
    FT = typeof(sec)
    corrtest = max(-1, min(1, flux / (sqrtw2 * sqrt_sec)))

    uniform = (sec <= tol * tol) | (abs(w1_2 - w1_1) <= 0)

    # tilde plume means
    x1_1 = (-1 * corrtest) / ifelse(uniform, one(FT), w1_2)
    x1_2 = (-1 * corrtest) / ifelse(uniform, one(FT), w1_1)

    tsign = abs(x1_2 - x1_1)
    skew_qw = ifelse(tsign > p.ℂᵍ₄, p.ℂᵍ₂ * Skew_w,
                     ifelse(tsign <= p.ℂᵍ₃, zero(FT),
                            ((p.ℂᵍ₂ * Skew_w) / (p.ℂᵍ₄ - p.ℂᵍ₃)) * (tsign - p.ℂᵍ₃)))
    skew = ifelse(skewed, skew_qw, zero(FT))

    x2_1 = plume_variance(true, x1_1, x1_2, a, skew, sec)
    x2_2 = plume_variance(false, x1_1, x1_2, a, skew, sec)

    x1_1 = x1_1 * sqrt_sec + x_first
    x1_2 = x1_2 * sqrt_sec + x_first

    x1_1 = ifelse(uniform, x_first, x1_1)
    x1_2 = ifelse(uniform, x_first, x1_2)
    x2_1 = ifelse(uniform, zero(FT), x2_1)
    x2_2 = ifelse(uniform, zero(FT), x2_2)

    return x1_1, x1_2, x2_1, x2_2, zero_safe_sqrt(x2_1), zero_safe_sqrt(x2_2)
end

# shoc_assumed_pdf_compute_qs for one plume
@inline function plume_saturation(Tl, pval, p)
    FT = typeof(Tl)
    esval = murphy_koop_liquid_svp(Tl)
    qs = convert(FT, 0.622) * esval / max(esval, pval - esval)
    beta = (p.rgas / p.rv) * (p.lcond / (p.rgas * Tl)) * (p.lcond / (p.cp * Tl))
    return qs, beta
end

# shoc_assumed_pdf_compute_s
@inline function plume_condensate(qw1, qs1, beta, pval, thl2, qw2, sqrtthl2, sqrtqw2, r_qwthl, p)
    FT = typeof(qw1)
    sqrt2 = sqrt(convert(FT, 2))
    sqrt2pi = sqrt(2 * convert(FT, 3.14159265358979323))

    s = qw1 - qs1 * ((1 + beta * qw1) / (1 + beta * qs1))
    opbq = 1 + beta * qs1
    cthl = ((1 + beta * qw1) / (opbq * opbq)) * (p.cp / p.lcond) * beta * qs1 * (pval / p.basepres)^(p.rgas / p.cp)
    cqt = 1 / (1 + beta * qs1)
    tmp_val = max(0, (cthl * cthl) * thl2 + (cqt * cqt) * qw2 - 2 * cthl * sqrtthl2 * cqt * sqrtqw2 * r_qwthl)
    std_s = zero_safe_sqrt(tmp_val)

    resolved = std_s > sqrt(floatmin(FT)) * 100
    std_safe = ifelse(resolved, std_s, one(FT))

    # A PDF of finite width
    C_wide = convert(FT, 0.5) * (1 + erf(s / (sqrt2 * std_safe)))
    sos = s / std_safe
    qn_wide = ifelse(C_wide != 0, s * C_wide + (std_safe / sqrt2pi) * exp(convert(FT, -0.5) * (sos * sos)), zero(FT))

    # A PDF of no width: all or nothing
    C_narrow = ifelse(s > 0, one(FT), zero(FT))
    qn_narrow = ifelse(s > 0, s, zero(FT))

    C = ifelse(resolved, C_wide, C_narrow)
    qn = ifelse(resolved, qn_wide, qn_narrow)

    # Prevent empty clouds, or the rare occurrence of negative cloud liquid
    empty = qn <= 0
    C = ifelse(empty, zero(FT), C)
    qn = ifelse(empty, zero(FT), qn)

    return s, std_s, qn, C
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The assumed PDF of level `k` (Analytic Double Gaussian 1 of Larson et al. 2002): return the
subgrid cloud fraction, liquid water, liquid water variance, liquid water flux and buoyancy flux.
"""
@inline function assumed_pdf(i, j, k, p, zt, zi, pres, w_field, thetal, qw, w_sec,
                             thl_sec, qw_sec, qwthl_sec, wthl_sec, wqw_sec, w3)
    FT = eltype(thetal)
    largeneg = convert(FT, -99999999.99)
    thl_tol = convert(FT, 1e-2)
    rt_tol = convert(FT, 1e-4)
    w_tol_sqd = convert(FT, 2e-2) * convert(FT, 2e-2)
    Tl_min = convert(FT, 100)
    epsterm = p.rgas / p.rv

    # Interpolate the interface moments to the level
    w3var = interpolate_to_level(i, j, k, zt, zi, w3, largeneg)
    thlsec = interpolate_to_level(i, j, k, zt, zi, thl_sec, zero(FT))
    wthlsec = interpolate_to_level(i, j, k, zt, zi, wthl_sec, largeneg)
    qwthlsec = interpolate_to_level(i, j, k, zt, zi, qwthl_sec, largeneg)
    wqwsec = interpolate_to_level(i, j, k, zt, zi, wqw_sec, largeneg)
    qwsec = interpolate_to_level(i, j, k, zt, zi, qw_sec, zero(FT))

    @inbounds begin
        pval = pres[i, j, k]
        thl_first = thetal[i, j, k]
        w_first = w_field[i, j, k]
        qw_first = qw[i, j, k]
        wsec = w_sec[i, j, k]
    end

    sqrtw2 = zero_safe_sqrt(wsec)
    sqrtthl = max(thl_tol, zero_safe_sqrt(thlsec))
    sqrtqt = max(rt_tol, zero_safe_sqrt(qwsec))

    # shoc_assumed_pdf_vv_parameters
    calm = wsec <= w_tol_sqd
    wsec_safe = ifelse(calm, one(FT), wsec)
    Skew_w = w3var / sqrt(wsec_safe * wsec_safe * wsec_safe)
    w2_1 = p.ℂᵍ₁
    omw = 1 - w2_1
    a = max(convert(FT, 0.01),
            min(convert(FT, 0.5) * (1 - Skew_w * sqrt(1 / (4 * (omw * omw * omw) + Skew_w * Skew_w))), convert(FT, 0.99)))
    sqrtw2t = sqrt(1 - w2_1)
    w1_1 = sqrt((1 - a) / a) * sqrtw2t
    w1_2 = -1 * sqrt(a / (1 - a)) * sqrtw2t

    Skew_w = ifelse(calm, zero(FT), Skew_w)
    a = ifelse(calm, convert(FT, 0.5), a)
    w1_1 = ifelse(calm, w_first, w1_1)
    w1_2 = ifelse(calm, w_first, w1_2)

    # shoc_assumed_pdf_thl_parameters and shoc_assumed_pdf_qw_parameters
    thl1_1, thl1_2, thl2_1, thl2_2, sqrtthl2_1, sqrtthl2_2 =
        scalar_plume_parameters(wthlsec, sqrtw2, sqrtthl, thlsec, thl_tol, thl_first, w1_1, w1_2, Skew_w, a, false, p)

    qw1_1, qw1_2, qw2_1, qw2_2, sqrtqw2_1, sqrtqw2_2 =
        scalar_plume_parameters(wqwsec, sqrtw2, sqrtqt, qwsec, rt_tol, qw_first, w1_1, w1_2, Skew_w, a, true, p)

    # shoc_assumed_pdf_tilde_to_real
    w1_1 = w1_1 * sqrtw2 + w_first
    w1_2 = w1_2 * sqrtw2 + w_first

    # shoc_assumed_pdf_inplume_correlations
    testvar = a * sqrtqw2_1 * sqrtthl2_1 + (1 - a) * sqrtqw2_2 * sqrtthl2_2
    uncorrelated = testvar == 0
    r_qwthl_1 = max(-1, min(1, (qwthlsec - a * (qw1_1 - qw_first) * (thl1_1 - thl_first)
                                - (1 - a) * (qw1_2 - qw_first) * (thl1_2 - thl_first)) /
                               ifelse(uncorrelated, one(FT), testvar)))
    r_qwthl_1 = ifelse(uncorrelated, zero(FT), r_qwthl_1)

    # shoc_assumed_pdf_compute_temperature
    exner_inverse = (p.basepres / pval)^(p.rgas / p.cp)
    Tl1_1 = thl1_1 / exner_inverse
    Tl1_2 = thl1_2 / exner_inverse
    Tl1_1 = ifelse(Tl1_1 <= Tl_min, Tl_min, Tl1_1)
    Tl1_2 = ifelse(Tl1_2 <= Tl_min, Tl_min, Tl1_2)

    # shoc_assumed_pdf_compute_qs
    qs1, beta1 = plume_saturation(Tl1_1, pval, p)
    qs2, beta2 = plume_saturation(Tl1_2, pval, p)

    # shoc_assumed_pdf_compute_s
    s1, std_s1, qn1, C1 = plume_condensate(qw1_1, qs1, beta1, pval, thl2_1, qw2_1, sqrtthl2_1, sqrtqw2_1, r_qwthl_1, p)
    s2, std_s2, qn2, C2 = plume_condensate(qw1_2, qs2, beta2, pval, thl2_2, qw2_2, sqrtthl2_2, sqrtqw2_2, r_qwthl_1, p)

    ql1 = min(qn1, qw1_1)
    ql2 = min(qn2, qw1_2)

    shoc_cldfrac = min(1, a * C1 + (1 - a) * C2)
    shoc_ql = max(0, a * ql1 + (1 - a) * ql2)

    shoc_ql2 = a * (s1 * ql1 + C1 * (std_s1 * std_s1)) + (1 - a) * (s2 * ql2 + C2 * (std_s2 * std_s2)) - (shoc_ql * shoc_ql)
    shoc_ql2 = max(0, shoc_ql2)

    wqls = a * ((w1_1 - w_first) * ql1) + (1 - a) * ((w1_2 - w_first) * ql2)

    wthv_sec = wthlsec + ((1 - epsterm) / epsterm) * p.basetemp * wqwsec +
               ((p.lcond / p.cp) * (p.basepres / pval)^(p.rgas / p.cp) - (1 / epsterm) * p.basetemp) * wqls

    return shoc_cldfrac, shoc_ql, shoc_ql2, wqls, wthv_sec
end

#####
##### shoc_main
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance column `(i, j)` by one SHOC step of length `dtime`: `shoc_main` of E3SM `shoc.F90` with
`nadv = 1`, without `update_host_dse` and `shoc_energy_fixer` (which act on the dry static energy
of E3SM's host model). `Nz` is the number of levels and `p` the [`SHOCParameters`](@ref).

The arguments are `NamedTuple`s of arrays indexed `[i, j, k]`, with `k = 1` at the bottom:

- `state`: `thetal`, `qw`, `u_wind`, `v_wind`, `tke` on levels; updated in place.
- `host`: `zt_grid`, `zi_grid` (heights above the surface of levels and interfaces), `pres`,
  `pdel`, `inv_exner`, `thv`, `w_field` on levels, and `host_dx`, `host_dy` (two-dimensional).
- `surface`: the kinematic surface fluxes `wthl_sfc`, `wqw_sfc`, `uw_sfc`, `vw_sfc`
  (two-dimensional).
- `diagnostics`: `wthv_sec`, `tkh`, `tk`, `shoc_ql`, `shoc_cldfrac` on levels, which carry the
  state of the previous step in and that of this step out; the outputs `shoc_mix`, `isotropy`,
  `w_sec`, `wqls_sec`, `brunt`, `shoc_ql2` on levels, `thl_sec`, `qw_sec`, `qwthl_sec`,
  `wthl_sec`, `wqw_sec`, `wtke_sec`, `uw_sec`, `vw_sec`, `w3` on interfaces, and `pblh`
  (two-dimensional).
- `scratch`: `du`, `dl`, `d` on levels.
"""
@inline function shoc_column!(i, j, Nz, dtime, p, state, host, surface, diagnostics, scratch)
    thetal, qw, u_wind, v_wind, tke = state.thetal, state.qw, state.u_wind, state.v_wind, state.tke
    zt, zi, pres, pdel = host.zt_grid, host.zi_grid, host.pres, host.pdel
    thv, w_field, inv_exner = host.thv, host.w_field, host.inv_exner
    wthv_sec, tkh, tk = diagnostics.wthv_sec, diagnostics.tkh, diagnostics.tk
    shoc_ql, shoc_cldfrac, shoc_ql2 = diagnostics.shoc_ql, diagnostics.shoc_cldfrac, diagnostics.shoc_ql2
    shoc_mix, isotropy, brunt = diagnostics.shoc_mix, diagnostics.isotropy, diagnostics.brunt
    w_sec, wqls_sec = diagnostics.w_sec, diagnostics.wqls_sec
    thl_sec, qw_sec, qwthl_sec = diagnostics.thl_sec, diagnostics.qw_sec, diagnostics.qwthl_sec
    wthl_sec, wqw_sec, wtke_sec = diagnostics.wthl_sec, diagnostics.wqw_sec, diagnostics.wtke_sec
    uw_sec, vw_sec, w3 = diagnostics.uw_sec, diagnostics.vw_sec, diagnostics.w3
    du, dl, d = scratch.du, scratch.dl, scratch.d

    FT = eltype(tke)
    g = p.ggr

    @inbounds begin
        wthl_sfc = surface.wthl_sfc[i, j, 1]
        wqw_sfc = surface.wqw_sfc[i, j, 1]
        uw_sfc = surface.uw_sfc[i, j, 1]
        vw_sfc = surface.vw_sfc[i, j, 1]
        host_dx = host.host_dx[i, j, 1]
        host_dy = host.host_dy[i, j, 1]
    end

    #####
    ##### check_tke
    #####

    @inbounds for k in 1:Nz
        tke[i, j, k] = max(p.mintke, tke[i, j, k])
    end

    #####
    ##### compute_shoc_temperature (only the lowest level is used), shoc_diag_obklen, pblintd
    #####

    @inbounds tabs_sfc = thetal[i, j, 1] / inv_exner[i, j, 1] + (p.lcond / p.cp) * shoc_ql[i, j, 1]

    pblh = pbl_height(i, j, Nz, p, zt, zi, pres, thetal, qw, shoc_ql, shoc_cldfrac, u_wind, v_wind,
                      uw_sfc, vw_sfc, wthl_sfc, wqw_sfc)

    #####
    ##### shoc_length: Brunt-Vaisala frequency and L_inf; shoc_tke: integrated column stability.
    ##### The sums run from the top of the column down, as in the Fortran.
    #####

    numer = zero(FT)
    denom = zero(FT)
    brunt_int = zero(FT)

    @inbounds for k in Nz:-1:1
        thv_zi_u = interpolate_to_interface(i, j, k+1, Nz, zt, zi, thv, zero(FT))
        thv_zi_l = interpolate_to_interface(i, j, k, Nz, zt, zi, thv, zero(FT))
        dz = dz_zt(i, j, k, zi)
        brunt[i, j, k] = (g / thv[i, j, k]) * (thv_zi_u - thv_zi_l) / dz

        tkes = sqrt(tke[i, j, k])
        numer = numer + tkes * zt[i, j, k] * dz
        denom = denom + tkes * dz

        brunt_int = ifelse(pres[i, j, k] > p.troppres, brunt_int + dz * brunt[i, j, k], brunt_int)
    end

    l_inf = p.ℂˡ₄ * (numer / denom)

    #####
    ##### Level by level: mixing length (compute_shoc_mix_shoc_length, check_length_scale_shoc_length),
    ##### adv_sgs_tke, isotropic_ts, eddy_diffusivities
    #####

    tscale = p.ℂˡ₅ # turnover time scale [s]
    maxiso = convert(FT, 20000) # maximum return-to-isotropy time scale [s]
    temp_crit = convert(FT, 182) # surface temperature below which extra mixing applies [K]
    pbl_trans = convert(FT, 200) # transition depth above the boundary layer top [m]

    Cs = p.ℂᴰ₂
    Ck = p.ℂᴰ₁
    Ce = (Ck * Ck * Ck) / ((Cs * Cs) * (Cs * Cs))
    Ce1 = Ce / convert(FT, 0.7) * convert(FT, 0.19)
    Ce2 = Ce / convert(FT, 0.7) * convert(FT, 0.51)
    Cee = Ce1 + Ce2

    host_scale = sqrt(host_dx * host_dy)

    # The damping of the return-to-isotropy time scale by column stability
    lambda_column = p.ℂᵝ₁ + ((brunt_int / g) - p.ℂᵝ₄) * p.ℂᵝ₃
    lambda_column = max(p.ℂᵝ₁, min(p.ℂᵝ₂, lambda_column))

    @inbounds for k in 1:Nz
        tke_k = tke[i, j, k]
        brunt_k = brunt[i, j, k]

        tkes = sqrt(tke_k)
        brunt2 = ifelse(brunt_k >= 0, brunt_k, zero(FT))

        mix = min(p.maxlen, (p.ℂˡ₂ * sqrt(1 / ((1 / (tscale * tkes * p.vk * zt[i, j, k]))
                             + (1 / (tscale * tkes * l_inf)) + p.ℂˡ₃ * (brunt2 / tke_k)))) / p.ℂˡ₁)
        mix = min(p.maxlen, mix)
        mix = max(p.minlen, mix)
        mix = min(host_scale, mix)
        shoc_mix[i, j, k] = mix

        # adv_sgs_tke
        sterm_zt = shear_production_term_zt(i, j, k, Nz, p, zt, zi, u_wind, v_wind)
        a_prod_bu = (g / p.basetemp) * wthv_sec[i, j, k]
        tke_k = max(0, tke_k)
        a_prod_sh = tk[i, j, k] * sterm_zt
        a_diss = Cee / mix * tke_k^convert(FT, 1.5)
        tke_k = max(p.mintke, tke_k + dtime * (max(0, a_prod_sh + a_prod_bu) - a_diss))
        tke_k = min(tke_k, p.maxtke)
        tke[i, j, k] = tke_k

        # isotropic_ts
        iso_scale = (2 * tke_k) / a_diss
        lambda = ifelse(brunt_k <= 0, zero(FT), lambda_column)
        iso = min(maxiso, iso_scale / (1 + lambda * brunt_k * (iso_scale * iso_scale)))
        isotropy[i, j, k] = iso

        # eddy_diffusivities
        runaway = (tabs_sfc < temp_crit) & (zt[i, j, k] < pblh + pbl_trans)
        sqrt_sterm = zero_safe_sqrt(sterm_zt)
        tkh[i, j, k] = ifelse(runaway, p.ℂᶜₛ * (mix * mix) * sqrt_sterm, p.ℂᶜ * iso * tke_k)
        tk[i, j, k]  = ifelse(runaway, p.ℂᵘₛ * (mix * mix) * sqrt_sterm, p.ℂᵘ * iso * tke_k)
    end

    #####
    ##### update_prognostics_implicit
    #####

    rho_zi_sfc = rho_zi(i, j, 1, Nz, p, zt, zi, pdel)
    rdp_zt_sfc = rdp_zt(i, j, 1, p, zi, pdel)

    # impli_srf_stress_term
    wsmin = one(FT) # minimum wind speed for ksrf [m/s]
    ksrfmin = convert(FT, 1e-4) # minimum surface drag coefficient [kg/s/m²]
    taux = rho_zi_sfc * uw_sfc
    tauy = rho_zi_sfc * vw_sfc
    @inbounds ws = max(zero_safe_sqrt(u_wind[i, j, 1] * u_wind[i, j, 1] + v_wind[i, j, 1] * v_wind[i, j, 1]), wsmin)
    tau = zero_safe_sqrt(taux * taux + tauy * tauy)
    ksrf = max(tau / ws, ksrfmin)

    # tke_srf_flux_term
    ustar = max(zero_safe_sqrt(zero_safe_sqrt(uw_sfc * uw_sfc + vw_sfc * vw_sfc)), convert(FT, 0.01))
    wtke_sfc = ustar * ustar * ustar

    # sfc_fluxes
    cmnfac = dtime * (g * rho_zi_sfc * rdp_zt_sfc)
    @inbounds begin
        thetal[i, j, 1] = thetal[i, j, 1] + cmnfac * wthl_sfc
        qw[i, j, 1]     = qw[i, j, 1]     + cmnfac * wqw_sfc
        tke[i, j, 1]    = tke[i, j, 1]    + cmnfac * wtke_sfc
    end

    # Momentum
    vd_shoc_decomp!(i, j, Nz, dtime, p, zt, zi, pdel, tk, ksrf, du, dl, d)
    vd_shoc_solve!(i, j, Nz, du, dl, d, u_wind)
    vd_shoc_solve!(i, j, Nz, du, dl, d, v_wind)

    # Thermodynamic variables and TKE (the surface fluxes were applied explicitly above)
    vd_shoc_decomp!(i, j, Nz, dtime, p, zt, zi, pdel, tkh, zero(FT), du, dl, d)
    vd_shoc_solve!(i, j, Nz, du, dl, d, thetal)
    vd_shoc_solve!(i, j, Nz, du, dl, d, qw)
    vd_shoc_solve!(i, j, Nz, du, dl, d, tke)

    #####
    ##### diag_second_shoc_moments
    #####

    # diag_second_moments_srf
    ustar2 = zero_safe_sqrt(uw_sfc * uw_sfc + vw_sfc * vw_sfc)
    heated = wthl_sfc > 0
    wstar_cubed = ifelse(heated, 1 / p.basetemp * g * wthl_sfc * 1, one(FT))
    wstar = ifelse(heated, wstar_cubed^(convert(FT, 1) / 3), zero(FT))

    # diag_second_moments_lbycond
    a_const = p.ℂᶠ
    uf = zero_safe_sqrt(ustar2 + convert(FT, 0.3) * wstar * wstar)
    uf = max(convert(FT, 0.01), uf)
    maxval = max(zero_safe_sqrt(ustar2), convert(FT, 0.01))

    @inbounds begin
        thl_sec[i, j, 1] = convert(FT, 0.4) * a_const * ((wthl_sfc / uf) * (wthl_sfc / uf))
        qw_sec[i, j, 1] = convert(FT, 0.4) * a_const * ((wqw_sfc / uf) * (wqw_sfc / uf))
        qwthl_sec[i, j, 1] = convert(FT, 0.2) * a_const * (wthl_sfc / uf) * (wqw_sfc / uf)
        wthl_sec[i, j, 1] = wthl_sfc
        wqw_sec[i, j, 1] = wqw_sfc
        uw_sec[i, j, 1] = uw_sfc
        vw_sec[i, j, 1] = vw_sfc
        wtke_sec[i, j, 1] = maxval * maxval * maxval
    end

    # diag_second_moments: vertical velocity variance
    w2fac = p.ℂʷ₁ * (convert(FT, 2) / 3)
    @inbounds for k in 1:Nz
        w_sec[i, j, k] = w2fac * tke[i, j, k]
    end

    # diag_second_moments: calc_shoc_varorcovar and calc_shoc_vertflux at the interior interfaces
    @inbounds for k in 2:Nz
        isotropy_zi = interpolate_to_interface(i, j, k, Nz, zt, zi, isotropy, zero(FT))
        tkh_zi = interpolate_to_interface(i, j, k, Nz, zt, zi, tkh, zero(FT))
        tk_zi = interpolate_to_interface(i, j, k, Nz, zt, zi, tk, zero(FT))

        grid_dz = 1 / dz_zi(i, j, k, zt)
        grid_dz2 = grid_dz * grid_dz
        sm = isotropy_zi * tkh_zi

        Δthetal = thetal[i, j, k] - thetal[i, j, k-1]
        Δqw = qw[i, j, k] - qw[i, j, k-1]

        thl_sec[i, j, k] = p.ℂᵛ₁ * sm * grid_dz2 * Δthetal * Δthetal
        qw_sec[i, j, k] = p.ℂᵛ₂ * sm * grid_dz2 * Δqw * Δqw
        qwthl_sec[i, j, k] = p.ℂᵛ₃ * sm * grid_dz2 * Δthetal * Δqw

        wthl_sec[i, j, k] = -1 * tkh_zi * grid_dz * Δthetal
        wqw_sec[i, j, k] = -1 * tkh_zi * grid_dz * Δqw
        wtke_sec[i, j, k] = -1 * tkh_zi * grid_dz * (tke[i, j, k] - tke[i, j, k-1])
        uw_sec[i, j, k] = -1 * tk_zi * grid_dz * (u_wind[i, j, k] - u_wind[i, j, k-1])
        vw_sec[i, j, k] = -1 * tk_zi * grid_dz * (v_wind[i, j, k] - v_wind[i, j, k-1])
    end

    # diag_second_moments_ubycond
    @inbounds begin
        thl_sec[i, j, Nz+1] = 0
        qw_sec[i, j, Nz+1] = 0
        qwthl_sec[i, j, Nz+1] = 0
        wthl_sec[i, j, Nz+1] = 0
        wqw_sec[i, j, Nz+1] = 0
        wtke_sec[i, j, Nz+1] = 0
        uw_sec[i, j, Nz+1] = 0
        vw_sec[i, j, Nz+1] = 0
    end

    #####
    ##### diag_third_shoc_moments
    #####

    @inbounds for k in 2:Nz
        w3_k = third_moment(i, j, k, Nz, p, zt, zi, thetal, tke, isotropy, brunt, w_sec, thl_sec, wthl_sec)
        w3[i, j, k] = clip_third_moment(i, j, k, Nz, p, zt, zi, w_sec, w3_k)
    end

    @inbounds begin
        w3[i, j, 1] = clip_third_moment(i, j, 1, Nz, p, zt, zi, w_sec, zero(FT))
        w3[i, j, Nz+1] = clip_third_moment(i, j, Nz+1, Nz, p, zt, zi, w_sec, zero(FT))
    end

    #####
    ##### shoc_assumed_pdf, then check_tke
    #####

    @inbounds for k in 1:Nz
        cldfrac_k, ql_k, ql2_k, wqls_k, wthv_k =
            assumed_pdf(i, j, k, p, zt, zi, pres, w_field, thetal, qw, w_sec,
                        thl_sec, qw_sec, qwthl_sec, wthl_sec, wqw_sec, w3)

        shoc_cldfrac[i, j, k] = cldfrac_k
        shoc_ql[i, j, k] = ql_k
        shoc_ql2[i, j, k] = ql2_k
        wqls_sec[i, j, k] = wqls_k
        wthv_sec[i, j, k] = wthv_k

        tke[i, j, k] = max(p.mintke, tke[i, j, k])
    end

    #####
    ##### The boundary layer depth of the updated state
    #####

    @inbounds diagnostics.pblh[i, j, 1] = pbl_height(i, j, Nz, p, zt, zi, pres, thetal, qw, shoc_ql, shoc_cldfrac,
                                                    u_wind, v_wind, uw_sfc, vw_sfc, wthl_sfc, wqw_sfc)

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance every column of the arrays by one SHOC step, one thread per column. See
[`shoc_column!`](@ref) for the arguments.
"""
@kernel function _shoc_columns!(Nz, dtime, p, state, host, surface, diagnostics, scratch)
    i, j = @index(Global, NTuple)
    shoc_column!(i, j, Nz, dtime, p, state, host, surface, diagnostics, scratch)
end
