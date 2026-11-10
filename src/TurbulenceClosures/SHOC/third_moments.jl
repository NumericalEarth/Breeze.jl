#####
##### Third moment of vertical velocity (compute_diag_third_shoc_moment and its helpers)
#####

# The third moment at an interior interface from the interface values of isotropy, Brunt-Vaisala
# frequency, w_sec and thetal, the inverse thicknesses `thedz` and `thedz2`, and the differences
# of the moments across the interface; `wthl` is wthl_sec at the interface.
@inline function third_moment(p, isotropy_zi, brunt_zi, w_sec_zi, thetal_zi, thedz, thedz2,
                              thl_sec_diff, wthl_sec_diff, wsec_diff, tke_diff, wthl)
    FT = typeof(wthl)
    c = p.ℂʷ₂

    # fterms_input_for_diag_third_shoc_moment
    iso = isotropy_zi
    isosqrd = iso * iso
    buoy_sgs2 = isosqrd * brunt_zi
    bet2 = p.ggr / thetal_zi

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
# `theterm` is w_sec interpolated to the interface
@inline function clip_third_moment(p, theterm, w3)
    FT = typeof(w3)
    cond = p.w3clip * sqrt(2 * (theterm * theterm * theterm))
    tsign = ifelse(w3 < 0, -one(FT), one(FT))
    return ifelse(tsign * w3 > cond, convert(FT, 0.02), w3)
end
