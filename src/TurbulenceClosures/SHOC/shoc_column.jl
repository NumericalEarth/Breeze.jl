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
    shoc_turbulence!(i, j, Nz, dtime, p, state, host, surface, diagnostics, scratch)
    shoc_implicit_moments!(i, j, Nz, dtime, p, state, host, surface, diagnostics, scratch)
    shoc_cloud!(i, j, Nz, p, state, host, surface, diagnostics)
    return nothing
end

# The first part of the step: boundary layer depth, length scale, TKE production and dissipation,
# eddy diffusivities. Split from `shoc_cloud!` so that each part can run as its own kernel, with
# fewer registers per thread.
@inline function shoc_turbulence!(i, j, Nz, dtime, p, state, host, surface, diagnostics, scratch)
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
    ##### compute_shoc_temperature (only the lowest level is used), shoc_diag_obklen, pblintd
    #####

    @inbounds tabs_sfc = thetal[i, j, 1] / inv_exner[i, j, 1] + (p.lcond / p.cp) * shoc_ql[i, j, 1]

    pblh = pbl_height(i, j, Nz, p, zt, zi, pres, thetal, qw, shoc_ql, shoc_cldfrac, u_wind, v_wind,
                      uw_sfc, vw_sfc, wthl_sfc, wqw_sfc)

    #####
    ##### shoc_length: Brunt-Vaisala frequency and L_inf; shoc_tke: integrated column stability.
    ##### The sums run from the top of the column down, as in the Fortran. check_tke (which
    ##### nothing before this loop reads) is fused into it.
    #####

    numer = zero(FT)
    denom = zero(FT)
    brunt_int = zero(FT)
    thv_zi_u = interpolate_to_interface(i, j, Nz+1, Nz, zt, zi, thv, zero(FT))

    @inbounds for k in Nz:-1:1
        tke_k = max(p.mintke, tke[i, j, k])
        tke[i, j, k] = tke_k

        thv_zi_l = interpolate_to_interface(i, j, k, Nz, zt, zi, thv, zero(FT))
        dz = dz_zt(i, j, k, zi)
        brunt[i, j, k] = (g / thv[i, j, k]) * (thv_zi_u - thv_zi_l) / dz
        thv_zi_u = thv_zi_l

        tkes = sqrt(tke_k)
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

    # The shear production term of the interface below the level, carried up the column
    sterm_l = shear_production_term(i, j, 1, Nz, p, zt, u_wind, v_wind)

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
        # compute_shr_prod at interface k+1, interpolated to level k with that of interface k
        sterm_u = shear_production_term(i, j, k+1, Nz, p, zt, u_wind, v_wind)
        sterm_zt = clip_below(linear_interp(zi[i, j, k+1], zi[i, j, k], sterm_u, sterm_l, zt[i, j, k]), zero(FT))
        sterm_l = sterm_u
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

    return nothing
end

# The implicit diffusion of the state and TKE, and the second and third moments.
@inline function shoc_implicit_moments!(i, j, Nz, dtime, p, state, host, surface, diagnostics, scratch)
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
    vd_shoc_solve!(i, j, Nz, du, dl, d, (u_wind, v_wind))

    # Thermodynamic variables and TKE (the surface fluxes were applied explicitly above)
    vd_shoc_decomp!(i, j, Nz, dtime, p, zt, zi, pdel, tkh, zero(FT), du, dl, d)
    vd_shoc_solve!(i, j, Nz, du, dl, d, (thetal, qw, tke))

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

    # diag_second_moments: the vertical velocity variance, and calc_shoc_varorcovar and
    # calc_shoc_vertflux at the interior interfaces, in one upward sweep that carries the values
    # of the level below the interface
    w2fac = p.ℂʷ₁ * (convert(FT, 2) / 3)

    @inbounds begin
        zt_l = zt[i, j, 1]
        iso_l = isotropy[i, j, 1]
        tkh_l = tkh[i, j, 1]
        tk_l = tk[i, j, 1]
        thetal_l = thetal[i, j, 1]
        qw_l = qw[i, j, 1]
        tke_l = tke[i, j, 1]
        u_l = u_wind[i, j, 1]
        v_l = v_wind[i, j, 1]
        w_sec[i, j, 1] = w2fac * tke_l
    end

    @inbounds for k in 2:Nz
        zt_u = zt[i, j, k]
        iso_u = isotropy[i, j, k]
        tkh_u = tkh[i, j, k]
        tk_u = tk[i, j, k]
        thetal_u = thetal[i, j, k]
        qw_u = qw[i, j, k]
        tke_u = tke[i, j, k]
        u_u = u_wind[i, j, k]
        v_u = v_wind[i, j, k]
        zi_k = zi[i, j, k]

        w_sec[i, j, k] = w2fac * tke_u

        isotropy_zi = clip_below(linear_interp(zt_u, zt_l, iso_u, iso_l, zi_k), zero(FT))
        tkh_zi = clip_below(linear_interp(zt_u, zt_l, tkh_u, tkh_l, zi_k), zero(FT))
        tk_zi = clip_below(linear_interp(zt_u, zt_l, tk_u, tk_l, zi_k), zero(FT))

        grid_dz = 1 / (zt_u - zt_l)
        grid_dz2 = grid_dz * grid_dz
        sm = isotropy_zi * tkh_zi

        Δthetal = thetal_u - thetal_l
        Δqw = qw_u - qw_l

        thl_sec[i, j, k] = p.ℂᵛ₁ * sm * grid_dz2 * Δthetal * Δthetal
        qw_sec[i, j, k] = p.ℂᵛ₂ * sm * grid_dz2 * Δqw * Δqw
        qwthl_sec[i, j, k] = p.ℂᵛ₃ * sm * grid_dz2 * Δthetal * Δqw

        wthl_sec[i, j, k] = -1 * tkh_zi * grid_dz * Δthetal
        wqw_sec[i, j, k] = -1 * tkh_zi * grid_dz * Δqw
        wtke_sec[i, j, k] = -1 * tkh_zi * grid_dz * (tke_u - tke_l)
        uw_sec[i, j, k] = -1 * tk_zi * grid_dz * (u_u - u_l)
        vw_sec[i, j, k] = -1 * tk_zi * grid_dz * (v_u - v_l)

        zt_l, iso_l, tkh_l, tk_l = zt_u, iso_u, tkh_u, tk_u
        thetal_l, qw_l, tke_l, u_l, v_l = thetal_u, qw_u, tke_u, u_u, v_u
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

    # An upward sweep over the interior interfaces that carries the values of the level below
    # and of the interface below
    largeneg = convert(FT, -99999999.99)
    w_sec_min = (convert(FT, 2) / 3) * p.mintke

    @inbounds begin
        zt_l = zt[i, j, 1]
        iso_l = isotropy[i, j, 1]
        brunt_l = brunt[i, j, 1]
        w_sec_l = w_sec[i, j, 1]
        thetal_l = thetal[i, j, 1]
        tke_l = tke[i, j, 1]
        dz_l = dz_zt(i, j, 1, zi)
        thl_sec_below = thl_sec[i, j, 1]
        wthl_sec_below = wthl_sec[i, j, 1]
        wthl_k = wthl_sec[i, j, 2]
    end

    @inbounds for k in 2:Nz
        zt_u = zt[i, j, k]
        iso_u = isotropy[i, j, k]
        brunt_u = brunt[i, j, k]
        w_sec_u = w_sec[i, j, k]
        thetal_u = thetal[i, j, k]
        tke_u = tke[i, j, k]
        dz_u = dz_zt(i, j, k, zi)
        zi_k = zi[i, j, k]
        thl_sec_above = thl_sec[i, j, k+1]
        wthl_sec_above = wthl_sec[i, j, k+1]

        isotropy_zi = clip_below(linear_interp(zt_u, zt_l, iso_u, iso_l, zi_k), zero(FT))
        brunt_zi = clip_below(linear_interp(zt_u, zt_l, brunt_u, brunt_l, zi_k), largeneg)
        w_sec_zi = clip_below(linear_interp(zt_u, zt_l, w_sec_u, w_sec_l, zi_k), w_sec_min)
        thetal_zi = clip_below(linear_interp(zt_u, zt_l, thetal_u, thetal_l, zi_k), zero(FT))

        thedz = 1 / (zt_u - zt_l)
        thedz2 = 1 / (dz_l + dz_u)

        w3_k = third_moment(p, isotropy_zi, brunt_zi, w_sec_zi, thetal_zi, thedz, thedz2,
                            thl_sec_above - thl_sec_below, wthl_sec_above - wthl_sec_below,
                            w_sec_u - w_sec_l, tke_u - tke_l, wthl_k)

        w3[i, j, k] = clip_third_moment(p, w_sec_zi, w3_k)

        zt_l, iso_l, brunt_l, w_sec_l, thetal_l, tke_l, dz_l = zt_u, iso_u, brunt_u, w_sec_u, thetal_u, tke_u, dz_u
        thl_sec_below = thl_sec[i, j, k]
        wthl_sec_below, wthl_k = wthl_k, wthl_sec_above
    end

    @inbounds begin
        w3[i, j, 1] = clip_third_moment(p, interpolate_to_interface(i, j, 1, Nz, zt, zi, w_sec, w_sec_min), zero(FT))
        w3[i, j, Nz+1] = clip_third_moment(p, interpolate_to_interface(i, j, Nz+1, Nz, zt, zi, w_sec, w_sec_min), zero(FT))
    end

    return nothing
end

# The second part of the step: the assumed PDF (subgrid cloud and buoyancy flux), check_tke and the
# boundary layer depth of the updated state.
@inline function shoc_cloud!(i, j, Nz, p, state, host, surface, diagnostics)
    thetal, qw, u_wind, v_wind, tke = state.thetal, state.qw, state.u_wind, state.v_wind, state.tke
    zt, zi, pres, w_field = host.zt_grid, host.zi_grid, host.pres, host.w_field
    wthv_sec, shoc_ql, shoc_cldfrac, shoc_ql2 = diagnostics.wthv_sec, diagnostics.shoc_ql, diagnostics.shoc_cldfrac, diagnostics.shoc_ql2
    w_sec, wqls_sec, w3 = diagnostics.w_sec, diagnostics.wqls_sec, diagnostics.w3
    thl_sec, qw_sec, qwthl_sec = diagnostics.thl_sec, diagnostics.qw_sec, diagnostics.qwthl_sec
    wthl_sec, wqw_sec = diagnostics.wthl_sec, diagnostics.wqw_sec

    @inbounds begin
        wthl_sfc = surface.wthl_sfc[i, j, 1]
        wqw_sfc = surface.wqw_sfc[i, j, 1]
        uw_sfc = surface.uw_sfc[i, j, 1]
        vw_sfc = surface.vw_sfc[i, j, 1]
    end

    #####
    ##### shoc_assumed_pdf, then check_tke
    #####

    interface_moments(k) = @inbounds (w3[i, j, k], thl_sec[i, j, k], wthl_sec[i, j, k],
                                      qwthl_sec[i, j, k], wqw_sec[i, j, k], qw_sec[i, j, k])
    moments_l = interface_moments(1)

    @inbounds for k in 1:Nz
        moments_u = interface_moments(k+1)
        cldfrac_k, ql_k, ql2_k, wqls_k, wthv_k =
            assumed_pdf(i, j, k, p, zt, zi, pres, w_field, thetal, qw, w_sec, moments_l, moments_u)
        moments_l = moments_u

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
