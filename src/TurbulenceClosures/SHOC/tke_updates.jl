#####
##### Shear production (compute_shr_prod), on interfaces and interpolated to levels
#####

@inline function shear_production_term(i, j, k, Nz, p, zt, u_wind, v_wind)
    ku = clamp(k, 2, Nz)
    kl = ku - 1
    @inbounds begin
        # The level spacing is written out rather than taken from `dz_zi(i, j, ku, zt)` (identical
        # values, since ku ≥ 2): with `dz_zi`'s own `k == 1` branch and `max(k-1, 1)` index nested
        # in the clamped index, Reactant 0.2.300 raises this function to a wrong program (the
        # shear production is wrong wherever `tk` is nonzero; Reactant 0.2.308 is unaffected).
        grid_dz = 1 / (zt[i, j, ku] - zt[i, j, kl])
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
    return hard_floor(y2, zero(y2))
end
