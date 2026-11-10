#####
##### Interpolation between levels and interfaces
#####

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
    return hard_floor(y2, minthresh)
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
    return hard_floor(y2, zero(y2))
end

# 1 / dp of level k
@inline rdp_zt(i, j, k, p, zi, pdel) = 1 / (p.ggr * rho_zt(i, j, k, p, zi, pdel) * dz_zt(i, j, k, zi))
