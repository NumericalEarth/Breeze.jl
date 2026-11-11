#####
##### Implicit diffusion (vd_shoc_decomp, vd_shoc_solve)
#####

# Superdiagonal `du` (coupling to the level below) and Thomas-factorized subdiagonal `dl` and
# diagonal `d` of the diffusion matrix for the level diffusivity `kv`. The factorization sweeps
# from the top of the column down, as in the Fortran.
#
# The matrix coefficients and the elimination of the Fortran's two loops are done in one downward
# sweep; the level quantities of the level above (height, diffusivity, density, 1/dp) and the
# factorized coefficients of the level above are carried in registers. The arithmetic of every
# coefficient is that of the Fortran.

# The coupling coefficients of interface k (2 ≤ k ≤ Nz) between levels k (upper, `_u`) and k-1
# (lower, `_l`): du of level k and the unfactorized dl of level k-1.
@inline function interface_coupling(dtime, p, zt_u, zt_l, kv_u, kv_l, rho_u, rho_l, rdp_u, rdp_l, zi_k)
    FT = typeof(zt_u)
    kv_zi = hard_floor(linear_interp(zt_u, zt_l, kv_u, kv_l, zi_k), zero(FT))
    rho_zi_k = hard_floor(linear_interp(zt_u, zt_l, rho_u, rho_l, zi_k), zero(FT))
    # tmpi = dt * (g * rho)² / dp at the interface, with dp = g * rho * dz
    tmpi = dtime * (p.ggr * rho_zi_k) / (zt_u - zt_l)
    du_k = -1 * kv_zi * tmpi * rdp_u
    dl_l = -1 * kv_zi * tmpi * rdp_l
    return du_k, dl_l
end

# The diagonal of level k with the implicit sink of rate `sink[i, j, k]` added; without a sink the
# diagonal is returned as it is (not as `diagonal + 0`).
@inline diagonal_with_sink(diagonal, ::Nothing, i, j, k, dtime) = diagonal
@inline diagonal_with_sink(diagonal, sink, i, j, k, dtime) = @inbounds diagonal + dtime * sink[i, j, k]

# With `sink`, a level array of non-negative rates ω, the matrix is that of
# (1 + Δt ω - Δt ∂z K ∂z): the diffusion with a linear sink treated implicitly.
@inline function vd_shoc_decomp!(i, j, Nz, dtime, p, zt, zi, pdel, kv, flux, du, dl, d, sink=nothing)
    FT = typeof(flux)

    @inbounds begin
        # Level Nz and interface Nz
        zt_u = zt[i, j, Nz]
        kv_u = kv[i, j, Nz]
        rho_u = rho_zt(i, j, Nz, p, zi, pdel)
        rdp_u = rdp_zt(i, j, Nz, p, zi, pdel)

        zt_l = zt[i, j, Nz-1]
        kv_l = kv[i, j, Nz-1]
        rho_l = rho_zt(i, j, Nz-1, p, zi, pdel)
        rdp_l = rdp_zt(i, j, Nz-1, p, zi, pdel)

        du_u, dl_k = interface_coupling(dtime, p, zt_u, zt_l, kv_u, kv_l, rho_u, rho_l, rdp_u, rdp_l, zi[i, j, Nz])
        d_u = diagonal_with_sink(1 - du_u, sink, i, j, Nz, dtime)
        du[i, j, Nz] = du_u
        dl[i, j, Nz] = 0
        d[i, j, Nz] = d_u

        # Levels Nz-1 to 2: dl_k is the unfactorized dl of level k
        for k in Nz-1:-1:2
            zt_u, kv_u, rho_u, rdp_u = zt_l, kv_l, rho_l, rdp_l
            zt_l = zt[i, j, k-1]
            kv_l = kv[i, j, k-1]
            rho_l = rho_zt(i, j, k-1, p, zi, pdel)
            rdp_l = rdp_zt(i, j, k-1, p, zi, pdel)

            du_k, dl_l = interface_coupling(dtime, p, zt_u, zt_l, kv_u, kv_l, rho_u, rho_l, rdp_u, rdp_l, zi[i, j, k])

            d_k = diagonal_with_sink(1 - du_k - dl_k, sink, i, j, k, dtime)
            dl_k = dl_k / d_u
            d_k = d_k - dl_k * du_u

            du[i, j, k] = du_k
            dl[i, j, k] = dl_k
            d[i, j, k] = d_k

            du_u, d_u, dl_k = du_k, d_k, dl_l
        end

        # Level 1, with the implicit surface flux
        d_1 = diagonal_with_sink(1 - dl_k + flux * dtime * p.ggr * rdp_l, sink, i, j, 1, dtime)
        dl_k = dl_k / d_u
        d_1 = d_1 - dl_k * du_u
        du[i, j, 1] = 0
        dl[i, j, 1] = dl_k
        d[i, j, 1] = d_1
    end

    return nothing
end

# Solve for one or more right-hand sides `vars` (a tuple of arrays) at once, so that the
# factorization is read once; the value of the neighbouring level is carried in a register.
@inline function vd_shoc_solve!(i, j, Nz, du, dl, d, vars::Tuple)
    @inbounds begin
        above = map(var -> var[i, j, Nz], vars)
        for k in Nz-1:-1:1
            dl_k = dl[i, j, k]
            above = map(vars, above) do var, a
                x = var[i, j, k] - dl_k * a
                var[i, j, k] = x
                x
            end
        end

        d_1 = d[i, j, 1]
        below = map(var -> (x = var[i, j, 1] / d_1; var[i, j, 1] = x; x), vars)

        for k in 2:Nz
            du_k = du[i, j, k]
            d_k = d[i, j, k]
            below = map(vars, below) do var, b
                x = (var[i, j, k] - du_k * b) / d_k
                var[i, j, k] = x
                x
            end
        end
    end

    return nothing
end

@inline vd_shoc_solve!(i, j, Nz, du, dl, d, var::AbstractArray) = vd_shoc_solve!(i, j, Nz, du, dl, d, (var,))
