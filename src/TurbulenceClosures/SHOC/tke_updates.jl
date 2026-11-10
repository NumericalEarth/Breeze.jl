#####
##### Slot: update of the turbulent kinetic energy
#####
##### A TKE update component holds the shear production coefficient `ℂᴾ` and implements
#####
#####     tke_step(tke_update, cp, c, dtime, e, P, B, ℓ, Sᴰ) -> (e_new, ε, ω)
#####
##### level by level in the turbulence sweep (`P` and `B` are the shear and buoyancy production,
##### `ℓ` the mixing length and `Sᴰ` the dissipation coefficient; it returns the turbulent kinetic
##### energy that the time scale and the diffusivities see, the dissipation `ε` and the rate `ω` of
##### the sinks left to the implicit solve, `nothing` if there are none), and
#####
#####     implicit_scalar_solve!(tke_update, i, j, Nz, dtime, c, cp, host, diagnostics, scratch, thetal, qw, tke)
#####
##### which diffuses θˡ, qᵗ and the turbulent kinetic energy implicitly with the heat diffusivity.
#####

#####
##### Shear production (compute_shr_prod), on interfaces
#####

@inline function shear_production_term(i, j, k, Nz, tke_update, zt, u_wind, v_wind)
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
    sterm = tke_update.ℂᴾ * (u_grad * u_grad + v_grad * v_grad)
    return ifelse((k == 1) | (k == Nz + 1), zero(sterm), sterm)
end

#####
##### Forward Euler (adv_sgs_tke)
#####

"""
$(TYPEDEF)

The update of turbulent kinetic energy of E3SM `shoc.F90` (`adv_sgs_tke`): forward Euler in the
production and the dissipation,

    e ← e + Δt (max(0, P + B) - ε),    ε = Sᴰ e^{3/2} / ℓ,    P = ℂᴾ Kᵘ |∂z u|²

kept between the minimum and the maximum of the `SHOCConstants`; the diffusion of `e` shares the
implicit solve of θˡ and qᵗ.

$(TYPEDFIELDS)
"""
struct ForwardEulerTKE{FT} <: AbstractSHOCComponent
    "Shear production coefficient (`Ck_sh`)"
    ℂᴾ :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ForwardEulerTKE{FT}` with the shear production coefficient of E3SM `shoc.F90`.
"""
ForwardEulerTKE(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ForwardEulerTKE, FT, (; ℂᴾ = 0.1), kw)

Adapt.@adapt_structure ForwardEulerTKE

@inline function tke_step(::ForwardEulerTKE, cp, c, dtime, tke_k, a_prod_sh, a_prod_bu, mix, Cee)
    FT = typeof(tke_k)
    production_scale = convert(FT, 1e-6) # scale of the production for the clipping policy [m² s⁻³]

    tke_k = max(0, tke_k)
    a_diss = Cee / mix * tke_k^convert(FT, 1.5)
    tke_k = clip_max(cp, c.mintke, tke_k + dtime * (clip_max(cp, 0, a_prod_sh + a_prod_bu, production_scale) - a_diss), c.mintke)
    tke_k = clip_min(cp, tke_k, c.maxtke, c.mintke)

    return tke_k, a_diss, nothing
end

# One factorization for θˡ, qᵗ and the turbulent kinetic energy (the surface fluxes have been
# applied explicitly)
@inline function implicit_scalar_solve!(::ForwardEulerTKE, i, j, Nz, dtime, c, cp, host, diagnostics, scratch,
                                        thetal, qw, tke)
    zt, zi, pdel = host.zt_grid, host.zi_grid, host.pdel
    du, dl, d = scratch.du, scratch.dl, scratch.d
    vd_shoc_decomp!(i, j, Nz, dtime, c, zt, zi, pdel, diagnostics.tkh, zero(eltype(tke)), du, dl, d)
    vd_shoc_solve!(i, j, Nz, du, dl, d, (thetal, qw, tke))
    return nothing
end

# The sink rate of a level, kept for the implicit solve by the updates that have one
@inline store_tke_sink_rate!(scratch, i, j, k, ::Nothing) = nothing
