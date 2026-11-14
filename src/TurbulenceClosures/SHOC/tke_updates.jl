#####
##### Slot: update of the turbulent kinetic energy
#####
##### A TKE update component holds the shear production coefficient `ℂᴾ` and implements
#####
#####     tke_step(tke_update, cp, c, dtime, e, P, B, ℓ, 𝕊ᴰ) -> (e_new, ε, ω, e_stored)
#####
##### level by level in the turbulence sweep (`P` and `B` are the shear and buoyancy production,
##### `ℓ` the mixing length and `𝕊ᴰ` the dissipation coefficient; it returns the turbulent kinetic
##### energy that the time scale and the diffusivities see, the dissipation `ε`, the rate `ω` of
##### the sinks left to the implicit solve, `nothing` if there are none, and the value to store,
##### which the implicit solve takes as its right-hand side), and
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

    e ← e + Δt (max(0, P + B) - ε),    ε = 𝕊ᴰ e^{3/2} / ℓ,    P = ℂᴾ Kᵘ |∂z u|²

kept between the minimum and the maximum of the `SHOCConstants`; the diffusion of `e` shares the
implicit solve of θˡ and qᵗ.

$(TYPEDFIELDS)
"""
struct ForwardEulerTKE{FT} <: AbstractSHOCComponent
    "Shear production coefficient (`Ck_sh`)"
    ℂᴾ :: FT
    "Magnitude of the production that scales the smoothing of its positive part, m² s⁻³; `HardClipping` ignores it"
    production_scale :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ForwardEulerTKE{FT}` with the shear production coefficient of E3SM `shoc.F90`. Every field is a
keyword argument.
"""
ForwardEulerTKE(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ForwardEulerTKE, FT, (ℂᴾ = 0.1, production_scale = 1e-6), kw)

Adapt.@adapt_structure ForwardEulerTKE

@inline function tke_step(tke_update::ForwardEulerTKE, cp, c, dtime, tke_k, a_prod_sh, a_prod_bu, mix, Cee)
    FT = typeof(tke_k)
    production_scale = tke_update.production_scale # scale of the production for the clipping policy [m² s⁻³]

    tke_k = max(0, tke_k)
    a_diss = Cee / mix * tke_k^convert(FT, 1.5)
    tke_k = clip_max(cp, c.mintke, tke_k + dtime * (clip_max(cp, 0, a_prod_sh + a_prod_bu, production_scale) - a_diss), c.mintke)
    tke_k = clip_min(cp, tke_k, c.maxtke, c.mintke)

    return tke_k, a_diss, nothing, tke_k
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

@inline function store_tke_sink_rate!(scratch, i, j, k, sink_rate)
    @inbounds scratch.tke_sink_rate[i, j, k] = sink_rate
    return nothing
end

# The fields a component needs beside those every configuration has
component_fields(grid, component) = NamedTuple()

#####
##### Sources explicit, sinks implicit
#####

"""
$(TYPEDEF)

An update of turbulent kinetic energy that keeps it positive for any time step without a floor,
as [`TKEBasedTurbulenceClosure`](@ref) does: the sources are explicit and the sinks are linear in
`e` and implicit, in the same vertical solve as the diffusion,

    (1 + Δt ω - Δt ∂z Kᶜ ∂z) eⁿ⁺¹ = e + Δt (P + B⁺),    ω = 𝕊ᴰ √e / ℓ + (-B⁻) / max(e, eᵐⁱⁿ)

where `B⁺` and `B⁻` are the positive and the negative part of the buoyancy production, taken
through the clipping policy. The dissipation rate `𝕊ᴰ √e / ℓ` times `e` is the dissipation
`𝕊ᴰ e^{3/2} / ℓ` of [`ForwardEulerTKE`](@ref) with `√e` of the old state, and a negative
buoyancy production removes the fraction of `e` it would remove explicitly. The time scale and
the diffusivities of the step see the estimate `(e + Δt (P + B⁺)) / (1 + Δt ω)`.

$(TYPEDFIELDS)
"""
struct ImplicitSinkTKE{FT} <: AbstractSHOCComponent
    "Shear production coefficient (`Ck_sh`)"
    ℂᴾ :: FT
    "Magnitude of the production that scales the smoothing of its positive part, m² s⁻³; `HardClipping` ignores it"
    production_scale :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ImplicitSinkTKE{FT}` with the shear production coefficient of E3SM `shoc.F90`. Every field is a
keyword argument.
"""
ImplicitSinkTKE(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ImplicitSinkTKE, FT, (ℂᴾ = 0.1, production_scale = 1e-6), kw)

Adapt.@adapt_structure ImplicitSinkTKE

component_fields(grid, ::ImplicitSinkTKE) = (; tke_sink_rate = CenterField(grid))

@inline function tke_step(tke_update::ImplicitSinkTKE, cp, c, dtime, tke_k, a_prod_sh, a_prod_bu, mix, Cee)
    FT = typeof(tke_k)
    production_scale = tke_update.production_scale # scale of the production for the clipping policy [m² s⁻³]

    # The buoyancy production in its positive part, a source, and its negative part, a sink
    buoyancy_source = clip_max(cp, 0, a_prod_bu, production_scale)
    buoyancy_sink = buoyancy_source - a_prod_bu

    tke_floored = max(c.mintke, tke_k)
    sink_rate = Cee * sqrt(tke_floored) / mix + buoyancy_sink / tke_floored

    tke_source = tke_k + dtime * (a_prod_sh + buoyancy_source)
    tke_estimate = tke_source / (1 + dtime * sink_rate)
    a_diss = Cee / mix * tke_estimate^convert(FT, 1.5)

    return tke_estimate, a_diss, sink_rate, tke_source
end

# θˡ and qᵗ share a factorization; the turbulent kinetic energy has its own, with the sink on
# the diagonal
@inline function implicit_scalar_solve!(::ImplicitSinkTKE, i, j, Nz, dtime, c, cp, host, diagnostics, scratch,
                                        thetal, qw, tke)
    zt, zi, pdel = host.zt_grid, host.zi_grid, host.pdel
    du, dl, d = scratch.du, scratch.dl, scratch.d
    no_surface_flux = zero(eltype(tke))

    vd_shoc_decomp!(i, j, Nz, dtime, c, zt, zi, pdel, diagnostics.tkh, no_surface_flux, du, dl, d)
    vd_shoc_solve!(i, j, Nz, du, dl, d, (thetal, qw))

    vd_shoc_decomp!(i, j, Nz, dtime, c, zt, zi, pdel, diagnostics.tkh, no_surface_flux, du, dl, d, scratch.tke_sink_rate)
    vd_shoc_solve!(i, j, Nz, du, dl, d, tke)

    return nothing
end
