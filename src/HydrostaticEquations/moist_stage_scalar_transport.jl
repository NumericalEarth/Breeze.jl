#####
##### Transport of the scalars of moist HydrostaticDynamics by the mass flux of a split-explicit stage
#####
#
# A stage of a split-explicit time stepper whose slow tendency enters alone (a diagonal `β`) and
# which recombines the increment of the stage before only, with the weight w ∈ (0, 1], changes the
# total surface pressure, and with it the layer mass ρ, by one mass flux:
#
#   ρⁱ = (1 - w) ρ⁰ + w ρⁱ⁻¹ - ∇·M ,
#   M  = Δtˢ [ρⁱ⁻¹ 𝐮ⁱ⁻¹ - ρᴸ 𝐮′ⁱ⁻¹] + ρᴸ ∫ 𝐮′ dτ ,
#
# with Δtˢ the length of the stage, ρⁱ⁻¹ 𝐮ⁱ⁻¹ the mass flux of the state the stage starts from,
# which its slow tendency holds, and ρᴸ 𝐮′ the mass flux of the velocity perturbation of the
# substeps on the frozen layer mass, whose time integral the substeps accumulate. The scalars are
# carried by the same mass flux, with the mass fraction per unit total mass of the state the stage
# starts from, r = ρc / ρ, reconstructed upwind of it:
#
#   (ρc)ⁱ = (1 - w) (ρc)⁰ + w [(ρc)ⁱ⁻¹ - Δtᵉ ∇·(M r̂ / Δtᵉ w)] + Δtˢ N ,     Δtᵉ = Δtˢ / w ,
#
# with N the tendencies other than transport. This is a convex combination of the state at the
# start of the time step and of one forward Euler step of length Δtᵉ from the state the stage
# starts from, with a mass flux that satisfies continuity with the change of the layer mass over
# that step. A transport scheme that keeps the mixing ratio within bounds over a forward Euler
# step (first-order upwind, bounds-preserving WENO within its Courant number, WENO with a flux
# limiter) therefore keeps it within them over the stage, and a uniform mixing ratio stays
# uniform. The limiters of the scheme are refreshed for exactly this step: its mass flux, its
# length and the mixing ratio it starts from.
#
# The mass flux is given to the transport operators as the velocity 𝐮̂ that carries it on the
# layer mass of the state the stage starts from, M = Δtˢ ρⁱ⁻¹ 𝐮̂, with the vertical component
# that continuity gives.

# The specific value of a prognostic density, r = ρc / ρ, per unit total mass. A cell without mass
# is a halo cell beyond a rigid wall, whose mixing ratio only ever multiplies the vanishing mass
# flux through the wall: it is zero rather than 0 / 0.
struct LayerMassFraction{FT, C, D} <: AbstractArray{FT, 3}
    weighted :: C  # ρc
    density :: D   # ρ, the total layer mass of the coordinate

    LayerMassFraction(ρc::C, ρ::D) where {C, D} = new{eltype(ρ), C, D}(ρc, ρ)
end

Base.size(r::LayerMassFraction) = size(r.density)

Base.@propagate_inbounds function Base.getindex(r::LayerMassFraction, i, j, k)
    ρ = r.density[i, j, k]
    massless = ρ == 0
    return ifelse(massless, zero(ρ), r.weighted[i, j, k] / ifelse(massless, one(ρ), ρ))
end

Adapt.adapt_structure(to, r::LayerMassFraction) = LayerMassFraction(adapt(to, r.weighted), adapt(to, r.density))

# Built on a multi-region grid by the seam budgets of the convex stages (`convex_stage_seam_budgets`), outside a
# region-by-region launch
@regional LayerMassFraction

#####
##### The tendencies of the scalars other than transport
#####

@kernel function _compute_scalar_forcing!(Gρc, grid, forcing, clock, model_fields)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρc[i, j, k] = forcing(i, j, k, grid, clock, model_fields)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Leave the transport of the scalars `names` of `model` out of their tendencies, which then hold
their forcing alone. Called once the tendencies of the surface pressure and of the potential
temperature density, which read the transport of the water, are complete; a time stepper that
transports the scalars with the mass flux of each of its stages asks for it through
[`leave_scalar_transport_to_stages!`](@ref).
"""
function withhold_scalar_transport!(model::MoistHydrostaticModel, names)
    grid = model.grid
    model_fields = Oceananigans.fields(model)

    for name in names
        launch!(grid.architecture, grid, :xyz, _compute_scalar_forcing!,
                model.timestepper.Gⁿ[name], grid, model.forcing[name], model.clock, model_fields)
    end

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Hook of the tendencies of moist `HydrostaticDynamics` for a `timestepper` that transports the
scalars itself, with the mass flux of its stages. Nothing by default: the tendencies hold the
transport by the mass flux of the state.
"""
leave_scalar_transport_to_stages!(timestepper, model) = nothing

#####
##### The velocity that carries the mass flux of a stage
#####

# 𝐮̂ = 𝐮ⁱ⁻¹ + (ρᴸ / ρⁱ⁻¹) ∫ (𝐮′ - 𝐮′ⁱ⁻¹) dτ / Δtˢ on the faces, in place of the integral
@kernel function _compute_stage_transport_velocities!(∫u′, ∫v′, grid, ρ, ρᴸ, u, v, Δtˢ⁻¹)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
        ρᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
        x_ratio = ℑxᶠᵃᵃ(i, j, k, grid, ρᴸ) / ifelse(ρᶠᶜᶜ == 0, one(ρᶠᶜᶜ), ρᶠᶜᶜ)
        y_ratio = ℑyᵃᶠᵃ(i, j, k, grid, ρᴸ) / ifelse(ρᶜᶠᶜ == 0, one(ρᶜᶠᶜ), ρᶜᶠᶜ)
        ∫u′[i, j, k] = u[i, j, k] + x_ratio * Δtˢ⁻¹ * ∫u′[i, j, k]
        ∫v′[i, j, k] = v[i, j, k] + y_ratio * Δtˢ⁻¹ * ∫v′[i, j, k]
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Replace the integrated velocity perturbations of `substepper`, net of the perturbation that the
slow tendency of the stage already carries, by the velocities that carry the mass flux of a stage
of length `Δtˢ` on the layer mass of the state the stage started from, which `model` still holds,
and diagnose the vertical component from continuity. Their halos are filled.
"""
function compute_stage_transport_velocities!(substepper::GravityWaveSubstepper, model, Δtˢ)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    ρ = dynamics.density
    ρᴸ = substepper.linearization.ρ
    u, v, _ = model.velocities
    ∫𝐮′ = substepper.integrated_velocities
    Δtˢ⁻¹ = convert(eltype(grid), 1 / Δtˢ)

    @apply_regionally launch!(arch, grid, :xyz, _compute_stage_transport_velocities!,
                              ∫𝐮′.u, ∫𝐮′.v, grid, ρ, ρᴸ, u, v, Δtˢ⁻¹)

    fill_halo_regions!((∫𝐮′.u, ∫𝐮′.v))

    @apply_regionally launch!(arch, grid, :xy, _compute_vertical_mass_flux!,
                              ∫𝐮′.ρw, ∫𝐮′.w, grid, dynamics.surface_pressure_coefficient, ρ, ∫𝐮′.u, ∫𝐮′.v)

    fill_halo_regions!(∫𝐮′.w)
    fill_coordinate_mass_flux_halos!(grid, ∫𝐮′.ρw)

    return nothing
end

#####
##### The stage of a scalar
#####

# In the units of the stage: per unit volume, or on a pressure-level grid the mass per unit area,
# with `ρc⁰` and `N` already in those units and `ρc` converted here (identities on other grids)
@kernel function _advance_scalar_with_stage_transport!(ρc⁺, grid, advection, ρ, 𝐮̂, r, ρc, ρc⁰, N, weight, Δtˢ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        G = N[i, j, k] - layer_mass_flux_divergence(i, j, k, grid, advection, ρ, 𝐮̂, r)
        ρc⁺[i, j, k] = (1 - weight) * ρc⁰[i, j, k] + weight * scalar_layer_mass(i, j, k, grid, ρc) + Δtˢ * G
    end
end

@kernel function _copy_interior!(a, b)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = b[i, j, k]
end

#####
##### Adaptive implicit vertical transport of the scalars of a stage (AIVA), on a pressure-level grid
#####
##### See derivations/hamip/plvd_aiva_derivation.md (lead, D494). In the forward Euler step of length Δtᵉ that a
##### convex stage holds, the vertical flux through a coordinate surface is split by the donor Courant number
##### C = |ρw̃| Δtᵉ / μ_donor (μ the layer mass Δp / g at the start of the stage) into an explicit part s ρw̃, with
##### s = min(1, cfl / C), transported by the scheme of the scalar, and an implicit part (1 - s) ρw̃, first-order
##### upwind and backward Euler over Δtᵉ:
#####
#####     μᴱₖ rₖ + Δtᵉ (Φₖ₊₁ - Φₖ) = Rₖ ,   Φₖ = (1 - sₖ) ρw̃ₖ r_donor(k) ,
#####
##### with R the explicit forward Euler result and μᴱ = μⁱ⁻¹ - Δtᵉ (D + δρw̃) the layer mass at its end, from the
##### same continuity that gives ρw̃. A uniform r stays uniform, the column content is that of R, and r is a convex
##### combination of R / μˣ (μˣ the mass of the explicit step) when μᴱ, μˣ > 0, at any Courant number of the
##### implicit part. The potential temperature and the momentum are not affected (open items of the derivation).

# The cfl of the adaptive implicit vertical transport of a scheme, or nothing (extended in Breeze.Advection)
stage_vertical_implicit_cfl(advection) = nothing
# The scheme of the explicit part (extended in Breeze.Advection)
stage_explicit_advection(advection) = advection

# Scratch of the split, one set per grid (a region on a multi-region grid).
# TODO: hold these in the GravityWaveSubstepper rather than in a cache keyed by grid.
const stage_aiva_scratch = IdDict{Any, Any}()
stage_aiva_fields(grid) = get!(() -> (ρw = ZFaceField(grid), w = ZFaceField(grid), s = ZFaceField(grid), γ = CenterField(grid)),
                               stage_aiva_scratch, grid)

@kernel function _split_stage_mass_flux!(sρw, sw, s, grid, ρw, w, Δtᵉ, cfl)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    @inbounds for k in 1:Nz+1
        F = ρw[i, j, k]
        donor = clamp(ifelse(F > 0, k - 1, k), 1, Nz)
        C = abs(F) * Δtᵉ / previous_layer_mass(i, j, donor, grid)
        sₖ = ifelse(C > cfl, cfl / C, one(C))
        s[i, j, k] = sₖ
        sρw[i, j, k] = sₖ * F
        sw[i, j, k] = sₖ * w[i, j, k]
    end
end

# The implicit part, solved per column from the explicit result of the stage Mⁱ (masses per unit area), in place
@kernel function _implicit_stage_vertical_transport!(Mⁱ, γ, grid, M⁰, ρ, u, v, ρw, s, weight, Δtᵉ)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    @inbounds begin
        # Forward sweep of the tridiagonal system, with Mⁱ holding the right-hand sides and then the solution
        β = one(Δtᵉ)
        for k in 1:Nz
            μᴱ = previous_layer_mass(i, j, k, grid) -
                 Δtᵉ * (horizontal_mass_flux_divergence(i, j, k, grid, ρ, u, v) + ρw[i, j, k+1] - ρw[i, j, k])
            a⁻ = (1 - s[i, j, k]) * ρw[i, j, k]       # through the lower face
            a⁺ = (1 - s[i, j, k+1]) * ρw[i, j, k+1]   # through the upper face
            lower = - Δtᵉ * max(a⁻, 0)                # r of the layer below enters with upward flow
            diagonal = μᴱ + Δtᵉ * max(a⁺, 0) + Δtᵉ * max(-a⁻, 0)
            R = (Mⁱ[i, j, k] - (1 - weight) * M⁰[i, j, k]) / weight
            if k == 1
                β = diagonal
                Mⁱ[i, j, k] = R / β
            else
                upper_previous = - Δtᵉ * max(-a⁻, 0)  # the upper coefficient of row k - 1
                γ[i, j, k] = upper_previous / β
                β = diagonal - lower * γ[i, j, k]
                Mⁱ[i, j, k] = (R - lower * Mⁱ[i, j, k-1]) / β
            end
        end
        for k in Nz-1:-1:1
            Mⁱ[i, j, k] -= γ[i, j, k+1] * Mⁱ[i, j, k+1]
        end
        # Back to the masses of the stage: Mⁱ = (1 - w) M⁰ + w μᴱ r
        for k in 1:Nz
            μᴱ = previous_layer_mass(i, j, k, grid) -
                 Δtᵉ * (horizontal_mass_flux_divergence(i, j, k, grid, ρ, u, v) + ρw[i, j, k+1] - ρw[i, j, k])
            Mⁱ[i, j, k] = (1 - weight) * M⁰[i, j, k] + weight * μᴱ * Mⁱ[i, j, k]
        end
    end
end

function launch_scalar_stage_transport!(ρc⁺, ρc, ρc⁰, N, model, substepper, name, weight, Δtˢ, Δtᵉ)
    grid = model.grid
    arch = grid.architecture
    FT = eltype(grid)
    advection = model.advection[name]
    ρ = model.dynamics.density
    ∫𝐮′ = substepper.integrated_velocities
    𝐮̂ = coordinate_transport_velocities(grid, ∫𝐮′.u, ∫𝐮′.v, ∫𝐮′.w, ∫𝐮′.ρw)
    r = LayerMassFraction(ρc, ρ)

    # The limiters of the scheme, for the forward Euler step of the stage
    update_bounds_preserving_limiter!(advection, grid, r, 𝐮̂, Δtᵉ)
    update_flux_limiter!(advection, grid, r, 𝐮̂, convert(FT, Δtᵉ), ρ)

    launch!(arch, grid, :xyz, _advance_scalar_with_stage_transport!,
            ρc⁺, grid, advection, ρ, 𝐮̂, r, ρc, ρc⁰, N, convert(FT, weight), convert(FT, Δtˢ))

    return nothing
end

launch_split_stage_mass_flux!(scratch, grid, ∫𝐮′, Δtᵉ, cfl) =
    launch!(grid.architecture, grid, :xy, _split_stage_mass_flux!, scratch.ρw, scratch.w, scratch.s, grid, ∫𝐮′.ρw, ∫𝐮′.w,
            convert(eltype(grid), Δtᵉ), convert(eltype(grid), cfl))

function launch_implicit_scalar_stage_transport!(scratch, ρc⁺, ρc, ρc⁰, N, model, substepper, name, weight, Δtˢ, Δtᵉ)
    grid = model.grid
    arch = grid.architecture
    FT = eltype(grid)
    advection = stage_explicit_advection(model.advection[name])
    ρ = model.dynamics.density
    ∫𝐮′ = substepper.integrated_velocities

    𝐮̂ˢ = coordinate_transport_velocities(grid, ∫𝐮′.u, ∫𝐮′.v, scratch.w, scratch.ρw)
    r = LayerMassFraction(ρc, ρ)

    # The limiters of the scheme, for the explicit part of the forward Euler step of the stage
    update_bounds_preserving_limiter!(advection, grid, r, 𝐮̂ˢ, Δtᵉ)
    update_flux_limiter!(advection, grid, r, 𝐮̂ˢ, convert(FT, Δtᵉ), ρ)

    launch!(arch, grid, :xyz, _advance_scalar_with_stage_transport!,
            ρc⁺, grid, advection, ρ, 𝐮̂ˢ, r, ρc, ρc⁰, N, convert(FT, weight), convert(FT, Δtˢ))

    launch!(arch, grid, :xy, _implicit_stage_vertical_transport!, ρc⁺, scratch.γ, grid, ρc⁰, ρ,
            ∫𝐮′.u, ∫𝐮′.v, ∫𝐮′.ρw, scratch.s, convert(FT, weight), convert(FT, Δtᵉ))

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Advance the prognostic scalar `name` of `model` over a stage of length `Δtˢ`,

```math
(ρc)ⁱ = (1 - w) (ρc)⁰ + w (ρc)ⁱ⁻¹ + Δtˢ [N - ∇·(ρ 𝐮̂ r̂)] ,
```

with `ρc⁰` its value at the start of the time step, `weight` ``w``, `N` its tendencies other than
transport, ``𝐮̂`` the velocities that [`compute_stage_transport_velocities!`](@ref) left in
`substepper`, and ``r̂`` the reconstruction of its mass fraction per unit total mass by its advection
scheme, whose limiters are refreshed for the forward Euler step of length `Δtᵉ` ``= Δtˢ / w`` that
the stage holds. `model` must hold the state the stage started from. The new state is left in
`scratch`, from which [`complete_scalar_stage_transport!`](@ref) sets `ρc` once the surface
pressure of the stage is complete.
"""
function advance_scalar_with_stage_transport!(ρc, ρc⁰, scratch, N, model::MoistHydrostaticModel,
                                              substepper::GravityWaveSubstepper, name, weight, Δtˢ, Δtᵉ)
    cfl = stage_vertical_implicit_cfl(model.advection[name])
    if isnothing(cfl)
        @apply_regionally launch_scalar_stage_transport!(scratch, ρc, ρc⁰, N, model, substepper, name, weight, Δtˢ, Δtᵉ)
    else
        grid = model.grid
        has_pressure_levels(grid) ||
            throw(ArgumentError("The adaptive implicit vertical transport of the convex stages requires a PressureLevelGrid."))
        # The split of the stage mass flux, once per grid and with its halos, then the stage of the scalar
        split = stage_aiva_fields(grid)
        ∫𝐮′ = substepper.integrated_velocities
        @apply_regionally launch_split_stage_mass_flux!(split, grid, ∫𝐮′, Δtᵉ, cfl)
        fill_halo_regions!(split.w)
        fill_halo_regions!(split.ρw)
        @apply_regionally launch_implicit_scalar_stage_transport!(split, scratch, ρc, ρc⁰, N, model, substepper, name,
                                                                  weight, Δtˢ, Δtᵉ)
    end
    return nothing
end

@kernel function _set_from_layer_mass!(ρc, grid, M)
    i, j, k = @index(Global, NTuple)
    @inbounds ρc[i, j, k] = scalar_density_from_layer_mass(i, j, k, grid, M[i, j, k])
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Set the prognostic scalar `ρc` from the state `scratch` that
[`advance_scalar_with_stage_transport!`](@ref) left: a copy, or on a pressure-level grid the
density of the mass per unit area at the surface pressure of the end of the stage.
"""
function complete_scalar_stage_transport!(ρc, scratch)
    grid = ρc.grid
    @apply_regionally launch_scalar_from_layer_mass!(grid, ρc, scratch)
    return nothing
end

launch_scalar_from_layer_mass!(grid, ρc, scratch) = launch!(grid.architecture, grid, :xyz, _copy_interior!, ρc, scratch)
launch_scalar_from_layer_mass!(grid::PressureLevelGrid, ρc, M) = launch!(grid.architecture, grid, :xyz, _set_from_layer_mass!, ρc, grid, M)
