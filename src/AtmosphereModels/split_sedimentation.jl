#####
##### T21-precip: operator-split column sedimentation of the precipitating condensates
#####
##### ANSWER-CHANGING when on. Default off (bitwise unchanged).
#####
##### `BREEZE_SEDIMENTATION_SUBSTEP` (read in `__init__`): the fall-speed part of the precipitation
##### transport leaves the RK stages. The scalar tendencies advect every condensate with the air
##### velocity only (`WithoutSedimentation` wraps the microphysics for those launches) and the
##### thermodynamic variable's explicit sedimentation term is skipped. Once per outer step, after the
##### third stage, `split_sedimentation_step!` moves each sedimenting condensate down its column over the
##### full Δt with first-order upwind fluxes at the fall speed of the last stage state, in an adaptive
##### number of explicit substeps per column, n = ceil(max_k C_k / SEDIMENTATION_CFL) with
##### C_k = |wˢ| Δt Az / V (capped at SEDIMENTATION_MAX_SUBSTEPS). The donor is max(ρq, 0), so a column
##### conserves mass exactly and no cell is driven negative by its own outflow. The condensate content
##### of the thermodynamic variable moves with the step-averaged mass fluxes through the existing
##### `sedimentation_content_tendency` (content and enthalpy of the donor cell).
#####
##### Limits: only the acoustic RK3 stepper calls the split step; the fall speed is frozen over the
##### substeps; `bottom_precipitation_flux` no longer sees the split flux; adaptive implicit vertical
##### advection of the condensates is not combined with it.
#####

const SEDIMENTATION_SUBSTEP = Ref(false)
const SEDIMENTATION_CFL = 0.9
const SEDIMENTATION_MAX_SUBSTEPS = 64

#####
##### `BREEZE_SEDIMENTATION_IMPLICIT` (answer-changing, default off; takes precedence over SUBSTEP): the
##### same split, but one backward-Euler first-order upwind step over Δt per column, solved in a single
##### downward sweep (the matrix is lower bidiagonal): ρqₖⁿ⁺¹ = (ρqₖ + Δt Az₍ₖ₊₁₎ M₍ₖ₊₁₎ⁿ⁺¹ / V) / (1 + Cₖ),
##### Mₖⁿ⁺¹ = |wˢₖ| ρqₖⁿ⁺¹. Unconditionally stable and positivity preserving; a cell with a negative
##### total (inflow plus content) sends nothing.
const SEDIMENTATION_IMPLICIT = Ref(false)

split_sedimentation_mode() = SEDIMENTATION_IMPLICIT[] ? Val(:implicit) :
                             SEDIMENTATION_SUBSTEP[] ? Val(:substep) : nothing
split_sedimentation_active() = !isnothing(split_sedimentation_mode())

"Microphysics seen by the scalar tendencies when sedimentation is operator-split: no fall velocity."
struct WithoutSedimentation{M}
    microphysics :: M
end

Adapt.adapt_structure(to, m::WithoutSedimentation) = WithoutSedimentation(adapt(to, m.microphysics))

@inline microphysical_velocities(::WithoutSedimentation, microphysical_fields, name) = nothing

scalar_tendency_microphysics(microphysics) =
    split_sedimentation_active() ? WithoutSedimentation(microphysics) : microphysics

# Step-averaged split fluxes per unit face density [m³ s⁻¹], upward positive, in the
# `mass_fluxes` protocol of `condensate_content_fluxes`: ((F⁻(W), F⁻(wᵗ)), (F⁺(W), F⁺(wᵗ))), wᵗ = 0.
struct SplitSedimentationFluxes{F}
    flux :: F
end

Adapt.adapt_structure(to, s::SplitSedimentationFluxes) = SplitSedimentationFluxes(adapt(to, s.flux))

@inline function (s::SplitSedimentationFluxes)(i, j, k, grid, condensate, wᵗ, ρ)
    z = zero(grid)
    @inbounds F⁻ = s.flux[i, j, k]
    @inbounds F⁺ = s.flux[i, j, k + 1]
    return (F⁻, z), (F⁺, z)
end

const SPLIT_SEDIMENTATION_FLUX = IdDict{Any, Any}()

split_sedimentation_flux_field(grid) = get!(() -> ZFaceField(grid), SPLIT_SEDIMENTATION_FLUX, grid)

"""
$(TYPEDSIGNATURES)

Operator-split sedimentation of every condensate in `model.sedimentation` over `Δt` (see the
header of `split_sedimentation.jl`). A no-op unless a split-sedimentation switch is on.
"""
function split_sedimentation_step!(model, Δt)
    mode = split_sedimentation_mode()
    isnothing(mode) && return nothing
    condensates = values(model.sedimentation)
    isempty(condensates) && return nothing
    grid = model.grid
    arch = grid.architecture
    FT = eltype(grid)
    F = split_sedimentation_flux_field(grid)
    ρ = total_density(model.dynamics)
    φ = thermodynamic_density(model.formulation)
    temperature = explicit_content_source(model)
    zero_w = ZeroField(FT)
    for condensate in condensates
        launch!(arch, grid, :xy, _split_sedimentation_column!, F, condensate.density, condensate.velocity, ρ, grid,
                kernel_time_step(arch, grid, Δt), mode, FT(SEDIMENTATION_CFL), SEDIMENTATION_MAX_SUBSTEPS)
        launch!(arch, grid, :xyz, _split_sedimentation_content!, φ, grid, kernel_time_step(arch, grid, Δt),
                (condensate,), zero_w, SplitSedimentationFluxes(F), model.formulation, model.dynamics,
                model.thermodynamic_constants, model.microphysics, model.microphysical_fields,
                specific_prognostic_moisture(model), temperature)
    end
    return nothing
end

@kernel function _split_sedimentation_content!(φ, grid, Δt, condensates, wᵗ, mass_fluxes, formulation, dynamics,
                                               constants, microphysics, microphysical_fields,
                                               specific_prognostic_moisture, temperature)
    i, j, k = @index(Global, NTuple)
    tendency = sedimentation_content_tendency(i, j, k, grid, condensates, wᵗ, mass_fluxes, formulation, dynamics,
                                              constants, microphysics, microphysical_fields,
                                              specific_prognostic_moisture, temperature)
    @inbounds φ[i, j, k] += Δt * tendency
end

# Fall speed (positive) through face k, the bottom face of cell k, and the Courant number of cell k.
@inline split_fall_speed(i, j, k, w) = @inbounds -min(w[i, j, k], zero(eltype(w)))
@inline split_courant(i, j, k, grid, w, Δt) = Δt * split_fall_speed(i, j, k, w) * Azᶜᶜᶠ(i, j, k, grid) * V⁻¹ᶜᶜᶜ(i, j, k, grid)

# Store the step-averaged downward mass flux M̄ [kg m⁻² s⁻¹] of face k as −M̄ Az / ρᶠ.
@inline split_store_flux!(F, i, j, k, grid, ρ, M̄) =
    @inbounds F[i, j, k] = -M̄ * Azᶜᶜᶠ(i, j, k, grid) / ℑzᵃᵃᶠ(i, j, k, grid, ρ)

@kernel function _split_sedimentation_column!(F, ρq, w, ρ, grid, Δt, mode, cfl, max_substeps)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz
    FT = eltype(grid)
    @inbounds F[i, j, Nz + 1] = zero(FT)
    split_sedimentation_column!(F, ρq, w, ρ, grid, Δt, mode, cfl, max_substeps, i, j, Nz)
end

@inline function split_sedimentation_column!(F, ρq, w, ρ, grid, Δt, ::Val{:substep}, cfl, max_substeps, i, j, Nz)
    FT = eltype(grid)
    Cmax = zero(FT)
    for k in 1:Nz
        Cmax = max(Cmax, split_courant(i, j, k, grid, w, Δt))
    end
    n = clamp(ceil(Int, Cmax / cfl), 1, max_substeps)
    δt = Δt / n
    for k in 1:Nz
        @inbounds F[i, j, k] = zero(FT)  # accumulates Σ M over the substeps
    end
    for _ in 1:n
        M_above = zero(FT)
        for k in Nz:-1:1
            @inbounds ρqₖ = ρq[i, j, k]
            Mₖ = split_fall_speed(i, j, k, w) * max(ρqₖ, zero(FT))
            @inbounds ρq[i, j, k] = ρqₖ + δt * V⁻¹ᶜᶜᶜ(i, j, k, grid) *
                                    (Azᶜᶜᶠ(i, j, k + 1, grid) * M_above - Azᶜᶜᶠ(i, j, k, grid) * Mₖ)
            @inbounds F[i, j, k] += Mₖ
            M_above = Mₖ
        end
    end
    for k in 1:Nz
        @inbounds M̄ = F[i, j, k] / n
        split_store_flux!(F, i, j, k, grid, ρ, M̄)
    end
    return nothing
end

@inline function split_sedimentation_column!(F, ρq, w, ρ, grid, Δt, ::Val{:implicit}, cfl, max_substeps, i, j, Nz)
    FT = eltype(grid)
    M_above = zero(FT)
    for k in Nz:-1:1
        V⁻¹ = V⁻¹ᶜᶜᶜ(i, j, k, grid)
        @inbounds x = ρq[i, j, k] + Δt * V⁻¹ * Azᶜᶜᶠ(i, j, k + 1, grid) * M_above
        C = split_courant(i, j, k, grid, w, Δt)
        ρqⁿ⁺¹ = ifelse(x > 0, x / (1 + C), x)
        Mₖ = split_fall_speed(i, j, k, w) * max(ρqⁿ⁺¹, zero(FT))
        @inbounds ρq[i, j, k] = ρqⁿ⁺¹
        split_store_flux!(F, i, j, k, grid, ρ, Mₖ)
        M_above = Mₖ
    end
    return nothing
end
