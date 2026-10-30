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
    F = split_sedimentation_flux_field(grid)
    # Function barrier: the cached mass-only tuple is inferred as `Any`.
    return launch_split_sedimentation!(model, grid, F, kernel_time_step(grid.architecture, grid, Δt), mode,
                                       condensates, split_mass_only_prognostics(model))
end

# T21-sedfuse: every sedimenting prognostic of the step in ONE column kernel. Each thread owns a column
# and runs, in the order of the former per-species launches, the column sedimentation of each condensate
# followed by its content update of φ over the column, then the mass-only prognostics. Every read of the
# content update (ρq of all species, θ or the stored contents, T, ρ, F) is column-local and is made after
# the same writes as with the separate launches, and φ is not read, so the result is bitwise identical.
function launch_split_sedimentation!(model, grid, F, Δt, mode, condensates, mass_only)
    FT = eltype(grid)
    launch!(grid.architecture, grid, :xy, _split_sedimentation!, F, thermodynamic_density(model.formulation), grid, Δt,
            mode, FT(SEDIMENTATION_CFL), SEDIMENTATION_MAX_SUBSTEPS, total_density(model.dynamics), condensates,
            mass_only, ZeroField(FT), model.formulation, model.dynamics, model.thermodynamic_constants,
            model.microphysics, model.microphysical_fields, specific_prognostic_moisture(model),
            explicit_content_source(model))
    return nothing
end

# Sedimenting prognostics that carry no condensate content (P3 number, rime volume, ...), moved for
# mass only: a tuple of (ρc, wˢ) pairs, built once per microphysical field set (the per-call generator
# over the names was type unstable).
const SPLIT_MASS_ONLY = IdDict{Any, Any}()

split_mass_only_prognostics(model) = get!(SPLIT_MASS_ONLY, model.microphysical_fields) do
    μ = model.microphysical_fields
    names = Tuple(n for n in prognostic_field_names(model.microphysics) if !haskey(model.sedimentation, n))
    pairs = Tuple((getproperty(μ, n), sedimentation_velocity(model.microphysics, μ, Val(n))) for n in names)
    Tuple(p for p in pairs if !isnothing(p[2]))
end

@kernel function _split_sedimentation!(F, φ, grid, Δt, mode, cfl, max_substeps, ρ, condensates, mass_only, wᵗ,
                                       formulation, dynamics, constants, microphysics, microphysical_fields,
                                       specific_prognostic_moisture, temperature)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz
    @inbounds F[i, j, Nz + 1] = zero(eltype(grid))
    split_condensate_columns!(F, φ, grid, Δt, mode, cfl, max_substeps, ρ, condensates, wᵗ, formulation, dynamics,
                              constants, microphysics, microphysical_fields, specific_prognostic_moisture,
                              temperature, i, j, Nz)
    split_mass_only_columns!(F, grid, Δt, mode, cfl, max_substeps, ρ, mass_only, i, j, Nz)
end

@inline split_condensate_columns!(F, φ, grid, Δt, mode, cfl, max_substeps, ρ, ::Tuple{}, args...) = nothing

@inline function split_condensate_columns!(F, φ, grid, Δt, mode, cfl, max_substeps, ρ, condensates::Tuple, wᵗ,
                                           formulation, dynamics, constants, microphysics, microphysical_fields,
                                           specific_prognostic_moisture, temperature, i, j, Nz)
    condensate = first(condensates)
    split_sedimentation_column!(F, condensate.density, condensate.velocity, ρ, grid, Δt, mode, cfl, max_substeps, i, j, Nz)
    split_content_column!(φ, F, grid, Δt, (condensate,), wᵗ, formulation, dynamics, constants, microphysics,
                          microphysical_fields, specific_prognostic_moisture, temperature, i, j, Nz)
    split_condensate_columns!(F, φ, grid, Δt, mode, cfl, max_substeps, ρ, Base.tail(condensates), wᵗ,
                              formulation, dynamics, constants, microphysics, microphysical_fields,
                              specific_prognostic_moisture, temperature, i, j, Nz)
    return nothing
end

# φ += Δt × the content tendency of `condensates` from the split fluxes in F, up the column, carrying the
# contents of cells k − 1, k, k + 1 (the clamps of `sedimentation_content_tendency`).
@inline function split_content_column!(φ, F, grid, Δt, condensates, wᵗ, formulation, dynamics, constants,
                                       microphysics, microphysical_fields, specific_prognostic_moisture,
                                       temperature, i, j, Nz)
    mass_fluxes = SplitSedimentationFluxes(F)
    c⁰ = condensate_content(i, j, 1, grid, formulation, dynamics, constants,
                            microphysics, microphysical_fields, specific_prognostic_moisture, temperature)
    c⁻ = c⁰
    for k in 1:Nz
        c⁺ = condensate_content(i, j, min(k + 1, Nz), grid, formulation, dynamics, constants,
                                microphysics, microphysical_fields, specific_prognostic_moisture, temperature)
        tendency = sedimentation_content_tendency(i, j, k, grid, condensates, wᵗ, mass_fluxes, dynamics, c⁻, c⁰, c⁺)
        @inbounds φ[i, j, k] += Δt * tendency
        c⁻ = c⁰
        c⁰ = c⁺
    end
    return nothing
end

@inline split_mass_only_columns!(F, grid, Δt, mode, cfl, max_substeps, ρ, ::Tuple{}, i, j, Nz) = nothing

@inline function split_mass_only_columns!(F, grid, Δt, mode, cfl, max_substeps, ρ, mass_only::Tuple, i, j, Nz)
    ρc, w = first(mass_only)
    split_sedimentation_column!(F, ρc, w, ρ, grid, Δt, mode, cfl, max_substeps, i, j, Nz)
    split_mass_only_columns!(F, grid, Δt, mode, cfl, max_substeps, ρ, Base.tail(mass_only), i, j, Nz)
    return nothing
end

# Fall speed (positive) through face k, the bottom face of cell k, and the Courant number of cell k.
@inline split_fall_speed(i, j, k, w) = @inbounds -min(w[i, j, k], zero(eltype(w)))
@inline split_courant(i, j, k, grid, w, Δt) = Δt * split_fall_speed(i, j, k, w) * Azᶜᶜᶠ(i, j, k, grid) * V⁻¹ᶜᶜᶜ(i, j, k, grid)

# Store the step-averaged downward mass flux M̄ [kg m⁻² s⁻¹] of face k as −M̄ Az / ρᶠ.
@inline split_store_flux!(F, i, j, k, grid, ρ, M̄) =
    @inbounds F[i, j, k] = -M̄ * Azᶜᶜᶠ(i, j, k, grid) / ℑzᵃᵃᶠ(i, j, k, grid, ρ)

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
