#####
##### The return-to-isotropy time scale of SHOC as stability functions of `TKEBasedTurbulenceClosure`
#####
##### SHOC's eddy diffusivities `Kᶜ = ℂᶜ τ e`, `Kᵘ = ℂᵘ τ e` and dissipation `ε = 𝕊ᴰ e^{3/2} / ℓ` are those of
##### the TKE closure, `K = 𝕊 ℓ √e`, with the stability functions `𝕊ᶜ = ℂᶜ τ √e / ℓ`, `𝕊ᵘ = ℂᵘ τ √e / ℓ` and the
##### constant `𝕊ᴰ`. The time scale is damped by one number of the column, `λ`, from the stability
##### integrated over the lower column; the closure stores it in `closure_fields.column_stability`.
#####

"""
$(TYPEDEF)

The return-to-isotropy time scale and eddy diffusivities of E3SM `shoc.F90` (`isotropic_ts`,
`eddy_diffusivities`, the dissipation of `adv_sgs_tke`) as stability functions of
[`TKEBasedTurbulenceClosure`](@ref): with `τ₀ = 2 e / ε = 2 ℓ / (𝕊ᴰ √e)` the time scale of the dissipation,
damped in stable air by the stability of the whole column,

    τ = τᵈ / √(1 + (τᵈ / τᵐᵃˣ)²),    τᵈ = τ₀ / (1 + λ N²⁺ τ₀²),

the diffusivities are `Kᶜ = ℂᶜ τ e` (scalars and turbulent kinetic energy, which `shoc.F90` diffuses with the
heat diffusivity) and `Kᵘ = ℂᵘ τ e`, that is, the stability functions

    𝕊ᶜ = 𝕊ᵉ = ℂᶜ τ √e / ℓ,    𝕊ᵘ = ℂᵘ τ √e / ℓ,    𝕊ᴰ = (ℂᴰ₃ + ℂᴰ₄) / ℂᴰ₅ × ℂᴰ₁³ / ℂᴰ₂⁴.

The damping `λ = ℂᵝ₁ + (∫ N² dz / g - ℂᵝ₄) ℂᵝ₃`, kept between `ℂᵝ₁` and `ℂᵝ₂`, is one number per column, with
the static stability integrated over the levels whose pressure exceeds `stability_integration_pressure`.
`shoc.F90` bounds the time scale with `min(maxiso, ⋅)`; here the upper bound `τᵐᵃˣ` (`maximum_time_scale`)
enters as a sum of inverse squares, which changes the time scale by the fraction `(τ / τᵐᵃˣ)² / 2` where it is
far below it. The cold-surface fallback of `shoc.F90` (diffusivities `ℂ ℓ² |∂z u|` over a surface below 182 K)
is not included.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂᵘ`  | `Ckm`            | 0.1    | eddy diffusivity of momentum, `Kᵘ = ℂᵘ τ e` |
| `ℂᶜ`  | `Ckh`            | 0.1    | eddy diffusivity of heat, scalars and turbulent kinetic energy, `Kᶜ = ℂᶜ τ e` |
| `ℂᵝ₁` | `lambda_low`     | 0.001  | lowest value of the stability damping `λ` |
| `ℂᵝ₂` | `lambda_high`    | 0.08   | highest value of `λ` |
| `ℂᵝ₃` | `lambda_slope`   | 2.65   | slope of `λ` in the integrated column stability |
| `ℂᵝ₄` | `lambda_thresh`  | 0.02   | integrated column stability at which `λ` starts to rise |
| `ℂᴰ₁` | `Ck` (`adv_sgs_tke`) | 0.1  | dissipation, see `𝕊ᴰ` |
| `ℂᴰ₂` | `Cs` (`adv_sgs_tke`) | 0.15 | dissipation, see `𝕊ᴰ` |
| `ℂᴰ₃` | literal `0.19`   | 0.19   | dissipation, see `𝕊ᴰ` |
| `ℂᴰ₄` | literal `0.51`   | 0.51   | dissipation, see `𝕊ᴰ` |
| `ℂᴰ₅` | literal `0.7`    | 0.7    | dissipation, see `𝕊ᴰ` |

The five `ℂᴰ` enter only through the product `𝕊ᴰ`, so a calibration can determine no more than one of them.

$(TYPEDFIELDS)
"""
struct ColumnStabilityDamping{FT}
    "Eddy diffusivity coefficient for momentum (`Ckm`)"
    ℂᵘ :: FT
    "Eddy diffusivity coefficient for heat, scalars and turbulent kinetic energy (`Ckh`)"
    ℂᶜ :: FT
    "Lowest stability damping of the return-to-isotropy time scale (`lambda_low`)"
    ℂᵝ₁ :: FT
    "Highest stability damping (`lambda_high`)"
    ℂᵝ₂ :: FT
    "Slope of the stability damping (`lambda_slope`)"
    ℂᵝ₃ :: FT
    "Column stability at which the stability damping rises (`lambda_thresh`)"
    ℂᵝ₄ :: FT
    "Dissipation coefficient (`Ck` of `adv_sgs_tke`)"
    ℂᴰ₁ :: FT
    "Dissipation coefficient (`Cs` of `adv_sgs_tke`)"
    ℂᴰ₂ :: FT
    "Dissipation coefficient (literal `0.19` of `Ce1`)"
    ℂᴰ₃ :: FT
    "Dissipation coefficient (literal `0.51` of `Ce2`)"
    ℂᴰ₄ :: FT
    "Dissipation coefficient (literal `0.7` that divides `Ce` in `Ce1` and `Ce2`)"
    ℂᴰ₅ :: FT
    "Largest return-to-isotropy time scale, s (`maxiso`), in the sum of inverse squares"
    maximum_time_scale :: FT
    "Pressure above which the static stability is integrated over the column, Pa (`troppres`)"
    stability_integration_pressure :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ColumnStabilityDamping{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword argument.
"""
ColumnStabilityDamping(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ColumnStabilityDamping, FT,
                            (ℂᵘ = 0.1, ℂᶜ = 0.1, ℂᵝ₁ = 0.001, ℂᵝ₂ = 0.08, ℂᵝ₃ = 2.65, ℂᵝ₄ = 0.02,
                             ℂᴰ₁ = 0.1, ℂᴰ₂ = 0.15, ℂᴰ₃ = 0.19, ℂᴰ₄ = 0.51, ℂᴰ₅ = 0.7,
                             maximum_time_scale = 20000, stability_integration_pressure = 80000), kw)

Adapt.@adapt_structure ColumnStabilityDamping

TurbulenceClosures.convert_eltype(::Type{FT}, sf::ColumnStabilityDamping) where FT =
    map_scalars(scalar -> convert(FT, scalar), sf)

Base.summary(sf::ColumnStabilityDamping{FT}) where FT = "ColumnStabilityDamping{$FT}"

function TurbulenceClosures.show_stability_function_lines(io::IO, sf::ColumnStabilityDamping, prefix)
    print(io, prefix, "├── ℂᵘ: ", prettysummary(sf.ℂᵘ), ", ℂᶜ: ", prettysummary(sf.ℂᶜ), '\n',
              prefix, "├── ℂᵝ₁: ", prettysummary(sf.ℂᵝ₁), ", ℂᵝ₂: ", prettysummary(sf.ℂᵝ₂),
                      ", ℂᵝ₃: ", prettysummary(sf.ℂᵝ₃), ", ℂᵝ₄: ", prettysummary(sf.ℂᵝ₄), '\n',
              prefix, "└── 𝕊ᴰ: ", prettysummary(shoc_dissipation_coefficient(sf)))
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The dissipation coefficient `𝕊ᴰ = (ℂᴰ₃ + ℂᴰ₄) / ℂᴰ₅ × ℂᴰ₁³ / ℂᴰ₂⁴` of `shoc.F90` (`Cee`).
"""
@inline function shoc_dissipation_coefficient(sf::ColumnStabilityDamping)
    Ce = sf.ℂᴰ₁ * sf.ℂᴰ₁ * sf.ℂᴰ₁ / ((sf.ℂᴰ₂ * sf.ℂᴰ₂) * (sf.ℂᴰ₂ * sf.ℂᴰ₂))
    return Ce / sf.ℂᴰ₅ * sf.ℂᴰ₃ + Ce / sf.ℂᴰ₅ * sf.ℂᴰ₄
end

const ColumnStabilityDampingClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:ColumnStabilityDamping}

@inline TurbulenceClosures.dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::ColumnStabilityDampingClosure, args...) =
    shoc_dissipation_coefficient(closure.stability_functions)

@inline TurbulenceClosures.dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ColumnStabilityDampingClosure, args...) =
    shoc_dissipation_coefficient(closure.stability_functions)

# `τ √e / ℓ` from the stability damping `λ` of the column, the length `ℓ`, the turbulent velocity `u` and
# the static stability `N²`: τ₀ √e / ℓ = 2 / 𝕊ᴰ damped, then bounded by τᵐᵃˣ. Finite at ℓ = 0.
@inline function isotropy_ratio(sf::ColumnStabilityDamping, λ, ℓ, u, N²)
    s₀ = 2 / shoc_dissipation_coefficient(sf)
    τ₀ = s₀ * ℓ / u
    s = s₀ / (1 + λ * max(N², zero(N²)) * τ₀ * τ₀)
    τᵈ = s * ℓ / u
    x = τᵈ / sf.maximum_time_scale
    return s / sqrt(1 + x * x)
end

@inline function isotropy_ratioᶜᶜᶠ(i, j, k, grid, closure, closure_fields, ℓ, u)
    λ = @inbounds closure_fields.column_stability[i, j, 1]
    N² = @inbounds closure_fields.N²[i, j, k]
    return isotropy_ratio(closure.stability_functions, λ, ℓ, u, N²)
end

@inline function TurbulenceClosures.diffusivity_stability_functionsᶜᶜᶠ(i, j, k, grid, closure::ColumnStabilityDampingClosure,
                                                                       velocities, e, closure_fields)
    ℓ = @inbounds closure_fields.ℓᵐ[i, j, k]
    u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
    s = isotropy_ratioᶜᶜᶠ(i, j, k, grid, closure, closure_fields, ℓ, u)
    sf = closure.stability_functions
    return sf.ℂᵘ * s, sf.ℂᶜ * s, sf.ℂᶜ * s
end

# The return-to-isotropy time scale τ = (τ √e / ℓ) ℓ / √e at a face, for the moments
@inline TurbulenceClosures.return_to_isotropy_time_scaleᶜᶜᶠ(i, j, k, grid, ::ColumnStabilityDamping, closure, closure_fields, ℓ, u) =
    isotropy_ratioᶜᶜᶠ(i, j, k, grid, closure, closure_fields, ℓ, u) * ℓ / u

#####
##### The stability damping of the column
#####

TurbulenceClosures.column_stability_field(grid, ::ColumnStabilityDamping) = Field{Center, Center, Nothing}(grid)

# λ = ℂᵝ₁ + (∫ N² dz / g - ℂᵝ₄) ℂᵝ₃ between ℂᵝ₁ and ℂᵝ₂, with N² at the centers from the two adjacent faces,
# summed over the levels below the pressure `stability_integration_pressure`
@kernel function _compute_column_stability_damping!(λ, grid, closure, N², pressure, g)
    i, j = @index(Global, NTuple)
    sf = getclosure(i, j, closure).stability_functions
    FT = grid_float_type(grid)
    integral = zero(FT)
    for k in 1:size(grid, 3)
        N²ᶜ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
        lower = @inbounds pressure[i, j, k] > sf.stability_integration_pressure
        integral += ifelse(lower, N²ᶜ * Δzᶜᶜᶜ(i, j, k, grid), zero(FT))
    end
    λᶜ = sf.ℂᵝ₁ + (integral / g - sf.ℂᵝ₄) * sf.ℂᵝ₃
    @inbounds λ[i, j, 1] = clamp(λᶜ, sf.ℂᵝ₁, sf.ℂᵝ₂)
end

function TurbulenceClosures.compute_column_stability!(::ColumnStabilityDamping, λ, grid, closure, N², model)
    pressure = AtmosphereModels.dynamics_pressure(model.dynamics)
    g = model.thermodynamic_constants.gravitational_acceleration
    launch!(grid.architecture, grid, :xy, _compute_column_stability_damping!, λ, grid, closure, N², pressure, g)
    return nothing
end

#####
##### The rate norm of a limit of a time scale, for `StratificationLimitedMomentTimeScale`
#####

# (1 + xᵖ)^(1/p) for x ≥ 0
@inline rate_norm_factor(::Val{1}, x) = 1 + x
@inline rate_norm_factor(::Val{2}, x) = sqrt(1 + x * x)
@inline function rate_norm_factor(::Val{4}, x)
    x² = x * x
    return sqrt(sqrt(1 + x² * x²))
end
@inline rate_norm_factor(::Val{P}, x) where P = (1 + x^P)^inv(convert(typeof(x), P))
