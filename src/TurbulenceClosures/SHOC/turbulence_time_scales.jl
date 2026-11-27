"""
$(TYPEDEF)

The return-to-isotropy time scale and eddy diffusivities of E3SM `shoc.F90` (`isotropic_ts`,
`eddy_diffusivities`, the dissipation of `adv_sgs_tke`) as stability functions of
[`TKEBasedTurbulenceClosure`](@ref): with `τ₀ = 2 e / ε = 2 ℓ / (𝕊ᴰ √e)` the time scale of the dissipation,
damped in stable air by the stability of the whole column and in unstable air by the buoyancy frequency,
"""
struct ColumnStabilityDamping{FT}
    ℂᵘ :: FT
    ℂᶜ :: FT
    ℂᵝ₁ :: FT
    ℂᵝ₂ :: FT
    ℂᵝ₃ :: FT
    ℂᵝ₄ :: FT
    ℂᴰ₁ :: FT
    ℂᴰ₂ :: FT
    ℂᴰ₃ :: FT
    ℂᴰ₄ :: FT
    ℂᴰ₅ :: FT
    maximum_time_scale :: FT
    stability_integration_pressure :: FT
    ℂᴾ :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ColumnStabilityDamping{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword argument.
"""
ColumnStabilityDamping(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ColumnStabilityDamping, FT,
                            (ℂᵘ = 0.1, ℂᶜ = 0.1, ℂᵝ₁ = 0.001, ℂᵝ₂ = 0.08, ℂᵝ₃ = 2.65, ℂᵝ₄ = 0.02,
                             ℂᴰ₁ = 0.1, ℂᴰ₂ = 0.15, ℂᴰ₃ = 0.19, ℂᴰ₄ = 0.51, ℂᴰ₅ = 0.7,
                             maximum_time_scale = 20000, stability_integration_pressure = 80000, ℂᴾ = 0.1), kw)

Adapt.@adapt_structure ColumnStabilityDamping

TurbulenceClosures.convert_eltype(::Type{FT}, damping::ColumnStabilityDamping) where FT =
    map_scalars(scalar -> convert(FT, scalar), damping)

Base.summary(damping::ColumnStabilityDamping{FT}) where FT = "ColumnStabilityDamping{$FT}"

function TurbulenceClosures.show_stability_function_lines(io::IO, damping::ColumnStabilityDamping, prefix)
    print(io, prefix, "├── ℂᵘ: ", prettysummary(damping.ℂᵘ), ", ℂᶜ: ", prettysummary(damping.ℂᶜ), '\n',
              prefix, "├── ℂᵝ₁: ", prettysummary(damping.ℂᵝ₁), ", ℂᵝ₂: ", prettysummary(damping.ℂᵝ₂),
                      ", ℂᵝ₃: ", prettysummary(damping.ℂᵝ₃), ", ℂᵝ₄: ", prettysummary(damping.ℂᵝ₄), '\n',
              prefix, "├── 𝕊ᴰ: ", prettysummary(shoc_dissipation_coefficient(damping)), '\n',
              prefix, "└── ℂᴾ: ", prettysummary(damping.ℂᴾ))
    return nothing
end

@inline function shoc_dissipation_coefficient(damping::ColumnStabilityDamping)
    dissipation_ratio = damping.ℂᴰ₁ * damping.ℂᴰ₁ * damping.ℂᴰ₁ / ((damping.ℂᴰ₂ * damping.ℂᴰ₂) * (damping.ℂᴰ₂ * damping.ℂᴰ₂))
    return dissipation_ratio / damping.ℂᴰ₅ * damping.ℂᴰ₃ + dissipation_ratio / damping.ℂᴰ₅ * damping.ℂᴰ₄
end

const ColumnStabilityDampingClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:ColumnStabilityDamping}

@inline TurbulenceClosures.shear_production_fraction(damping::ColumnStabilityDamping) = damping.ℂᴾ

@inline TurbulenceClosures.dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::ColumnStabilityDampingClosure, args...) =
    shoc_dissipation_coefficient(closure.stability_functions)

@inline TurbulenceClosures.dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ColumnStabilityDampingClosure, args...) =
    shoc_dissipation_coefficient(closure.stability_functions)

@inline function isotropy_ratio(damping::ColumnStabilityDamping, λ, ℓ, u, N²)
    s₀ = 2 / shoc_dissipation_coefficient(damping)
    τ₀ = s₀ * ℓ / u
    αᴺ = N² * τ₀ * τ₀
    α̃ᴺ = sqrt(αᴺ * αᴺ + 1)
    λ⁻ = damping.ℂᶜ * damping.ℂᶜ / 4
    s = s₀ / (1 + (λ * (α̃ᴺ + αᴺ) + λ⁻ * (α̃ᴺ - αᴺ)) / 2)
    τᵈ = s * ℓ / u
    x = τᵈ / damping.maximum_time_scale
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
    damping = closure.stability_functions
    return damping.ℂᵘ * s, damping.ℂᶜ * s, damping.ℂᶜ * s
end

@inline TurbulenceClosures.return_to_isotropy_time_scaleᶜᶜᶠ(i, j, k, grid, ::ColumnStabilityDamping, closure, closure_fields, ℓ, u) =
    isotropy_ratioᶜᶜᶠ(i, j, k, grid, closure, closure_fields, ℓ, u) * ℓ / u

#####
##### The stability damping of the column
#####

TurbulenceClosures.column_stability_field(grid, ::ColumnStabilityDamping) = Field{Center, Center, Nothing}(grid)

@inline function lower_troposphere_weight(p, pᴵ)
    r = pᴵ / p
    r² = r * r
    r⁴ = r² * r²
    r⁸ = r⁴ * r⁴
    return 1 / (1 + r⁸ * r⁸)
end

@inline function column_stability_damping(damping::ColumnStabilityDamping, I)
    Δλ = (damping.ℂᵝ₂ - damping.ℂᵝ₁) / 2
    x = ((I - damping.ℂᵝ₄) * damping.ℂᵝ₃ - Δλ) / Δλ
    x² = x * x
    R² = sqrt(1 + x² * x²)
    R = sqrt(R²)
    smaller = 1 / (2 * R * (R + abs(x)) * (R² + x²)) # the weight of the bound the line is farther from, in (0, 1/2]
    rising = x > 0
    return damping.ℂᵝ₁ * ifelse(rising, smaller, 1 - smaller) + damping.ℂᵝ₂ * ifelse(rising, 1 - smaller, smaller)
end

@kernel function _compute_column_stability_damping!(λ, grid, closure, N², pressure, g)
    i, j = @index(Global, NTuple)
    damping = getclosure(i, j, closure).stability_functions
    FT = grid_float_type(grid)
    integral = zero(FT)
    for k in 1:size(grid, 3)
        N²ᶜ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
        weight = lower_troposphere_weight(@inbounds(pressure[i, j, k]), damping.stability_integration_pressure)
        integral += weight * N²ᶜ * Δzᶜᶜᶜ(i, j, k, grid)
    end
    @inbounds λ[i, j, 1] = column_stability_damping(damping, integral / g)
end

function TurbulenceClosures.compute_column_stability!(::ColumnStabilityDamping, λ, grid, closure, N², model; parameters = :xyz)
    pressure = AtmosphereModels.dynamics_pressure(model.dynamics)
    g = model.thermodynamic_constants.gravitational_acceleration
    launch!(grid.architecture, grid, column_kernel_parameters(parameters), _compute_column_stability_damping!, λ, grid, closure, N², pressure, g)
    return nothing
end

#####
##### The rate norm of a limit of a time scale, for `StratificationLimitedMomentTimeScale`
#####

@inline rate_norm_factor(::Val{1}, x) = 1 + x
@inline rate_norm_factor(::Val{2}, x) = sqrt(1 + x * x)
@inline function rate_norm_factor(::Val{4}, x)
    x² = x * x
    return sqrt(sqrt(1 + x² * x²))
end
@inline rate_norm_factor(::Val{P}, x) where P = (1 + x^P)^inv(convert(typeof(x), P))
