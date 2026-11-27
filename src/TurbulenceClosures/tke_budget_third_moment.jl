"""
$(TYPEDEF)

The third moment of vertical velocity at the cell faces from the vertical integral of the budget of turbulent kinetic
energy, with a damped local member carried by the latent source of partly cloudy layers.
"""
struct TKEBudgetThirdMoment{FT} <: AbstractSubgridDistributionComponent
    ℂʷᵉ :: FT
    ℂʷᵋ :: FT
    ℂʷᴸ :: FT
    ℂʷʳ :: FT
    ℂʷˡ :: FT
    ℂʷᴺ :: FT
    ℂʷᵀ :: FT
    stratification_damping_width :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`TKEBudgetThirdMoment`](@ref).
"""
function TKEBudgetThirdMoment(FT::DataType = Oceananigans.defaults.FloatType;
                              ℂʷᵉ = 0.12, ℂʷᵋ = 0.34, ℂʷᴸ = 8, ℂʷʳ = 0, ℂʷˡ = 8, ℂʷᴺ = 0, ℂʷᵀ = 0,
                              stratification_damping_width = 1)
    abs(ℂʷʳ) < 1 || throw(ArgumentError("ℂʷʳ must lie in (-1, 1) for a positive ratio, got $(ℂʷʳ)."))
    0 <= ℂʷᵀ <= 1 || throw(ArgumentError("ℂʷᵀ must lie in [0, 1], got $(ℂʷᵀ)."))
    ℂʷᴺ >= 0 || throw(ArgumentError("ℂʷᴺ must not be negative (the damping must not vanish), got $(ℂʷᴺ)."))
    0 < ℂʷᴸ < Inf || throw(ArgumentError("ℂʷᴸ must be positive and finite, got $(ℂʷᴸ)."))
    stratification_damping_width > 0 ||
        throw(ArgumentError("stratification_damping_width must be positive, got $(stratification_damping_width)."))
    scalars = (ℂʷᵉ, ℂʷᵋ, ℂʷᴸ, ℂʷʳ, ℂʷˡ, ℂʷᴺ, ℂʷᵀ, stratification_damping_width)
    return TKEBudgetThirdMoment(map(c -> convert(FT, c), scalars)...)
end

Adapt.@adapt_structure TKEBudgetThirdMoment

const TKEBudgetDistribution = DoubleGaussian{<:Any, <:Any, <:TKEBudgetThirdMoment}

third_moment_fields(grid, third_moment) = NamedTuple()

third_moment_fields(grid, tm::TKEBudgetThirdMoment) =
    (Fᵉ = CenterField(grid), Fᵋ = CenterField(grid), uᵉ = CenterField(grid), uᵋ = CenterField(grid), εʷ = CenterField(grid),
     w′³ᵇ = ZFaceField(grid), Tᵉ = CenterField(grid), budget_work = CenterField(grid),
     NamedTuple(pass_name(name) => ZFaceField(grid) for name in PASS_CLOSURE_FIELD_NAMES)...,
     NamedTuple(pass_name(name) => CenterField(grid) for name in PASS_CLOSURE_CENTER_FIELD_NAMES)...)
const PASS_CLOSURE_FIELD_NAMES = (:N², :ℓᵐ, :ℓˢ, :Kᵘ, :Kᶜ, :Kᵉ)
const PASS_CLOSURE_CENTER_FIELD_NAMES = (:ℓˢᶜ,) # the centers, whose parents are one level shorter
pass_name(name) = Symbol(:pass_, name)

function pass_closure_fields(closure_fields::TKEClosureFields, cl)
    Kᵘ, Kᶜ, Kᵉ = cl.pass_Kᵘ, cl.pass_Kᶜ, cl.pass_Kᵉ
    tracer_diffusivities = map(K -> K === closure_fields.Kᵉ ? Kᵉ : Kᶜ, closure_fields.tupled_tracer_diffusivities)
    return TKEClosureFields(Kᵘ, Kᶜ, Kᵉ, cl.pass_N², cl.pass_ℓᵐ, cl.pass_ℓˢ, cl.pass_ℓˢᶜ, closure_fields.Lᵉ, tracer_diffusivities,
                            closure_fields.tupled_implicit_linear_coefficients, closure_fields.subgrid_cloud,
                            closure_fields.column_stability)
end

@kernel function _copy_parents!(destinations, sources)
    i, j, k = @index(Global, NTuple)
    map((destination, source) -> @inbounds(destination[i, j, k] = source[i, j, k]), destinations, sources)
end

function start_pass_closure_fields!(pass_fields, closure_fields, grid)
    copy_parents!(pass_fields, closure_fields, grid, PASS_CLOSURE_FIELD_NAMES)
    copy_parents!(pass_fields, closure_fields, grid, PASS_CLOSURE_CENTER_FIELD_NAMES)
    return nothing
end

function copy_parents!(pass_fields, closure_fields, grid, names)
    destinations = map(name -> parent(getproperty(pass_fields, name)), names)
    sources = map(name -> parent(getproperty(closure_fields, name)), names)
    launch!(grid.architecture, grid, size(first(sources)), _copy_parents!, destinations, sources)
    return nothing
end

@inline function sink_exchange_coefficientᶜᶜᶠ(i, j, k, grid, ℂᴸ, ρ, ℓᵐ)
    Λ = @inbounds ℂᴸ * ℓᵐ[i, j, k]
    return 2 * ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Λ * Λ / Δzᶜᶜᶠ(i, j, k, grid)
end

@inline function turbulent_shape(e, eᵐⁱⁿ)
    turbulent = e > 0
    e₊ = ifelse(turbulent, e, one(e))
    return ifelse(turbulent, e₊ * (e₊ / (e₊ + eᵐⁱⁿ)), zero(e))
end

struct TurbulentShape{X, S}
    e :: X
    minimum_tke :: S
end

@inline Base.getindex(shape::TurbulentShape, i, j, k) = @inbounds turbulent_shape(shape.e[i, j, k], shape.minimum_tke)

struct SumOfArrays{A, B}
    a :: A
    b :: B
end

@inline Base.getindex(s::SumOfArrays, i, j, k) = @inbounds s.a[i, j, k] + s.b[i, j, k]

@inline function exchange_amplitudes!(s₁, work, i, j, grid, ℂᴸ, shape, source₁, ρ, ℓᵐ)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    tiny = sqrt(floatmin(FT))
    @inbounds begin
        ω⁻ = zero(FT)
        c′ = zero(FT)
        d′₁ = zero(FT)
        a⁻ = zero(FT) # the exchange coefficient at the face below
        ω = shape[i, j, 1]
        for k in 1:Nz
            m = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            diagonal = m
            lower = zero(FT)
            upper = zero(FT)
            ω⁺ = zero(FT)
            if k > 1
                share = ω + ω⁻ + tiny
                diagonal += a⁻ * ω⁻ / share
                lower = -a⁻ * ω / share
            end
            if k < Nz
                a = sink_exchange_coefficientᶜᶜᶠ(i, j, k + 1, grid, ℂᴸ, ρ, ℓᵐ)
                ω⁺ = shape[i, j, k+1]
                share = ω + ω⁺ + tiny
                diagonal += a * ω⁺ / share
                upper = -a * ω / share
                a⁻ = a
            end
            pivot = diagonal - lower * c′
            d′₁ = (m * source₁[i, j, k] - lower * d′₁) / pivot
            c′ = upper / pivot
            work[i, j, k] = c′
            s₁[i, j, k] = d′₁
            ω⁻ = ω
            ω = ω⁺
        end
        for k in Nz-1:-1:1
            s₁[i, j, k] -= work[i, j, k] * s₁[i, j, k+1]
        end
    end
    return nothing
end

@inline balance_sink!(u, work, i, j, grid, ℂʷᴸ, cl, ρ, ℓᵐ) =
    exchange_amplitudes!(u, work, i, j, grid, ℂʷᴸ, u, SumOfArrays(cl.P, cl.B), ρ, ℓᵐ)

#####
##### The damped local third moment
#####

@inline function smooth_positive_part(x)
    a = max(abs(x), one(x))
    s = a * sqrt((x / a)^2 + (1 / a)^2)
    return ifelse(x >= 0, (x + s) / 2, 1 / (2 * (s - x)))
end

@inline function smooth_positive_part(x, δ)
    s = sqrt(x * x + δ * δ)
    return ifelse(x >= 0, (x + s) / 2, δ * δ / (2 * (s - x)))
end

@inline stratification_damping(ℂʷᴺ, δ, αᴺ) = 1 + ℂʷᴺ * δ * smooth_positive_part(αᴺ / δ)

@inline latent_sourceᶜᶜᶜ(i, j, k, cl) =
    @inbounds (1 - cl.aᶜ[i, j, k]) * smooth_positive_part(cl.Bˡ[i, j, k], cl.aᶜ[i, j, k] * cl.εʷ[i, j, k])

@inline damped_local_third_moment(tm, τ, u, αᴺ, latent_source) =
    τ * (u * (tm.ℂʷˡ * latent_source)) / stratification_damping(tm.ℂʷᴺ, tm.stratification_damping_width, αᴺ)

@kernel function _compute_tke_budget_flux!(cl, grid, closure, closure_fields, e, velocities, ρ, diagnose)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    tm = closure_ij.subgrid_distribution.third_moment
    ℓᵐ = closure_fields.ℓᵐ
    eᵐⁱⁿ = closure_ij.minimum_tke
    @inbounds for k in 1:size(grid, 3)
        cl.uᵉ[i, j, k] = turbulent_shape(e[i, j, k], eᵐⁱⁿ)
        cl.εʷ[i, j, k] = dissipationᶜᶜᶜ(i, j, k, grid, closure_ij, e, velocities, closure_fields)
        cl.uᵋ[i, j, k] = cl.εʷ[i, j, k]
    end
    balance_sink!(cl.uᵉ, cl.budget_work, i, j, grid, tm.ℂʷᴸ, cl, ρ, ℓᵐ)
    balance_sink!(cl.uᵋ, cl.budget_work, i, j, grid, tm.ℂʷᴸ, cl, ρ, ℓᵐ)
    N² = closure_fields.N²
    belowᵉ = zero(grid_float_type(grid))
    belowᵋ = zero(grid_float_type(grid))
    @inbounds for k in 1:size(grid, 3)
        ρΔz = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
        rᵉ = ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵉ[i, j, k])
        rᵋ = ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵋ[i, j, k])
        cl.Fᵉ[i, j, k] = (belowᵉ + rᵉ / 2) / ρ[i, j, k]
        cl.Fᵋ[i, j, k] = (belowᵋ + rᵋ / 2) / ρ[i, j, k]
        belowᵉ += rᵉ
        belowᵋ += rᵋ

        cl.Tᵉ[i, j, k] = tm.ℂʷᵀ * (cl.uᵉ[i, j, k] - cl.P[i, j, k] - cl.B[i, j, k])
    end
    diagnose_face_third_moments!(diagnose, cl, i, j, grid, tm, closure_ij, e, ℓᵐ, N², ρ)
end

@inline diagnose_face_third_moments!(::Val{false}, args...) = nothing

@inline function diagnose_face_third_moments!(::Val{true}, cl, i, j, grid, tm, closure_ij, e, ℓᵐ, N², ρ)
    belowᵉ = zero(grid_float_type(grid))
    belowᵋ = zero(grid_float_type(grid))
    @inbounds for k in 1:size(grid, 3)-1
        ρΔz = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
        belowᵉ += ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵉ[i, j, k])
        belowᵋ += ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵋ[i, j, k])
        diagnose_face_third_moment!(cl, i, j, k + 1, grid, tm, closure_ij, e, ℓᵐ, N², ρ, belowᵉ, belowᵋ)
    end
    return nothing
end

@inline function diagnose_face_third_moment!(cl, i, j, k, grid, tm, closure_ij, e, ℓᵐ, N², ρ, ρFᵉ, ρFᵋ)
    @inbounds begin
        ρᶠ = (ρ[i, j, k] + ρ[i, j, k-1]) / 2
        ℓ = ℓᵐ[i, j, k]
        u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij, e)
        τ = ℓ / u
        αᴺ = τ * τ * N²[i, j, k]
        r = 1 + tm.ℂʷʳ * αᴺ / (1 + abs(αᴺ))
        latent = (latent_sourceᶜᶜᶜ(i, j, k, cl) + latent_sourceᶜᶜᶜ(i, j, k-1, cl)) / 2
        cl.w′³ᵇ[i, j, k] = r * (tm.ℂʷᵉ * ρFᵉ + tm.ℂʷᵋ * ρFᵋ) / ρᶠ +
                           damped_local_third_moment(tm, τ, u, αᴺ, latent)
    end
    return nothing
end

function compute_tke_budget_flux!(fields, closure_fields, closure, model, tracers, diagnose)
    grid = model.grid
    launch!(grid.architecture, grid, :xy, _compute_tke_budget_flux!, fields.cloud, grid, closure, closure_fields,
            tracers[TKE_NAME], model.velocities, total_density(model.dynamics), diagnose)
    return nothing
end

#####
##### The third moment of one evaluation of the closure fields
#####

diagnose_third_moment!(fields, distribution, closure_fields, closure, model, tracers, parameters) = nothing
compute_tke_budget!(fields, distribution, closure_fields, closure, model, tracers) = nothing

function diagnose_third_moment!(fields::SubgridCloudFields, distribution::TKEBudgetDistribution, closure_fields, closure, model, tracers, parameters)
    cl = fields.cloud
    pass_fields = pass_closure_fields(closure_fields, cl)
    fill!(parent(cl.w′³ᵇ), 0)
    compute_buoyancy_derivatives!(fields, model)
    compute_budget_fluxes!(fields, closure_fields, closure, model, tracers)
    start_pass_closure_fields!(pass_fields, closure_fields, model.grid)
    compute_subgrid_distribution!(fields, closure_fields, closure, model)
    solve_static_stability!(closure.static_stability, fields.energetics, fields, pass_fields, closure, model, tracers, parameters)
    compute_applied_flux_conversions!(fields, pass_fields, closure, model, tracers)
    compute_tke_budget_flux!(fields, pass_fields, closure, model, tracers, Val(true))
    return nothing
end

compute_subgrid_cloud_buoyancy_derivatives!(fields, ::TKEBudgetDistribution, model) = nothing
compute_subgrid_cloud_budget_fluxes!(fields, ::TKEBudgetDistribution, closure_fields, closure, model, tracers) = nothing

compute_tke_budget!(fields::SubgridCloudFields, ::TKEBudgetDistribution, closure_fields, closure, model, tracers) =
    compute_tke_budget_flux!(fields, closure_fields, closure, model, tracers, Val(false))

#####
##### The third moment at its location k: the one diagnosed from the budget
#####

@inline third_moment_diagnosis(tm::TKEBudgetThirdMoment, i, j, k, d, cl, g, algebraic_arguments) =
    @inbounds cl.w′³ᵇ[i, j, k]

#####
##### The transport of TKE by the budget flux, in the stage tendency of ρe
#####

@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, distribution, fields) = zero(grid)
@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, ::TKEBudgetDistribution, fields) = @inbounds fields.cloud.Tᵉ[i, j, k]
