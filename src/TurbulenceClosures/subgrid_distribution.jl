using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶠ

#####
##### The time scale of the moments
#####
"""
$(TYPEDEF)

The time scale `τ` of the second moments (`θˡⁱ′² = ℂᵛ₁ τ Kᶜ (∂z θˡⁱ)²`, ...) and of the third-moment
diagnosis, at the faces, from the closure's mixing length `ℓ`, turbulent velocity `√e` and
static stability `N²` there: `ℂᵗ` eddy turnover times, limited by the buoyancy period in stable air and
then by a `p`-norm of its rate and the buoyancy rate `N⁺ / ℂᴺ`,
"""
struct StratificationLimitedMomentTimeScale{P, FT}
    ℂᵗ :: FT
    ℂᴺ :: FT
    minimum_buoyancy_frequency :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`StratificationLimitedMomentTimeScale`](@ref): `ℂᵗ = 1` eddy turnover, the limit `ℂᴺ = 4.5` on `τ N` and
the exponent `p = 4` of its norm by default.
"""
StratificationLimitedMomentTimeScale(FT::DataType = Oceananigans.defaults.FloatType;
                                     ℂᵗ = 1, ℂᴺ = 4.5, exponent = 4, minimum_buoyancy_frequency = 1e-4) =
    StratificationLimitedMomentTimeScale{exponent, FT}(convert(FT, ℂᵗ), convert(FT, ℂᴺ), convert(FT, minimum_buoyancy_frequency))

function Adapt.adapt_structure(to, ts::StratificationLimitedMomentTimeScale{P}) where P
    ℂᵗ = adapt(to, ts.ℂᵗ)
    return StratificationLimitedMomentTimeScale{P, typeof(ℂᵗ)}(ℂᵗ, adapt(to, ts.ℂᴺ), adapt(to, ts.minimum_buoyancy_frequency))
end

convert_eltype(::Type{FT}, ts::StratificationLimitedMomentTimeScale{P}) where {P, FT} =
    StratificationLimitedMomentTimeScale{P, FT}(convert(FT, ts.ℂᵗ), convert(FT, ts.ℂᴺ), convert(FT, ts.minimum_buoyancy_frequency))

Base.summary(ts::StratificationLimitedMomentTimeScale{P}) where P =
    string("StratificationLimitedMomentTimeScale(ℂᵗ = ", prettysummary(ts.ℂᵗ), ", ℂᴺ = ", prettysummary(ts.ℂᴺ), ", p = ", P, ")")

@inline function moment_time_scale(ts::StratificationLimitedMomentTimeScale{P}, ℓ, u, N²) where P
    N²₊ = max(N², zero(N²))
    τᵉ = ts.ℂᵗ * ℓ / u
    τ₀ = τᵉ / sqrt(1 + τᵉ * τᵉ * N²₊)
    Nₘᵢₙ = ts.minimum_buoyancy_frequency
    N⁺ = N²₊ / (sqrt(N²₊ + Nₘᵢₙ * Nₘᵢₙ) + Nₘᵢₙ)
    return τ₀ / rate_norm_factor(Val(P), τ₀ * N⁺ / ts.ℂᴺ)
end

#####
##### Time scales of the second moments from the scalar-variance budget
#####
"""
$(TYPEDEF)

The time scale `τ` of the second moments from the variance budget of a scalar `φ` (θˡⁱ or qᵗ) in
local equilibrium, production equal to dissipation, with a scalar-variance dissipation time
tied to the dissipation time of the closure's TKE,
"""
struct DissipationTimeScale{FT, T3}
    ℂᵠ :: FT
    third_moment_time_scale :: T3
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`DissipationTimeScale`](@ref); the third moment keeps `StratificationLimitedMomentTimeScale(FT)`.
"""
DissipationTimeScale(FT::DataType = Oceananigans.defaults.FloatType; ℂᵠ = 2,
                     third_moment_time_scale = StratificationLimitedMomentTimeScale(FT)) =
    DissipationTimeScale(convert(FT, ℂᵠ), third_moment_time_scale)

for TS in (:DissipationTimeScale,)
    @eval begin
        Adapt.adapt_structure(to, ts::$TS) = $TS(adapt(to, ts.ℂᵠ), adapt(to, ts.third_moment_time_scale))
        convert_eltype(::Type{FT}, ts::$TS) where FT = $TS(convert(FT, ts.ℂᵠ), convert_eltype(FT, ts.third_moment_time_scale))
        Base.summary(ts::$TS) = string($(string(TS)), "(ℂᵠ = ", prettysummary(ts.ℂᵠ), ", third moment: ",
                                       summary(ts.third_moment_time_scale), ")")
    end
end

#####
##### The return-to-isotropy time scale of the stability functions
#####
"""
$(TYPEDEF)

The time scale `τ` of the second moments (`θˡⁱ′² = ℂᵛ₁ τ Kᶜ (∂z θˡⁱ)²`, ...) and of the third-moment diagnosis
taken from the closure's stability functions: the return-to-isotropy time scale with which they form the
diffusivities, `Kᶜ = ℂᶜ τ e` (SHOC's `isotropy`, [`ColumnStabilityDamping`](@ref)), so that the moments and the
fluxes share one time scale. It needs stability functions that have one.
"""
struct ReturnToIsotropyTimeScale end

convert_eltype(::Type{FT}, ts::ReturnToIsotropyTimeScale) where FT = ts
Base.summary(::ReturnToIsotropyTimeScale) = "ReturnToIsotropyTimeScale()"
Base.show(io::IO, ts::ReturnToIsotropyTimeScale) = print(io, summary(ts))

validate_moment_time_scale(stability_functions, time_scale) = nothing

@inline function moment_time_scaleᶜᶜᶠ(i, j, k, grid, ts::DissipationTimeScale, closure, velocities, closure_fields, e, ℓ, u)
    𝕊ᴰ = dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, closure_fields.N², e, closure_fields.ℓᵐ)
    return 2 * ℓ / (ts.ℂᵠ * 𝕊ᴰ * u)
end

@inline moment_time_scaleᶜᶜᶠ(i, j, k, grid, ::ReturnToIsotropyTimeScale, closure, velocities, closure_fields, e, ℓ, u) =
    return_to_isotropy_time_scaleᶜᶜᶠ(i, j, k, grid, closure.stability_functions, closure, closure_fields, ℓ, u)

@inline third_moment_time_scale(::ReturnToIsotropyTimeScale, τ, ℓ, u, N²) = τ
@inline third_moment_time_scale(ts::DissipationTimeScale, τ, ℓ, u, N²) =
    moment_time_scale(ts.third_moment_time_scale, ℓ, u, N²)

#####
##### Part 1: the subgrid distribution
#####
"""
$(TYPEDEF)

The subgrid distribution of θˡⁱ, qᵗ and w of [`TKEBasedTurbulenceClosure`](@ref) (`subgrid_distribution`): the
second moments, the third moment and its limiter and the assumed PDF, evaluated on the mixing length, turbulent
kinetic energy, diffusivities and static stability of the closure. The PDF condenses with the thermodynamics of
the model, on the phase of the condensate of the water it distributes (`condensate_phase_equilibrium`): liquid, or
liquid and ice for mixed-phase saturation adjustment. With P3 the distributed water is the vapor and the cloud
liquid, which condenses over liquid; P3's ice enters through its relaxation rate and the saturation excess over ice.
"""
struct DoubleGaussian{TS, SM, TM, SB, SK, PDF, VT, VS, FT}
    time_scale :: TS
    second_moments :: SM
    third_moment :: TM
    stratification_number_bound :: SB
    skewness :: SK
    assumed_pdf :: PDF
    variance_transport :: VT
    variance_stability_functions :: VS
    ℂᵐ :: FT
end

Adapt.@adapt_structure DoubleGaussian
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`DoubleGaussian`](@ref). The defaults are the moment time scale of the scalar-variance budget
([`DissipationTimeScale`](@ref) with `ℂᵠ = 1.3`), SHOC's down-gradient second moments
([`DowngradientSecondMoments`](@ref)), the third moment of the TKE budget ([`TKEBudgetThirdMoment`](@ref), in place
of SHOC's [`AlgebraicThirdMoment`](@ref)) with the bound [`RationalSaturatedStratificationNumber`](@ref) of the
stratification number, the double Gaussian with equal within-component widths
([`EqualWidthDoubleGaussianPDF`](@ref), whose bounds are smooth) and the limiter
[`PolynomialSaturatedSkewness`](@ref) with the largest skewness that `assumed_pdf` represents,
"""
function DoubleGaussian(FT::DataType = Oceananigans.defaults.FloatType;
                             time_scale = DissipationTimeScale(FT; ℂᵠ = 1.3),
                             second_moments = DowngradientSecondMoments(FT),
                             third_moment = TKEBudgetThirdMoment(FT),
                             stratification_number_bound = RationalSaturatedStratificationNumber(FT),
                             assumed_pdf = EqualWidthDoubleGaussianPDF(FT),
                             skewness = PolynomialSaturatedSkewness(FT; maximum_skewness = representable_skewness(assumed_pdf)),
                             variance_transport = ScalarVarianceTransport(FT),
                             variance_stability_functions = VarianceStabilityFunctions(FT),
                             ℂᵐ = 1)
    variance_transport = convert_component(FT, variance_transport)
    variance_stability_functions = convert_component(FT, variance_stability_functions)
    return DoubleGaussian(time_scale, second_moments, third_moment, stratification_number_bound,
                               skewness, assumed_pdf, variance_transport,
                               variance_stability_functions, convert(FT, ℂᵐ))
end

convert_eltype(::Type{FT}, d::DoubleGaussian) where FT =
    DoubleGaussian(convert_eltype(FT, d.time_scale), map(c -> convert_component(FT, c),
                        (d.second_moments, d.third_moment, d.stratification_number_bound, d.skewness,
                         d.assumed_pdf, d.variance_transport,
                         d.variance_stability_functions, d.ℂᵐ))...)

convert_component(FT, ::Nothing) = nothing
convert_component(FT, c) = convert_eltype(FT, c)
convert_component(FT, c::Number) = convert(FT, c)

Base.summary(d::DoubleGaussian) = string("DoubleGaussian(", nameof(typeof(d.assumed_pdf)), ", third moment: ",
                                              d.third_moment === nothing ? "none" : nameof(typeof(d.third_moment)), ")")

struct AppliedFluxEnergetics end
"""
$(TYPEDEF)

The `fraction` of [`SaturationWeightedStaticStability`](@ref) given by the closure's own
[`DoubleGaussian`](@ref) distribution: the saturated fraction is the ratio of the subgrid condensate
flux to the flux of the saturation excess `s = c_q qᵗ - c_θ θˡⁱ` (linearized at the level),
"""
struct CondensateFluxFraction{S, T}
    initial_stability :: S
    stratification :: T
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`CondensateFluxFraction`](@ref).
"""
CondensateFluxFraction(initial_stability = DryStaticStability(); stratification = LatentStratification()) =
    CondensateFluxFraction(initial_stability, stratification)

Base.show(io::IO, f::CondensateFluxFraction) =
    print(io, "CondensateFluxFraction(", summary(f.initial_stability), stratification_summary(f.stratification), ")")
"""
$(TYPEDEF)

The static stability of [`CondensateFluxFraction`](@ref) once the distribution exists that carries the latent part of
the subgrid condensation, `N² = N²ᶜ + (b_qˡ - b_qᵛ) [η aʷ ∂z s̄ + (1 - η) ∂z qᶜ]`: the share `aᴺ = η aʷ` of the flux of the
saturation excess that condenses feels the saturated stratification. The static stability and the buoyancy production
of the TKE are then one.
"""
struct LatentStratification end
"""
$(TYPEDEF)

The static stability of [`CondensateFluxFraction`](@ref) once the distribution exists of the grid-mean state with the
distribution's mean liquid `q̄ˡ` as its condensate, the stratification of a displacement without phase change,
"""
struct MeanLiquidStratification end

Base.summary(::LatentStratification) = "LatentStratification()"
Base.summary(::MeanLiquidStratification) = "MeanLiquidStratification()"

stratification_summary(::LatentStratification) = ""
stratification_summary(stratification) = string(", ", summary(stratification))

@inline condensate_flux_stratification(::LatentStratification, N²ᵉ, N²ᵇ, bˡᵛ, Δqˡ, Δqᶜ, Δz, N²⁰) = N²ᵉ
@inline condensate_flux_stratification(::MeanLiquidStratification, N²ᵉ, N²ᵇ, bˡᵛ, Δqˡ, Δqᶜ, Δz, N²⁰) = N²ᵇ + bˡᵛ * (Δqˡ - Δqᶜ) / Δz

const CondensateFluxFractionStability = SaturationWeightedStaticStability{<:CondensateFluxFraction}

@inline static_stabilityᶜᶜᶠ(i, j, k, grid, ss::CondensateFluxFractionStability, buoyancy, tracers, closure_fields) =
    static_stabilityᶜᶜᶠ(i, j, k, grid, ss.fraction.initial_stability, buoyancy, tracers, closure_fields)

#####
##### Validation of the keywords of the closure
#####

function validate_subgrid_coupling(static_stability, distribution)
    if static_stability isa CondensateFluxFractionStability && isnothing(distribution)
        throw(ArgumentError("SaturationWeightedStaticStability(CondensateFluxFraction(...)) needs a subgrid_distribution, " *
                            "whose condensate flux gives the saturated fraction."))
    end
    return nothing
end

#####
##### Fields
#####

struct SubgridCloudFields{BF, M, C, P}
    energetics :: BF
    moments :: M
    cloud :: C
    presolve :: P # every closure scalar just before its implicit solve (the stage energy budget), or `nothing`
end

Adapt.adapt_structure(to, f::SubgridCloudFields) =
    SubgridCloudFields(f.energetics, adapt(to, f.moments), adapt(to, f.cloud),
                       adapt(to, f.presolve))

build_subgrid_cloud_fields(grid, ::Nothing, static_stability, tracer_names, heating_budget_flux = nothing) = nothing

function build_subgrid_cloud_fields(grid, distribution, static_stability, tracer_names, heating_budget_flux = nothing)
    energetics = AppliedFluxEnergetics()
    moments = (τ = ZFaceField(grid), w′³ = ZFaceField(grid),
               θˡⁱ′² = ZFaceField(grid), qᵗ′² = ZFaceField(grid), θˡⁱ′qᵗ′ = ZFaceField(grid),
               w′θˡⁱ′ = ZFaceField(grid), w′qᵗ′ = ZFaceField(grid),
               unit_variance = ZFaceField(grid), qⁱ′² = ZFaceField(grid), qⁱ′sⁱ′ = ZFaceField(grid), qᵈ′² = ZFaceField(grid),
               scalar_variance_transport_fields(grid, distribution.variance_transport)...)
    cloud = (w′² = CenterField(grid), w′³ = CenterField(grid), aᶜ = CenterField(grid), qˡ = CenterField(grid), qˡ′² = CenterField(grid),
             w′qˡ′ = CenterField(grid), w′θᵛ′ = CenterField(grid), Bˡ = CenterField(grid),
             vapor_excess_component_fields(grid)...,
             qⁱ′sⁱ′ = CenterField(grid), qᵈ′² = CenterField(grid),
             applied_flux_conversion_fields(grid, energetics)...,
             third_moment_fields(grid, distribution.third_moment)...,
             heating_budget_flux_fields(grid, heating_budget_flux)...)
    return SubgridCloudFields(energetics, moments, cloud,
                              build_presolve_fields(grid, energetics, tracer_names))
end

function applied_flux_conversion_fields(grid, ::AppliedFluxEnergetics)
    centers = NamedTuple(name => CenterField(grid) for name in (:B, :Bᴸ, :Bᴸⁱ, :Bⁱ, :P, :Pⁱ, :Bˢ, :Pˢ, :ωᴮ, :b_θ, :b_qᵛ, :b_qˡ, :b_qⁱ, :b_qᶜ, :ĉᵖΠ, :Bᵖ, :Pᵖ, :ε, :given, :stage_b_θ, :stage_b_qᵛ, :stage_b_qˡ, :stage_b_qⁱ, :stage_surface, :shortfall, :after_solves_ρθ, :stage_combination))
    faces = (aʷ = ZFaceField(grid), aᴺ = ZFaceField(grid), ΔN²ᴸ = ZFaceField(grid), η = ZFaceField(grid), c_q = ZFaceField(grid), c_θ = ZFaceField(grid),
             ηⁱ = ZFaceField(grid), ηʷⁱ = ZFaceField(grid), c_qⁱ = ZFaceField(grid), c_θⁱ = ZFaceField(grid))
    return merge(centers, faces, (ρu⁻ = XFaceField(grid), ρv⁻ = YFaceField(grid),
                                  stage_time_step = Oceananigans.Fields.Field{Nothing, Nothing, Nothing}(grid)))
end

build_presolve_fields(grid, ::AppliedFluxEnergetics, tracer_names) = NamedTuple(name => CenterField(grid) for name in tracer_names)
#####
##### Specific values from densities, for the kernels
#####

struct SpecificValue{F, R}
    field :: F
    density :: R
end

Adapt.adapt_structure(to, s::SpecificValue) = SpecificValue(adapt(to, s.field), adapt(to, s.density))
@inline Base.getindex(s::SpecificValue, i, j, k) = @inbounds s.field[i, j, k] / s.density[i, j, k]
@inline Base.getindex(s::SpecificValue{<:Any, Nothing}, i, j, k) = @inbounds s.field[i, j, k]

struct XFaceVelocity{F, R, G}
    momentum :: F
    density :: R
    grid :: G
end

struct YFaceVelocity{F, R, G}
    momentum :: F
    density :: R
    grid :: G
end

Adapt.adapt_structure(to, u::XFaceVelocity) = XFaceVelocity(adapt(to, u.momentum), adapt(to, u.density), adapt(to, u.grid))
Adapt.adapt_structure(to, v::YFaceVelocity) = YFaceVelocity(adapt(to, v.momentum), adapt(to, v.density), adapt(to, v.grid))
@inline Base.getindex(u::XFaceVelocity, i, j, k) = @inbounds u.momentum[i, j, k] / ℑxᶠᵃᵃ(i, j, k, u.grid, u.density)
@inline Base.getindex(v::YFaceVelocity, i, j, k) = @inbounds v.momentum[i, j, k] / ℑyᵃᶠᵃ(i, j, k, v.grid, v.density)

struct SpecificTotalWater{V, L, R}
    vapor :: V
    cloud_liquid :: L
    density :: R
end

Adapt.adapt_structure(to, q::SpecificTotalWater) =
    SpecificTotalWater(adapt(to, q.vapor), adapt(to, q.cloud_liquid), adapt(to, q.density))

@inline Base.getindex(q::SpecificTotalWater, i, j, k) =
    @inbounds (q.vapor[i, j, k] + q.cloud_liquid[i, j, k]) / q.density[i, j, k]
@inline Base.getindex(q::SpecificTotalWater{<:Any, Nothing}, i, j, k) =
    @inbounds q.vapor[i, j, k] / q.density[i, j, k]

@inline cloud_liquid_density(microphysical_fields) =
    hasproperty(microphysical_fields, :ρqᶜˡ) ? microphysical_fields.ρqᶜˡ : nothing

function distribution_potential_temperature_density(model)
    AtmosphereModels.thermodynamic_density_name(model.formulation) === :ρθ ||
        throw(ArgumentError("The subgrid distribution of TKEBasedTurbulenceClosure requires the liquid-ice " *
                            "potential temperature formulation, but the model formulation is $(summary(model.formulation))."))
    return Oceananigans.prognostic_fields(model.formulation).ρθ
end

struct DistributedWaterPotentialTemperature{Θ, R, V, L, P, G, M, F, C, X, S}
    ρθ :: Θ
    density :: R
    moisture_density :: V
    cloud_liquid :: L
    pressure :: P
    grid :: G
    microphysics :: M
    microphysical_fields :: F
    constants :: C
    exner_exponent :: X
    standard_pressure :: S
end

Adapt.adapt_structure(to, θ::DistributedWaterPotentialTemperature) =
    DistributedWaterPotentialTemperature(adapt(to, θ.ρθ), adapt(to, θ.density), adapt(to, θ.moisture_density),
                                         adapt(to, θ.cloud_liquid), adapt(to, θ.pressure), adapt(to, θ.grid),
                                         adapt(to, θ.microphysics), adapt(to, θ.microphysical_fields),
                                         θ.constants, θ.exner_exponent, θ.standard_pressure)

@inline function Base.getindex(θ::DistributedWaterPotentialTemperature, i, j, k)
    constants = θ.constants
    @inbounds begin
        ρ = θ.density[i, j, k]
        θˡⁱ = θ.ρθ[i, j, k] / ρ
        qᶜˡ = θ.cloud_liquid[i, j, k] / ρ
        p = θ.pressure[i, j, k]
        qᵛᵉ = θ.moisture_density[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, θ.grid, θ.microphysics, ρ, qᵛᵉ, θ.microphysical_fields)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    Π = (p / θ.standard_pressure)^exner_exponent(θ.exner_exponent, q, constants)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    return θˡⁱ + (ℒˡ₀ * (q.liquid - qᶜˡ) + ℒⁱ₀ * q.ice) / (ĉᵖ * Π)
end

struct HeldCondensate{R, V, L, G, M, F}
    density :: R
    moisture_density :: V
    cloud_liquid :: L
    grid :: G
    microphysics :: M
    microphysical_fields :: F
end

Adapt.adapt_structure(to, h::HeldCondensate) =
    HeldCondensate(adapt(to, h.density), adapt(to, h.moisture_density), adapt(to, h.cloud_liquid), adapt(to, h.grid),
                   adapt(to, h.microphysics), adapt(to, h.microphysical_fields))

@inline function Base.getindex(h::HeldCondensate, i, j, k)
    @inbounds begin
        ρ = h.density[i, j, k]
        qᶜˡ = h.cloud_liquid[i, j, k] / ρ
        qᵛᵉ = h.moisture_density[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, h.grid, h.microphysics, ρ, qᵛᵉ, h.microphysical_fields)
    return (; liquid = q.liquid - qᶜˡ, ice = q.ice)
end

@inline held_condensate_field(s, grid, ::Nothing) = nothing
@inline held_condensate_field(s, grid, ρqᶜˡ) = HeldCondensate(s.ρ, s.ρqᵛᵉ, ρqᶜˡ, grid, s.microphysics, s.microphysical_fields)

@inline level_thermodynamics(thermodynamics::NamedTuple{names}, i, j, k) where names =
    level_thermodynamics(Val(:held_condensate ∈ names), thermodynamics, i, j, k)
@inline level_thermodynamics(::Val{false}, thermodynamics, i, j, k) = thermodynamics
@inline level_thermodynamics(::Val{true}, thermodynamics, i, j, k) =
    level_thermodynamics(thermodynamics.held_condensate, thermodynamics, i, j, k)
@inline level_thermodynamics(::Nothing, thermodynamics, i, j, k) = thermodynamics
@inline level_thermodynamics(held, thermodynamics, i, j, k) = merge(thermodynamics, (; held = held[i, j, k]))

@inline distributed_water_potential_temperature(s, grid, ::Nothing) = SpecificValue(s.ρθ, s.ρ)

@inline distributed_water_potential_temperature(s, grid, ρqᶜˡ) =
    DistributedWaterPotentialTemperature(s.ρθ, s.ρ, s.ρqᵛᵉ, ρqᶜˡ, s.p, grid, s.microphysics, s.microphysical_fields, s.constants,
                                         s.exner_exponent, s.standard_pressure)

@inline function distributed_water(s, grid)
    ρqᶜˡ = cloud_liquid_density(s.microphysical_fields)
    θ = distributed_water_potential_temperature(s, grid, ρqᶜˡ)
    qᵗ = SpecificTotalWater(s.ρqᵛᵉ, ρqᶜˡ, s.ρ)
    thermodynamics = (constants = s.constants, standard_pressure = s.standard_pressure, exner_exponent = s.exner_exponent,
                      phase_equilibrium = s.phase_equilibrium, held_condensate = held_condensate_field(s, grid, ρqᶜˡ))
    return (; θ, qᵗ, thermodynamics)
end

#####
##### The column computation of the distribution
#####

@inline subgrid_third_moment(tm, d, g, args...) = third_moment(tm, d.stratification_number_bound, g, args...)

@inline third_moment_diagnosis(tm, i, j, k, d, cl, g, algebraic_arguments) =
    subgrid_third_moment(tm, d, g, algebraic_arguments...)

@inline function face_third_moment(i, j, k, grid, d, closure, closure_fields, m, cl, g, θ, e)
    FT = grid_float_type(grid)
    eᵐⁱⁿ = closure.minimum_tke
    ccc = (Center(), Center(), Center())
    @inbounds begin
        zᵘ = znode(i, j, k, grid, ccc...)
        zˡ = znode(i, j, k - 1, grid, ccc...)
        zᶠ = znode(i, j, k, grid, Center(), Center(), Face())
        ℓ = closure_fields.ℓᵐ[i, j, k]
        N² = closure_fields.N²[i, j, k]
        u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
        τ³ = third_moment_time_scale(d.time_scale, m.τ[i, j, k], ℓ, u, N²)
        τ³ = ifelse(isfinite(τ³), τ³, zero(FT))
        w′² = linearly_interpolate(zᵘ, zˡ, cl.w′²[i, j, k], cl.w′²[i, j, k-1], zᶠ)
        θᶠ = linearly_interpolate(zᵘ, zˡ, θ[i, j, k], θ[i, j, k-1], zᶠ)
        algebraic_arguments = (τ³, N², w′², θᶠ,
                               1 / (zᵘ - zˡ), 1 / (Δzᶜᶜᶜ(i, j, k, grid) + Δzᶜᶜᶜ(i, j, k-1, grid)),
                               m.θˡⁱ′²[i, j, k+1] - m.θˡⁱ′²[i, j, k-1],
                               m.w′θˡⁱ′[i, j, k+1] - m.w′θˡⁱ′[i, j, k-1],
                               cl.w′²[i, j, k] - cl.w′²[i, j, k-1],
                               max(e[i, j, k], eᵐⁱⁿ) - max(e[i, j, k-1], eᵐⁱⁿ),
                               m.w′θˡⁱ′[i, j, k])
        w′³ = third_moment_diagnosis(d.third_moment, i, j, k, d, cl, g, algebraic_arguments)
    end
    return w′³
end

@inline function third_moment_profile!(m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())
    @inbounds begin
        m.w′³[i, j, 1] = zero(FT)
        m.w′³[i, j, Nz+1] = zero(FT)
        for k in 2:Nz
            m.w′³[i, j, k] = face_third_moment(i, j, k, grid, d, closure, closure_fields, m, cl, g, θ, e)
        end
        for k in 1:Nz
            zᶜ = znode(i, j, k, grid, ccc...)
            zˡ = znode(i, j, k, grid, ccf...)
            zᵘ = znode(i, j, k+1, grid, ccf...)
            w′³ = linearly_interpolate(zᵘ, zˡ, m.w′³[i, j, k+1], m.w′³[i, j, k], zᶜ)
            cl.w′³[i, j, k] = limit_third_moment(d.skewness, cl.w′²[i, j, k], w′³)
        end
    end
    return nothing
end

@inline function subgrid_distribution_column!(m, cl, i, j, grid, d, closure, closure_fields, θ, qᵗ, e, velocities, p, thermodynamics)
    w = velocities.w
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    sm = d.second_moments
    eᵐⁱⁿ = closure.minimum_tke
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())

    g = thermodynamics.constants.gravitational_acceleration

    @inbounds begin
        for k in 2:Nz
            Δz = Δzᶜᶜᶠ(i, j, k, grid)
            Kᶜ = closure_fields.Kᶜ[i, j, k]
            Kᵘ = closure_fields.Kᵘ[i, j, k]
            ℓ = closure_fields.ℓᵐ[i, j, k]
            u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
            τ = moment_time_scaleᶜᶜᶠ(i, j, k, grid, d.time_scale, closure, velocities, closure_fields, e, ℓ, u)
            τ = ifelse(isfinite(τ), τ, zero(FT))

            Δθ = θ[i, j, k] - θ[i, j, k-1]
            Δq = qᵗ[i, j, k] - qᵗ[i, j, k-1]
            θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′, w′θˡⁱ′, w′qᵗ′, _, _, _ =
                interface_second_moments(sm, τ, Kᶜ, Kᵘ, 1 / Δz, Δθ, Δq, zero(FT), zero(FT), zero(FT))

            θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′ = local_scalar_second_moments(d.variance_stability_functions, i, j, k, grid, closure, velocities,
                                                             closure_fields, e, Δθ, Δq, Δz, θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′)

            w′θˡⁱ′ += applied_heating_flux(cl, i, j, k)

            _, unit_variance, _ = interface_second_moments(sm, τ, Kᶜ, Kᵘ, 1 / Δz, zero(FT), Δz, zero(FT), zero(FT), zero(FT))
            _, unit_variance, _ = local_scalar_second_moments(d.variance_stability_functions, i, j, k, grid, closure, velocities,
                                                              closure_fields, e, zero(FT), Δz, Δz, zero(FT), unit_variance, zero(FT))
            m.unit_variance[i, j, k] = unit_variance

            m.τ[i, j, k] = τ
            m.θˡⁱ′²[i, j, k] = θˡⁱ′²
            m.qᵗ′²[i, j, k] = qᵗ′²
            m.θˡⁱ′qᵗ′[i, j, k] = θˡⁱ′qᵗ′
            m.w′θˡⁱ′[i, j, k] = w′θˡⁱ′
            m.w′qᵗ′[i, j, k] = w′qᵗ′
        end

        k′ = min(2, Nz)
        m.τ[i, j, 1] = m.τ[i, j, k′]
        m.θˡⁱ′²[i, j, 1] = m.θˡⁱ′²[i, j, k′]
        m.qᵗ′²[i, j, 1] = m.qᵗ′²[i, j, k′]
        m.θˡⁱ′qᵗ′[i, j, 1] = m.θˡⁱ′qᵗ′[i, j, k′]
        m.w′θˡⁱ′[i, j, 1] = m.w′θˡⁱ′[i, j, k′]
        m.w′qᵗ′[i, j, 1] = m.w′qᵗ′[i, j, k′]
        m.τ[i, j, Nz+1] = 0
        m.θˡⁱ′²[i, j, Nz+1] = 0
        m.qᵗ′²[i, j, Nz+1] = 0
        m.θˡⁱ′qᵗ′[i, j, Nz+1] = 0
        m.w′θˡⁱ′[i, j, Nz+1] = 0
        m.w′qᵗ′[i, j, Nz+1] = 0

        for k in 1:Nz
            cl.w′²[i, j, k] = vertical_velocity_variance(sm, max(e[i, j, k], eᵐⁱⁿ))
        end

        transport_scalar_variances!(d.variance_transport, m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)

        m.θˡⁱ′²[i, j, 1] = m.θˡⁱ′²[i, j, k′]
        m.qᵗ′²[i, j, 1] = m.qᵗ′²[i, j, k′]
        m.θˡⁱ′qᵗ′[i, j, 1] = m.θˡⁱ′qᵗ′[i, j, k′]

        third_moment_profile!(m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)

        interface_moments(k) = @inbounds (m.θˡⁱ′²[i, j, k], m.w′θˡⁱ′[i, j, k], m.θˡⁱ′qᵗ′[i, j, k], m.w′qᵗ′[i, j, k], m.qᵗ′²[i, j, k])
        for k in 1:Nz
            zᶜ = znode(i, j, k, grid, ccc...)
            zˡ = znode(i, j, k, grid, ccf...)
            zᵘ = znode(i, j, k+1, grid, ccf...)
            wᶜ = (w[i, j, k] + w[i, j, k+1]) / 2
            θᵏ = θ[i, j, k]
            w′²ᶜ = cl.w′²[i, j, k]
            mˡ = interface_moments(k)
            mᵘ = interface_moments(k+1)
            mᶜ = map((yᵘ, yˡ) -> linearly_interpolate(zᵘ, zˡ, yᵘ, yˡ, zᶜ), mᵘ, mˡ)
            mᶜ = (cl.w′³[i, j, k], mᶜ...)
            thermodynamicsᵏ = level_thermodynamics(thermodynamics, i, j, k)
            C, qˡ, qˡ′², w′qˡ′, w′θᵛ′ = assumed_pdf(d.assumed_pdf, thermodynamicsᵏ, zᵘ, zˡ, zᶜ,
                                                   p[i, j, k], θᵏ, wᶜ, qᵗ[i, j, k], w′²ᶜ, mᶜ, mᶜ)
            store_vapor_excess_components!(cl, i, j, k,
                                           component_vapor_excess(d.assumed_pdf, thermodynamicsᵏ, zᵘ, zˡ, zᶜ,
                                                                  p[i, j, k], θᵏ, wᶜ, qᵗ[i, j, k], w′²ᶜ, mᶜ, mᶜ))
            cl.aᶜ[i, j, k] = C
            ℓᶜ = (closure_fields.ℓᵐ[i, j, k] + closure_fields.ℓᵐ[i, j, k+1]) / 2
            uᶜ = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
            cl.ωᵐ[i, j, k] = ifelse(ℓᶜ > 0, uᶜ / (d.ℂᵐ * ifelse(ℓᶜ > 0, ℓᶜ, one(FT))), zero(FT))
            cl.qˡ[i, j, k] = qˡ
            cl.qˡ′²[i, j, k] = qˡ′²
            cl.w′qˡ′[i, j, k] = w′qˡ′
            cl.w′θᵛ′[i, j, k] = w′θᵛ′
            cl.Bˡ[i, j, k] = (cl.b_qᶜ[i, j, k] - cl.b_qᵛ[i, j, k]) * w′qˡ′
        end
    end

    return nothing
end

@kernel function _compute_subgrid_distribution!(closure_fields, grid, closure, e, velocities, s)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    d = closure_ij.subgrid_distribution
    fields = closure_fields.subgrid_cloud
    water = distributed_water(s, grid)
    subgrid_distribution_column!(fields.moments, fields.cloud, i, j, grid, d, closure_ij, closure_fields, water.θ, water.qᵗ, e,
                                 velocities, s.p, water.thermodynamics)
end

function compute_subgrid_distribution!(fields::SubgridCloudFields, closure_fields, closure, model)
    grid = model.grid
    e = SpecificValue(model.tracers[TKE_NAME], nothing) # within `update_state!` the tracer holds its specific value
    launch!(grid.architecture, grid, :xy, _compute_subgrid_distribution!, closure_fields, grid, closure, e, model.velocities,
            applied_flux_state(model))
    compute_outside_condensate_moments!(fields, closure_fields, model)
    return nothing
end

compute_outside_condensate_moments!(fields, closure_fields, model) = nothing

function compute_outside_condensate_moments!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, model)
    grid = model.grid
    launch!(grid.architecture, grid, :xy, _compute_outside_condensate_moments!, fields.moments, fields.cloud, grid,
            closure_fields.Kᶜ, applied_flux_state(model))
    return nothing
end

@inline outside_ice(::Nothing, i, j, k) = 0
@inline outside_ice(held, i, j, k) = held[i, j, k].ice

@kernel function _compute_outside_condensate_moments!(m, cl, grid, Kᶜ, s)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    θ, qʷ, thermodynamics = distributed_water(s, grid)
    ice = ice_saturation_thermodynamics(thermodynamics)
    held = thermodynamics.held_condensate
    @inbounds begin
        for k in 2:Nz
            Δz = Δzᶜᶜᶠ(i, j, k, grid)
            K = Kᶜ[i, j, k]
            resolved = K > 0
            τᵠ = ifelse(resolved, m.unit_variance[i, j, k] / (2 * ifelse(resolved, K, one(FT))), zero(FT))
            vⁱ_q⁺, vⁱ_θ⁺ = vapor_excess_coefficients(ice, i, j, k, grid, s, θ[i, j, k])
            vⁱ_q⁻, vⁱ_θ⁻ = vapor_excess_coefficients(ice, i, j, k-1, grid, s, θ[i, j, k-1])
            vⁱ_q = (vⁱ_q⁺ + vⁱ_q⁻) / 2
            vⁱ_θ = (vⁱ_θ⁺ + vⁱ_θ⁻) / 2
            sⁱ′² = max(0, vⁱ_q * vⁱ_q * m.qᵗ′²[i, j, k] - 2 * vⁱ_q * vⁱ_θ * m.θˡⁱ′qᵗ′[i, j, k] + vⁱ_θ * vⁱ_θ * m.θˡⁱ′²[i, j, k])
            ∂z_sⁱ = (vⁱ_q * (qʷ[i, j, k] - qʷ[i, j, k-1]) - vⁱ_θ * (θ[i, j, k] - θ[i, j, k-1])) / Δz
            ∂z_qⁱ = (convert(FT, outside_ice(held, i, j, k)) - convert(FT, outside_ice(held, i, j, k-1))) / Δz
            ω⁺ = cell_phase_relaxation_rates(i, j, k, grid, s)
            ω⁻ = cell_phase_relaxation_rates(i, j, k-1, grid, s)
            ωⁱ = (ω⁺.ωᵈⁱ + ω⁻.ωᵈⁱ) / 2
            ωˡ = (ω⁺.ωˡ + ω⁻.ωˡ) / 2
            covariance = τᵠ * (2 * K * ∂z_qⁱ * ∂z_sⁱ + ωⁱ * sⁱ′²) / (1 + τᵠ * ωⁱ)
            variance = τᵠ * (2 * K * ∂z_qⁱ * ∂z_qⁱ + 2 * ωⁱ * covariance)
            bound = sqrt(max(0, variance) * sⁱ′²)
            scale = max(bound, abs(covariance))
            resolved = scale > 0
            scaleₛ = ifelse(resolved, scale, one(FT))
            m.qⁱ′²[i, j, k] = max(0, variance)
            m.qⁱ′sⁱ′[i, j, k] = ifelse(resolved, covariance * bound / (scaleₛ * sqrt(sqrt((bound / scaleₛ)^4 + (covariance / scaleₛ)^4))), zero(FT))
            d⁺ = distributed_condensateᶜᶜᶜ(i, j, k, grid, s) - cl.qˡ[i, j, k]
            d⁻ = distributed_condensateᶜᶜᶜ(i, j, k-1, grid, s) - cl.qˡ[i, j, k-1]
            ∂z_d = (d⁺ - d⁻) / Δz
            m.qᵈ′²[i, j, k] = 2 * K * τᵠ * ∂z_d * ∂z_d / (1 + 2 * τᵠ * ωˡ)
        end
        m.qⁱ′²[i, j, 1] = m.qⁱ′²[i, j, min(2, Nz)]
        m.qⁱ′sⁱ′[i, j, 1] = m.qⁱ′sⁱ′[i, j, min(2, Nz)]
        m.qᵈ′²[i, j, 1] = m.qᵈ′²[i, j, min(2, Nz)]
        m.qⁱ′²[i, j, Nz+1] = 0
        m.qⁱ′sⁱ′[i, j, Nz+1] = 0
        m.qᵈ′²[i, j, Nz+1] = 0
        for k in 1:Nz
            cl.qⁱ′sⁱ′[i, j, k] = (m.qⁱ′sⁱ′[i, j, k] + m.qⁱ′sⁱ′[i, j, k+1]) / 2
            cl.qᵈ′²[i, j, k] = (m.qᵈ′²[i, j, k] + m.qᵈ′²[i, j, k+1]) / 2
        end
    end
end

#####
##### Coupling to the closure
#####

compute_subgrid_cloud_buoyancy_derivatives!(fields, distribution, model) = compute_buoyancy_derivatives!(fields, model)

function compute_budget_fluxes!(fields, closure_fields, closure, model, tracers)
    compute_heating_budget_flux!(fields, heating_budget_flux(closure), closure_fields, closure, model, tracers)
    return nothing
end

compute_subgrid_cloud_budget_fluxes!(fields, distribution, closure_fields, closure, model, tracers) =
    compute_budget_fluxes!(fields, closure_fields, closure, model, tracers)

function compute_subgrid_cloud!(fields::SubgridCloudFields, closure_fields, closure, model, tracers, parameters)
    compute_subgrid_cloud_buoyancy_derivatives!(fields, subgrid_distribution(closure), model)
    compute_subgrid_cloud_budget_fluxes!(fields, subgrid_distribution(closure), closure_fields, closure, model, tracers)
    compute_subgrid_distribution!(fields, closure_fields, closure, model)
    solve_static_stability!(closure.static_stability, fields.energetics, fields, closure_fields, closure, model, tracers, parameters)
    return nothing
end

solve_static_stability!(static_stability, energetics, args...) = nothing
solve_static_stability!(::CondensateFluxFractionStability, energetics, args...) = applied_flux_stability!(energetics, args...)

@inline function condensate_flux_stabilityᶜᶜᶠ(i, j, k, grid, cl, s, θ, qᵗ, thermodynamics)
    km = max(k - 1, 1)
    g = saturation_excess_gradientᶜᶜᶠ(i, j, k, grid, cl, s, θ, qᵗ, thermodynamics)
    @inbounds begin
        b_qᵛᶠ = (cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, km]) / 2
        bˡᵛaʷ⁺ = (cl.b_qᶜ[i, j, k] - cl.b_qᵛ[i, j, k]) * center_condensate_flux_fraction(i, j, k, cl)
        bˡᵛaʷ⁻ = (cl.b_qᶜ[i, j, km] - cl.b_qᵛ[i, j, km]) * center_condensate_flux_fraction(i, j, km, cl)
    end
    N²ᶜ = g.b_θᶠ * g.∂zθ + b_qᵛᶠ * g.∂zq
    N²ᵇ = N²ᶜ + (bˡᵛaʷ⁺ + bˡᵛaʷ⁻) / 2 * g.∂zs
    return N²ᶜ, N²ᵇ, g.∂zs, g.bᶜᵛ
end

@inline function saturation_excess_gradientᶜᶜᶠ(i, j, k, grid, cl, s, θ, qᵗ, thermodynamics)
    km = max(k - 1, 1)
    @inbounds begin
        Δz = Δzᶜᶜᶠ(i, j, k, grid)
        Δθ = θ[i, j, k] - θ[i, j, km]
        Δq = qᵗ[i, j, k] - qᵗ[i, j, km]
        b_θᶠ = (cl.b_θ[i, j, k] + cl.b_θ[i, j, km]) / 2
        bᶜᵛ = (cl.b_qᶜ[i, j, k] + cl.b_qᶜ[i, j, km] - cl.b_qᵛ[i, j, k] - cl.b_qᵛ[i, j, km]) / 2
        c_q⁺, c_θ⁺ = saturated_condensate_coefficients(thermodynamics, i, j, k, grid, s, θ[i, j, k])
        c_q⁻, c_θ⁻ = saturated_condensate_coefficients(thermodynamics, i, j, km, grid, s, θ[i, j, km])
    end
    ∂zθ = Δθ / Δz
    ∂zq = Δq / Δz
    ∂zs = (c_q⁺ + c_q⁻) / 2 * ∂zq - (c_θ⁺ + c_θ⁻) / 2 * ∂zθ
    return (; Δz, Δθ, Δq, ∂zθ, ∂zq, b_θᶠ, ∂zs, bᶜᵛ)
end

applied_flux_stability!(::AppliedFluxEnergetics, fields, closure_fields, closure, model, tracers, parameters) =
    applied_flux_stability_pass!(fields, closure_fields, closure, model, tracers, parameters)

@inline tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields) =
    @inbounds fields.cloud.B[i, j, k]

closure_components(closure::TKEBasedTurbulenceClosure) =
    (closure.mixing_length, closure.stability_functions, closure.static_stability, closure.shear, closure.subgrid_distribution,
     closure.heating_budget_flux)

rebuild_component(::Nothing, scalars, n) = nothing, n

component_constructor(::Type{<:StratificationLimitedMomentTimeScale{P}}) where P =
    (ℂᵗ, ℂᴺ, N) -> StratificationLimitedMomentTimeScale{P, typeof(ℂᵗ)}(ℂᵗ, ℂᴺ, N)
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a copy of the [`TKEBasedTurbulenceClosure`](@ref) `closure` in which the scalars named by the
keyword arguments are replaced, whichever component (mixing length, stability functions, static
stability, shear, subgrid distribution, heating budget flux) holds them. [`closure_parameters`](@ref) lists the
coefficients. The non-integer scalars are promoted with the replacements, so that traced scalars
give a closure whose components all have the promoted scalar type.
"""
function with_parameters(closure::TKEBasedTurbulenceClosure{TD}; kw...) where TD
    scalars = closure_scalars(closure)
    names = keys(scalars)
    all(name -> name ∈ names, keys(kw)) ||
        throw(ArgumentError("with_parameters: $(setdiff(keys(kw), names)) are not scalars of the components of " *
                            "$(summary(closure)); they are $(names)."))
    replaced = map(name -> haskey(kw, name) ? kw[name] : scalars[name], names)
    is_count(x) = x isa Integer
    T = promote_type(map(typeof, filter(!is_count, replaced))...)
    promoted = map(x -> is_count(x) ? x : convert(T, x), replaced)

    n = 1
    components = ()
    for component in closure_components(closure)
        component, n = rebuild_component(component, promoted, n)
        components = (components..., component)
    end
    mixing_length, stability_functions, static_stability, shear, subgrid_distribution, heating_budget_flux = components

    return TKEBasedTurbulenceClosure{TD}(mixing_length, stability_functions, static_stability,
                                         closure.minimum_tke, shear,
                                         closure.negative_tke_damping_time_scale, subgrid_distribution,
                                         heating_budget_flux)
end

vapor_excess_component_fields(grid) = (a₁ = CenterField(grid), μ₁ = CenterField(grid), μ₂ = CenterField(grid),
                                       σ₁ = CenterField(grid), σ₂ = CenterField(grid), ωᵐ = CenterField(grid),
                                       μⁱ₁ = CenterField(grid), μⁱ₂ = CenterField(grid),
                                       σⁱ₁ = CenterField(grid), σⁱ₂ = CenterField(grid),
                                       𝕏ⁱ₁ = CenterField(grid), 𝕏ⁱ₂ = CenterField(grid))

@inline function store_vapor_excess_components!(components, i, j, k, (a, μ₁, μ₂, σ₁, σ₂, μⁱ₁, μⁱ₂, σⁱ₁, σⁱ₂, r₁, r₂))
    @inbounds begin
        components.a₁[i, j, k] = a
        components.μ₁[i, j, k] = μ₁
        components.μ₂[i, j, k] = μ₂
        components.σ₁[i, j, k] = σ₁
        components.σ₂[i, j, k] = σ₂
        components.μⁱ₁[i, j, k] = μⁱ₁
        components.μⁱ₂[i, j, k] = μⁱ₂
        components.σⁱ₁[i, j, k] = σⁱ₁
        components.σⁱ₂[i, j, k] = σⁱ₂
        components.𝕏ⁱ₁[i, j, k] = r₁
        components.𝕏ⁱ₂[i, j, k] = r₂
    end
    return nothing
end

@inline function stored_vapor_excess_components(i, j, k, components)
    @inbounds begin
        a = components.a₁[i, j, k]
        return (; a = (a, 1 - a), μ = (components.μ₁[i, j, k], components.μ₂[i, j, k]),
                  σ = (components.σ₁[i, j, k], components.σ₂[i, j, k]), f⁺ = components.aᶜ[i, j, k],
                  ωᵐ = components.ωᵐ[i, j, k],
                  μⁱ = (components.μⁱ₁[i, j, k], components.μⁱ₂[i, j, k]),
                  σⁱ = (components.σⁱ₁[i, j, k], components.σⁱ₂[i, j, k]),
                  𝕏ⁱ = (components.𝕏ⁱ₁[i, j, k], components.𝕏ⁱ₂[i, j, k]),
                  ice_covariance = components.qⁱ′sⁱ′[i, j, k], departure_variance = components.qᵈ′²[i, j, k])
    end
end

@inline AtmosphereModels.supersaturation_statistics(::TurbulenceClosureStatistics, i, j, k, grid, ::FlavorOfTKEClosure,
                                                    closure_fields::TKEClosureFields, model_fields) =
    closure_vapor_excess_statistics(i, j, k, grid, closure_fields.subgrid_cloud)

@inline function closure_vapor_excess_statistics(i, j, k, grid, ::Nothing)
    FT = grid_float_type(grid)
    return (; μ = zero(FT), σ = zero(FT), skewness = zero(FT), f⁺ = zero(FT))
end

@inline closure_vapor_excess_statistics(i, j, k, grid, fields::SubgridCloudFields{<:Any, Nothing}) =
    closure_vapor_excess_statistics(i, j, k, grid, nothing)

@inline closure_vapor_excess_statistics(i, j, k, grid, fields::SubgridCloudFields) =
    stored_vapor_excess_components(i, j, k, fields.cloud)
