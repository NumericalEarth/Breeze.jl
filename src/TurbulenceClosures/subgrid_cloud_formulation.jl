#####
##### The subgrid cloud formulation of `TKEBasedTurbulenceClosure`: SHOC's subgrid cloud scheme on
##### the closure's own mixing length, turbulent kinetic energy and diffusivities
#####
##### Three parts, each switched on or off independently (`nothing` is off):
#####
#####   1. `distribution` (`SubgridDistribution`): the second moments θˡ′², qᵗ′², θˡ′qᵗ′, w′θˡ′, w′qᵗ′
#####      down the gradient, `ℂᵛ τ Kᶜ ∂z a ∂z b` and `-Kᶜ ∂z a`, with the closure's `Kᶜ` and a
#####      moment time scale `τ` built from its `ℓ`, `e` and `N²`; `w′² = ℂʷ₁ (2/3) e`; the third
#####      moment w′³ of SHOC's algebraic diagnosis and its limiter; and the assumed double-Gaussian
#####      distribution of SHOC with its saturation (`pdf_saturation`), which gives the cloud
#####      fraction, the cloud liquid, the liquid flux w′qˡ′ and the buoyancy flux
#####      w′θᵥ′ = w′θˡ′ + A_q w′qᵗ′ + A_c w′qˡ′.
#####   2. `buoyancy_flux` (`CloudBuoyancyFlux`): the TKE equation takes its buoyancy production
#####      from (g / θ₀) w′θᵥ′ instead of -Kᶜ N². The positive part is an explicit source and the
#####      negative part joins the implicit linear coefficient, as the buoyancy sink does without
#####      the formulation. Optionally (`stability = true`) the stability that the mixing length
#####      and the stability functions see becomes N²ᵇ = -(g / θ₀) w′θᵥ′ / Kᶜ at the faces.
#####   3. `cloud` (`ClosureOwnedCloud`): with a microphysics whose subgrid choice is `ClosureCloud`,
#####      the closure supplies the cloud fraction and cloud liquid (`ClosureCloudStatistics`) and
#####      sets the cloud liquid of the model once per time step, after the last stage, to the
#####      liquid of the distribution, keeping the total water and θˡⁱ. Without it the
#####      microphysics keeps its own condensation.
#####
##### The components of part 1 are those of `SimplifiedHigherOrderClosure`: the formulation is
##### shared, the coefficients are this closure's own. The time stepping is the closure's: the
##### moments are diagnosed once per stage, after the diffusivities, and there is no split solve.
#####
##### The fields of part 1 (the moments at faces, w′² and w′³, the cloud at centers) are stored,
##### so that a microphysics correlation may later read the spread of the saturation deficit or
##### the skewness through `subgrid_statisticsᶜᶜᶜ` without changing this structure.
#####

using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶠ

using .SimplifiedHigherOrderClosures: interface_second_moments, vertical_velocity_variance,
                                      third_moment, limit_third_moment, assumed_pdf,
                                      clip_below, shoc_saturation_thermodynamics,
                                      rate_norm_factor

using ..AtmosphereModels: ClosureCloud, ClosureCloudStatistics, subgrid_correlation

#####
##### The time scale of the moments
#####

"""
$(TYPEDEF)

The time scale `τ` of the second moments (`θˡ′² = ℂᵛ₁ τ Kᶜ (∂z θˡ)²`, ...) and of the third-moment
diagnosis, at the faces, from the closure's mixing length `ℓ`, turbulent velocity `√e` and
static stability `N²` there: the return-to-isotropy time scale of SHOC's
`RichardsonStabilityFunctions` limited by stratification as SHOC's
`StratificationLimitedTimeScale`,

    τ₀ = τᵉ / √(1 + τᵉ² N²⁺),    τᵉ = ℂᵗ ℓ / √e,
    τ  = τ₀ / (1 + (τ₀ N⁺ / ℂᴺ)ᵖ)^(1/p),    N⁺ = √(N²⁺ + Nₘᵢₙ²) - Nₘᵢₙ

so that `τ N⁺ < ℂᴺ`. `ℂᴺ = Inf` removes the stratification limit. The exponent `p` is a type
parameter.

$(TYPEDFIELDS)
"""
struct StratificationLimitedMomentTimeScale{P, FT}
    "Eddy turnovers in the time scale, `τᵉ = ℂᵗ ℓ / √e`"
    ℂᵗ :: FT
    "Largest `τ N`"
    ℂᴺ :: FT
    "Frequency that regularizes `N⁺` at `N² = 0`, s⁻¹"
    minimum_buoyancy_frequency :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`StratificationLimitedMomentTimeScale`](@ref) with the values of the fixed smooth SHOC
configuration: `ℂᵗ = 1`, `ℂᴺ = 4.5`, `exponent = 4`.
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
    N²⁺ = max(N², zero(N²))
    τᵉ = ts.ℂᵗ * ℓ / u
    τ₀ = τᵉ / sqrt(1 + τᵉ * τᵉ * N²⁺)
    Nₘᵢₙ = ts.minimum_buoyancy_frequency
    N⁺ = N²⁺ / (sqrt(N²⁺ + Nₘᵢₙ * Nₘᵢₙ) + Nₘᵢₙ)
    return τ₀ / rate_norm_factor(Val(P), τ₀ * N⁺ / ts.ℂᴺ)
end

#####
##### Part 1: the subgrid distribution
#####

"""
$(TYPEDEF)

The subgrid distribution of θˡ, qᵗ and w of a [`SubgridCloudFormulation`](@ref): the
components of `SimplifiedHigherOrderClosure` for the second moments, the third moment and its
limiter, the assumed PDF and its saturation, evaluated on the mixing length, turbulent kinetic
energy, diffusivities and static stability of [`TKEBasedTurbulenceClosure`](@ref).

$(TYPEDFIELDS)
"""
struct SubgridDistribution{TS, SM, TM, SB, SK, PDF, PS, CP, C}
    "Time scale of the moments ([`StratificationLimitedMomentTimeScale`](@ref))"
    time_scale :: TS
    "Second moments ([`DowngradientSecondMoments`](@ref)); `ℂᶠ`… of the surface are not used"
    second_moments :: SM
    "Third moment of vertical velocity ([`AlgebraicThirdMoment`](@ref)), or `nothing` for none (unskewed)"
    third_moment :: TM
    "Bound of the stratification number of the third-moment diagnosis"
    stratification_number_bound :: SB
    "Limiter of the third moment"
    skewness :: SK
    "Assumed PDF ([`DoubleGaussianPDF`](@ref) or [`SmoothDoubleGaussianPDF`](@ref))"
    assumed_pdf :: PDF
    "Saturation of the assumed PDF ([`FixedConstantsLiquidSaturation`](@ref) or [`ModelThermodynamicsSaturation`](@ref))"
    saturation :: PS
    "Clipping policy of the SHOC components"
    clipping :: CP
    "SHOC's constants the components read (`ggr`, `basetemp`, `mintke`, …)"
    constants :: C
end

Adapt.@adapt_structure SubgridDistribution

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`SubgridDistribution`](@ref). The defaults are the components of the fixed smooth SHOC
configuration with published coefficients.
"""
function SubgridDistribution(FT::DataType = Oceananigans.defaults.FloatType;
                             time_scale = StratificationLimitedMomentTimeScale(FT),
                             second_moments = DowngradientSecondMoments(FT),
                             third_moment = AlgebraicThirdMoment(FT),
                             stratification_number_bound = RationalSaturatedStratificationNumber(FT),
                             skewness = PolynomialSaturatedSkewness(FT),
                             assumed_pdf = DoubleGaussianPDF(FT),
                             saturation = FixedConstantsLiquidSaturation(),
                             clipping = HardClipping(),
                             constants = SHOCConstants(FT))
    return SubgridDistribution(time_scale, second_moments, third_moment, stratification_number_bound,
                               skewness, assumed_pdf, saturation, clipping, constants)
end

convert_eltype(::Type{FT}, d::SubgridDistribution) where FT =
    SubgridDistribution(convert_eltype(FT, d.time_scale), map(c -> convert_component(FT, c),
                        (d.second_moments, d.third_moment, d.stratification_number_bound, d.skewness,
                         d.assumed_pdf, d.saturation, d.clipping, d.constants))...)

convert_component(FT, ::Nothing) = nothing
convert_component(FT, c) = convert_eltype(FT, c)

Base.summary(d::SubgridDistribution) = string("SubgridDistribution(", nameof(typeof(d.assumed_pdf)), ", ",
                                              nameof(typeof(d.saturation)), ", third moment: ",
                                              d.third_moment === nothing ? "none" : nameof(typeof(d.third_moment)), ")")

#####
##### Part 2: the cloud buoyancy flux
#####

"""
$(TYPEDEF)

The buoyancy production of the TKE equation from the buoyancy flux of the subgrid distribution,
`B = (g / θ₀) w′θᵥ′` with `w′θᵥ′ = w′θˡ′ + A_q w′qᵗ′ + A_c w′qˡ′`, instead of `-Kᶜ N²`. The
positive part enters the stage tendency; the negative part enters the implicit linear coefficient.
`θ₀` is SHOC's `basetemp` with [`FixedConstantsLiquidSaturation`](@ref) and the level's θˡⁱ with
[`ModelThermodynamicsSaturation`](@ref).

With `stability = true` the static stability seen by the mixing length and by the stability
functions becomes, at the faces where `Kᶜ` is positive, `N²ᵇ = -B / Kᶜ`, the stratification
consistent with the cloud buoyancy flux; the diffusivities are then formed again from it.

$(TYPEDFIELDS)
"""
struct CloudBuoyancyFlux
    "Whether the stability of the diffusivities and the mixing length is `-B / Kᶜ`"
    stability :: Bool
end

CloudBuoyancyFlux(; stability = false) = CloudBuoyancyFlux(stability)
Base.summary(b::CloudBuoyancyFlux) = string("CloudBuoyancyFlux(stability = ", b.stability, ")")

#####
##### Part 3: cloud owned by the closure
#####

"""
$(TYPEDEF)

The closure owns the cloud of the model: with a microphysics whose subgrid choice is
`ClosureCloud`, the closure supplies `ClosureCloudStatistics` (the cloud fraction and liquid of
the subgrid distribution) and, once per time step after the last stage, sets the cloud liquid of
the model to the liquid of the distribution diagnosed from the state at the end of the step,
moving the difference to or from the vapor; the total water and the liquid-ice potential
temperature are unchanged. With any other microphysics it does nothing.
"""
struct ClosureOwnedCloud end
Base.summary(::ClosureOwnedCloud) = "ClosureOwnedCloud"

#####
##### The formulation
#####

"""
$(TYPEDEF)

SHOC's subgrid cloud scheme as a property of [`TKEBasedTurbulenceClosure`](@ref), in three
independently switchable parts (`nothing` switches a part off):

1. `distribution`: a [`SubgridDistribution`](@ref);
2. `buoyancy_flux`: a [`CloudBuoyancyFlux`](@ref), which needs the distribution;
3. `cloud`: [`ClosureOwnedCloud`](@ref), which needs the distribution and a microphysics with
   `subgrid = ClosureCloud(...)`.

$(TYPEDFIELDS)
"""
struct SubgridCloudFormulation{D, B, O}
    "The subgrid distribution, or `nothing`"
    distribution :: D
    "The cloud buoyancy flux in the TKE equation, or `nothing`"
    buoyancy_flux :: B
    "Cloud ownership, or `nothing`"
    cloud :: O
end

Adapt.@adapt_structure SubgridCloudFormulation

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`SubgridCloudFormulation`](@ref), all three parts on by default.
"""
function SubgridCloudFormulation(FT::DataType = Oceananigans.defaults.FloatType;
                                 distribution = SubgridDistribution(FT),
                                 buoyancy_flux = CloudBuoyancyFlux(),
                                 cloud = ClosureOwnedCloud())
    if isnothing(distribution) && !(isnothing(buoyancy_flux) && isnothing(cloud))
        throw(ArgumentError("The cloud buoyancy flux and cloud ownership read the subgrid distribution, " *
                            "but `distribution = nothing`."))
    end
    return SubgridCloudFormulation(distribution, buoyancy_flux, cloud)
end

convert_eltype(::Type{FT}, f::SubgridCloudFormulation) where FT =
    SubgridCloudFormulation(convert_component(FT, f.distribution), f.buoyancy_flux, f.cloud)

Base.summary(f::SubgridCloudFormulation) =
    string("SubgridCloudFormulation(", summary(f.distribution), ", ", summary(f.buoyancy_flux), ", ", summary(f.cloud), ")")

#####
##### Fields
#####

"""
$(TYPEDEF)

The fields of a [`SubgridCloudFormulation`](@ref) in the closure fields of
[`TKEBasedTurbulenceClosure`](@ref): the moments at faces (`τ`, `θˡ′²`, `qᵗ′²`, `θˡ′qᵗ′`,
`w′θˡ′`, `w′qᵗ′`, `w′³`) and the cloud at centers (`w′²`, cloud fraction `C`, liquid `qˡ`, its
variance `qˡ′²`, the fluxes `w′qˡ′` and `w′θᵥ′`, and the buoyancy production `B`).
"""
struct SubgridCloudFields{BF, CO, M, C}
    buoyancy_flux :: BF
    cloud_ownership :: CO
    moments :: M
    cloud :: C
end

Adapt.adapt_structure(to, f::SubgridCloudFields) =
    SubgridCloudFields(f.buoyancy_flux, f.cloud_ownership, adapt(to, f.moments), adapt(to, f.cloud))

build_subgrid_cloud_fields(grid, ::SubgridCloudFormulation{Nothing}) = nothing

function build_subgrid_cloud_fields(grid, f::SubgridCloudFormulation)
    moments = (τ = ZFaceField(grid), θˡ′² = ZFaceField(grid), qᵗ′² = ZFaceField(grid), θˡ′qᵗ′ = ZFaceField(grid),
               w′θˡ′ = ZFaceField(grid), w′qᵗ′ = ZFaceField(grid), w′³ = ZFaceField(grid))
    cloud = (w′² = CenterField(grid), C = CenterField(grid), qˡ = CenterField(grid), qˡ′² = CenterField(grid),
             w′qˡ′ = CenterField(grid), w′θᵥ′ = CenterField(grid), B = CenterField(grid))
    return SubgridCloudFields(f.buoyancy_flux, f.cloud, moments, cloud)
end

#####
##### Specific values from densities, for the kernels
#####

# `f / ρ` at (i, j, k); with `ρ = nothing`, `f` itself
struct SpecificValue{F, R}
    field :: F
    density :: R
end

Adapt.adapt_structure(to, s::SpecificValue) = SpecificValue(adapt(to, s.field), adapt(to, s.density))
@inline Base.getindex(s::SpecificValue, i, j, k) = @inbounds s.field[i, j, k] / s.density[i, j, k]
@inline Base.getindex(s::SpecificValue{<:Any, Nothing}, i, j, k) = @inbounds s.field[i, j, k]

# The total water of the distribution, the vapor and the cloud liquid, (ρqᵛ + ρqᶜˡ) / ρ
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

cloud_liquid_density(microphysical_fields) =
    hasproperty(microphysical_fields, :ρqᶜˡ) ? microphysical_fields.ρqᶜˡ : nothing

function distribution_potential_temperature_density(model)
    AtmosphereModels.thermodynamic_density_name(model.formulation) === :ρθ ||
        throw(ArgumentError("The subgrid distribution of TKEBasedTurbulenceClosure requires the liquid-ice " *
                            "potential temperature formulation, but the model formulation is $(summary(model.formulation))."))
    return Oceananigans.prognostic_fields(model.formulation).ρθ
end

# The state the distribution reads: θˡⁱ, qᵗ, e, w, p and the saturation thermodynamics. `e_density`
# is the density of the specific TKE `ρe` holds: `nothing` within `update_state!`, where the tracer
# holds its specific value, and `ρ` after the stages, where it holds its density.
function distribution_state(model, e_density)
    ρ = AtmosphereModels.total_density(model.dynamics)
    θ = SpecificValue(distribution_potential_temperature_density(model), ρ)
    qᵗ = SpecificTotalWater(model.moisture_density, cloud_liquid_density(model.microphysical_fields), ρ)
    e = SpecificValue(model.tracers[TKE_NAME], e_density)
    w = model.velocities.w
    p = AtmosphereModels.dynamics_pressure(model.dynamics)
    thermodynamics = shoc_saturation_thermodynamics(model)
    return (; θ, qᵗ, e, w, p, thermodynamics)
end

#####
##### The column computation of the distribution
#####

@inline subgrid_third_moment(::Nothing, d, c, args...) = zero(eltype(args[1]))
@inline subgrid_third_moment(tm, d, c, args...) = third_moment(tm, d.stratification_number_bound, c, args...)

# g / θ₀ of the buoyancy production
@inline cloud_buoyancy_parameter(::FixedConstantsLiquidSaturation, c, thermodynamics, θ) = c.ggr / c.basetemp
@inline cloud_buoyancy_parameter(::ModelThermodynamicsSaturation, c, thermodynamics, θ) =
    thermodynamics.constants.gravitational_acceleration / θ

@inline function subgrid_distribution_column!(m, cl, i, j, grid, d, closure, closure_fields, θ, qᵗ, e, w, p, thermodynamics)
    Nz = size(grid, 3)
    FT = eltype(grid)
    c = d.constants
    cp = d.clipping
    sm = d.second_moments
    eᵐⁱⁿ = closure.minimum_tke
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())

    @inbounds begin
        # The second moments at the interior faces, down the gradient
        for k in 2:Nz
            Δz = Δzᶜᶜᶠ(i, j, k, grid)
            Kᶜ = closure_fields.Kᶜ[i, j, k]
            Kᵘ = closure_fields.Kᵘ[i, j, k]
            ℓ = closure_fields.ℓᵐ[i, j, k]
            u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
            τ = moment_time_scale(d.time_scale, ℓ, u, closure_fields.N²[i, j, k])
            τ = ifelse(isfinite(τ), τ, zero(FT))

            θˡ′², qᵗ′², θˡ′qᵗ′, w′θˡ′, w′qᵗ′, _, _, _ =
                interface_second_moments(sm, τ, Kᶜ, Kᵘ, 1 / Δz, θ[i, j, k] - θ[i, j, k-1], qᵗ[i, j, k] - qᵗ[i, j, k-1],
                                         zero(FT), zero(FT), zero(FT))

            m.τ[i, j, k] = τ
            m.θˡ′²[i, j, k] = θˡ′²
            m.qᵗ′²[i, j, k] = qᵗ′²
            m.θˡ′qᵗ′[i, j, k] = θˡ′qᵗ′
            m.w′θˡ′[i, j, k] = w′θˡ′
            m.w′qᵗ′[i, j, k] = w′qᵗ′
        end

        # The surface face takes the moments of the face above (the closure leaves the surface
        # flux of TKE to the boundary condition, and its buoyancy production at the first level is
        # that of the face above, too); the top face has none
        k′ = min(2, Nz)
        m.τ[i, j, 1] = m.τ[i, j, k′]
        m.θˡ′²[i, j, 1] = m.θˡ′²[i, j, k′]
        m.qᵗ′²[i, j, 1] = m.qᵗ′²[i, j, k′]
        m.θˡ′qᵗ′[i, j, 1] = m.θˡ′qᵗ′[i, j, k′]
        m.w′θˡ′[i, j, 1] = m.w′θˡ′[i, j, k′]
        m.w′qᵗ′[i, j, 1] = m.w′qᵗ′[i, j, k′]
        m.τ[i, j, Nz+1] = 0
        m.θˡ′²[i, j, Nz+1] = 0
        m.qᵗ′²[i, j, Nz+1] = 0
        m.θˡ′qᵗ′[i, j, Nz+1] = 0
        m.w′θˡ′[i, j, Nz+1] = 0
        m.w′qᵗ′[i, j, Nz+1] = 0
        m.w′³[i, j, 1] = 0
        m.w′³[i, j, Nz+1] = 0

        # The vertical velocity variance at the centers
        for k in 1:Nz
            cl.w′²[i, j, k] = vertical_velocity_variance(sm, max(e[i, j, k], eᵐⁱⁿ))
        end

        # The third moment at the interior faces
        w′²ᵐⁱⁿ = (convert(FT, 2) / 3) * c.mintke
        for k in 2:Nz
            w′²ᶠ = clip_below(cp, (cl.w′²[i, j, k] + cl.w′²[i, j, k-1]) / 2, w′²ᵐⁱⁿ, c.mintke)
            θᶠ = (θ[i, j, k] + θ[i, j, k-1]) / 2
            thedz = 1 / Δzᶜᶜᶠ(i, j, k, grid)
            thedz2 = 1 / (Δzᶜᶜᶜ(i, j, k-1, grid) + Δzᶜᶜᶜ(i, j, k, grid))
            w′³ = subgrid_third_moment(d.third_moment, d, c, m.τ[i, j, k], closure_fields.N²[i, j, k], w′²ᶠ, θᶠ, thedz, thedz2,
                                       m.θˡ′²[i, j, k+1] - m.θˡ′²[i, j, k-1],
                                       m.w′θˡ′[i, j, k+1] - m.w′θˡ′[i, j, k-1],
                                       cl.w′²[i, j, k] - cl.w′²[i, j, k-1],
                                       max(e[i, j, k], eᵐⁱⁿ) - max(e[i, j, k-1], eᵐⁱⁿ),
                                       m.w′θˡ′[i, j, k])
            m.w′³[i, j, k] = limit_third_moment(d.skewness, cp, w′²ᶠ, w′³)
        end

        # The assumed PDF at the centers
        interface_moments(k) = (m.w′³[i, j, k], m.θˡ′²[i, j, k], m.w′θˡ′[i, j, k],
                                m.θˡ′qᵗ′[i, j, k], m.w′qᵗ′[i, j, k], m.qᵗ′²[i, j, k])
        for k in 1:Nz
            zᶜ = znode(i, j, k, grid, ccc...)
            zˡ = znode(i, j, k, grid, ccf...)
            zᵘ = znode(i, j, k+1, grid, ccf...)
            wᶜ = (w[i, j, k] + w[i, j, k+1]) / 2
            θᵏ = θ[i, j, k]
            C, qˡ, qˡ′², w′qˡ′, w′θᵥ′ = assumed_pdf(d.assumed_pdf, d.saturation, thermodynamics, cp, c, zᵘ, zˡ, zᶜ,
                                                   p[i, j, k], θᵏ, wᶜ, qᵗ[i, j, k], cl.w′²[i, j, k],
                                                   interface_moments(k), interface_moments(k+1))
            cl.C[i, j, k] = C
            cl.qˡ[i, j, k] = qˡ
            cl.qˡ′²[i, j, k] = qˡ′²
            cl.w′qˡ′[i, j, k] = w′qˡ′
            cl.w′θᵥ′[i, j, k] = w′θᵥ′
            cl.B[i, j, k] = cloud_buoyancy_parameter(d.saturation, c, thermodynamics, θᵏ) * w′θᵥ′
        end
    end

    return nothing
end

@kernel function _compute_subgrid_distribution!(moments, cloud, grid, closure, closure_fields, θ, qᵗ, e, w, p, thermodynamics)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    d = closure_ij.subgrid_cloud_formulation.distribution
    subgrid_distribution_column!(moments, cloud, i, j, grid, d, closure_ij, closure_fields, θ, qᵗ, e, w, p, thermodynamics)
end

function compute_subgrid_distribution!(fields::SubgridCloudFields, closure_fields, closure, model, e_density)
    grid = model.grid
    s = distribution_state(model, e_density)
    launch!(grid.architecture, grid, :xy, _compute_subgrid_distribution!,
            fields.moments, fields.cloud, grid, closure, closure_fields, s.θ, s.qᵗ, s.e, s.w, s.p, s.thermodynamics)
    return nothing
end

#####
##### Coupling to the closure
#####

# Within `update_state!`, after the diffusivities and before the implicit coefficient
function compute_subgrid_cloud!(fields::SubgridCloudFields, closure_fields, closure, model, tracers, parameters)
    compute_subgrid_distribution!(fields, closure_fields, closure, model, nothing)
    cloud_buoyancy_stability!(fields.buoyancy_flux, fields, closure_fields, closure, model, tracers, parameters)
    return nothing
end

cloud_buoyancy_stability!(buoyancy_flux, args...) = nothing

# N²ᵇ = -B / Kᶜ at the faces, then the mixing length and the diffusivities again from it
@kernel function _cloud_buoyancy_stability!(N², grid, B, Kᶜ)
    i, j, k = @index(Global, NTuple)
    FT = eltype(grid)
    Bᶠ = ℑzᵃᵃᶠ(i, j, k, grid, B)
    Kᶜᵢ = @inbounds Kᶜ[i, j, k]
    resolved = (Kᶜᵢ > sqrt(floatmin(FT))) & (k > 1)
    @inbounds N²[i, j, k] = ifelse(resolved, -Bᶠ / ifelse(resolved, Kᶜᵢ, one(FT)), N²[i, j, k])
end

function cloud_buoyancy_stability!(buoyancy_flux::CloudBuoyancyFlux, fields, closure_fields, closure, model, tracers, parameters)
    buoyancy_flux.stability || return nothing
    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, parameters, _cloud_buoyancy_stability!, closure_fields.N², grid, fields.cloud.B, closure_fields.Kᶜ)
    compute_mixing_length!(closure_fields.ℓᵐ, closure_fields.ℓˢ, grid, closure, tracers[TKE_NAME], closure_fields.N²)
    launch!(arch, grid, parameters, _compute_tke_closure_fields!, closure_fields, grid, closure, model.velocities, tracers)
    return nothing
end

# The buoyancy production of the TKE equation: the distribution's with the cloud buoyancy flux
@inline tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:CloudBuoyancyFlux}, closure_fields) =
    @inbounds fields.cloud.B[i, j, k]

@inline tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, ::SubgridCloudFields{Nothing}, closure_fields) =
    ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, closure_fields.Kᶜ, closure_fields.N²)

#####
##### Part 3: the cloud of the model
#####

@inline AtmosphereModels.subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::ClosureCloud, ::FlavorOfTKEClosure, fields::TKEClosureFields) =
    closure_cloud_statistics(i, j, k, grid, fields.subgrid_cloud)

@inline closure_cloud_statistics(i, j, k, grid, fields) = ClosureCloudStatistics(zero(grid), zero(grid))

@inline closure_cloud_statistics(i, j, k, grid, fields::SubgridCloudFields{<:Any, ClosureOwnedCloud}) =
    @inbounds ClosureCloudStatistics(fields.cloud.C[i, j, k], fields.cloud.qˡ[i, j, k])

# The liquid of the distribution becomes the cloud liquid of the model, bounded by the water there is
@kernel function _set_closure_cloud_liquid!(ρqᵛ, ρqᶜˡ, ρ, qˡ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        ρqᵗ = ρqᵛ[i, j, k] + ρqᶜˡ[i, j, k]
        qᵗ = ρqᵗ / ρᵢ
        qᶜˡ = min(max(0, qˡ[i, j, k]), max(0, qᵗ))
        ρqᶜˡ[i, j, k] = ρᵢ * qᶜˡ
        ρqᵛ[i, j, k] = ρqᵗ - ρᵢ * qᶜˡ
    end
end

function Oceananigans.TimeSteppers.step_closure_prognostics!(closure_fields::TKEClosureFields, closure::FlavorOfTKEClosure, model, Δt)
    step_subgrid_cloud!(closure_fields.subgrid_cloud, closure_fields, closure, model)
    return nothing
end

step_subgrid_cloud!(fields, closure_fields, closure, model) = nothing

function step_subgrid_cloud!(fields::SubgridCloudFields{<:Any, ClosureOwnedCloud}, closure_fields, closure, model)
    subgrid_correlation(model.microphysics) isa ClosureCloud || return nothing
    ρqᶜˡ = cloud_liquid_density(model.microphysical_fields)
    isnothing(ρqᶜˡ) && return nothing

    # The distribution of the state at the end of the step, with the diffusivities of its last stage
    ρ = AtmosphereModels.total_density(model.dynamics)
    compute_subgrid_distribution!(fields, closure_fields, closure, model, ρ)

    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _set_closure_cloud_liquid!, model.moisture_density, ρqᶜˡ, ρ, fields.cloud.qˡ)
    return nothing
end
