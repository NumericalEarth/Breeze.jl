"""
    SimplifiedHigherOrderClosure

The Simplified Higher-Order Closure of [Bogenschutz and Krueger (2013)](@cite Bogenschutz2013), as E3SM
implements it (`shoc.F90`, pinned at E3SM `ba53243f`), as a configuration of [`TKEBasedTurbulenceClosure`](@ref):
the TKE closure with SHOC's mixing length ([`BlendedMixingLength`](@ref)), SHOC's return-to-isotropy time scale
and diffusivities as stability functions ([`ColumnStabilityDamping`](@ref)) and a [`DoubleGaussian`](@ref)
subgrid distribution. A closure is a `SimplifiedHigherOrderClosure` when it has these three component types.
"""
const SimplifiedHigherOrderClosure{TD} = TKEBasedTurbulenceClosure{TD, <:BlendedMixingLength, <:ColumnStabilityDamping,
                                                                   <:Any, <:Any, <:Any, <:DoubleGaussian} where TD
"""
A shorter name for [`SimplifiedHigherOrderClosure`](@ref).
"""
const SHOC = SimplifiedHigherOrderClosure

shoc_distribution(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    DoubleGaussian(FT; time_scale = ReturnToIsotropyTimeScale(),
                       second_moments = DowngradientSecondMoments(FT),
                       third_moment = AlgebraicThirdMoment(FT),
                       stratification_number_bound = RationalSaturatedStratificationNumber(FT),
                       skewness = PolynomialSaturatedSkewness(FT),
                       assumed_pdf = DoubleGaussianPDF(FT),
                       kw...)
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`SimplifiedHigherOrderClosure`](@ref): a [`TKEBasedTurbulenceClosure`](@ref) with the components of
E3SM `shoc.F90` and its published coefficients.
"""
function SimplifiedHigherOrderClosure(time_discretization = VerticallyImplicitTimeDiscretization(),
                                      FT = Oceananigans.defaults.FloatType;
                                      mixing_length = BlendedMixingLength(Float64),
                                      stability_functions = ColumnStabilityDamping(Float64),
                                      static_stability = SaturationWeightedStaticStability(CondensateFluxFraction(stratification = MeanLiquidStratification())),
                                      subgrid_distribution = shoc_distribution(Float64),
                                      kw...)

    return TKEBasedTurbulenceClosure(time_discretization, FT; mixing_length, stability_functions, static_stability,
                                     subgrid_distribution, kw...)
end

SimplifiedHigherOrderClosure(FT::DataType; kw...) =
    SimplifiedHigherOrderClosure(VerticallyImplicitTimeDiscretization(), FT; kw...)

validate_moment_time_scale(::ColumnStabilityDamping, ::ReturnToIsotropyTimeScale) = nothing
validate_moment_time_scale(stability_functions, ::ReturnToIsotropyTimeScale) =
    throw(ArgumentError("ReturnToIsotropyTimeScale takes the time scale of the moments from the stability functions, " *
                        "which $(summary(stability_functions)) do not have; use ColumnStabilityDamping or another time scale."))

function Base.summary(closure::SimplifiedHigherOrderClosure)
    TD = nameof(typeof(time_discretization(closure)))
    return string("SimplifiedHigherOrderClosure{$TD}")
end

static_stability_summary(static_stability) = summary(static_stability)
static_stability_summary(ss::SaturationWeightedStaticStability) = string(summary(ss), "(", sprint(show, ss.fraction), ")")

function Base.show(io::IO, closure::SimplifiedHigherOrderClosure)
    d = closure.subgrid_distribution
    print(io, summary(closure), '\n',
              "├── mixing_length: ", closure.mixing_length, '\n',
              "├── stability_functions: ", summary(closure.stability_functions), '\n')
    show_stability_function_lines(io, closure.stability_functions, "│   ")
    print(io, '\n',
              "├── static_stability: ", static_stability_summary(closure.static_stability), '\n',
              "├── subgrid_distribution: DoubleGaussian", '\n',
              "│   ├── time_scale: ", summary(d.time_scale), '\n',
              "│   ├── second_moments: ", summary(d.second_moments), '\n',
              "│   ├── third_moment: ", summary(d.third_moment), '\n',
              "│   ├── stratification_number_bound: ", summary(d.stratification_number_bound), '\n',
              "│   ├── skewness: ", summary(d.skewness), '\n',
              "│   └── assumed_pdf: ", summary(d.assumed_pdf), '\n')
    isnothing(closure.heating_budget_flux) || print(io, "├── heating_budget_flux: ", summary(closure.heating_budget_flux), '\n')
    print(io, "└── minimum_tke: ", prettysummary(closure.minimum_tke))
end
