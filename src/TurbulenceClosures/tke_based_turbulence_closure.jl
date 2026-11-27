using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    shearᶜᶜᶠ, ℑbzᵃᵃᶜ, mask_diffusivity, turbulent_velocityᶜᶜᶜ

using Oceananigans.TurbulenceClosures: height_above_bottomᶜᶜᶠ, height_above_bottomᶜᶜᶜ, shear_production
using Oceananigans.Units: minute

#####
##### Stability functions
#####
"""
$(TYPEDEF)

Constant stability functions for [`TKEBasedTurbulenceClosure`](@ref): ``ℓᵘ = ℂᵘ ℓ``, ``ℓᶜ = ℂᶜ ℓ``, ``ℓᵉ = ℂᵉ ℓ``
and ``ε = ℂᴰ e^{3/2} / ℓ``.
"""
Base.@kwdef struct ConstantStabilityFunctions{FT}
    ℂᵘ :: FT = 0.149
    ℂᶜ :: FT = 0.201
    ℂᵉ :: FT = 0.298
    ℂᴰ :: FT = 0.388
end

ConstantStabilityFunctions(ℂᵘ, ℂᶜ, ℂᵉ, ℂᴰ) = ConstantStabilityFunctions(promote(ℂᵘ, ℂᶜ, ℂᵉ, ℂᴰ)...)

Base.summary(sf::ConstantStabilityFunctions{FT}) where FT = "ConstantStabilityFunctions{$FT}"

function show_stability_function_lines(io::IO, sf::ConstantStabilityFunctions, prefix)
    print(io, prefix, "├── ℂᵘ: ", prettysummary(sf.ℂᵘ), '\n',
              prefix, "├── ℂᶜ: ", prettysummary(sf.ℂᶜ), '\n',
              prefix, "├── ℂᵉ: ", prettysummary(sf.ℂᵉ), '\n',
              prefix, "└── ℂᴰ: ", prettysummary(sf.ℂᴰ))
    return nothing
end

function Base.show(io::IO, sf::ConstantStabilityFunctions)
    print(io, summary(sf), '\n')
    show_stability_function_lines(io, sf, "")
end

#####
##### Mixing length
#####
"""
$(TYPEDEF)

Supertype of the primary mixing length ``ℓ`` of [`TKEBasedTurbulenceClosure`](@ref), built from the wall length
``ℂˢ z`` and the buoyancy penetration depth ``\\sqrt{e} / N``.
"""
abstract type AbstractMixingLength end

@inline inverse_buoyancy_length(ml, u, N²) = sqrt(max(N², zero(N²))) / (buoyancy_length_coefficient(ml) * u)
"""
$(TYPEDEF)

The mixing length ``ℓ = \\min(ℂˢ z, ℓᵇ)``, the smaller of the wall length and the buoyancy penetration depth
at the eddy's own level ([Deardorff 1980](@cite Deardorff1980)).
"""
Base.@kwdef struct LocalMinimumMixingLength{FT} <: AbstractMixingLength
    ℂˢ :: FT = 1.316 # coefficient of the wall length ℓ = ℂˢ z
end
"""
$(TYPEDEF)

The mixing length from the parcel displacements of [Bougeault and Lacarrère (1989)](@cite BougeaultLacarrere1989),
``ℓ = ℂˢ \\min(ℓ↑, ℓ↓)``.
"""
Base.@kwdef struct IntegralMixingLength{FT} <: AbstractMixingLength
    ℂˢ :: FT = 1.316 # coefficient of the wall length ℓ = ℂˢ z
end
"""
$(TYPEDEF)

A gradient-limited mixing length ([Gaspar et al. 1990](@cite Gaspar1990)) with wall coefficient ``ℂˢ``, interior
slope ``ℂⁱ`` and buoyancy-length coefficient ``ℂˡᵇ``:
``ℓ(z) = \\min(ℂˢ z, \\min_{z′} [ℂˡᵇ \\sqrt{e} / N (z′) + ℂⁱ |z - z′|])``.
"""
struct GradientLimitedMixingLength{FT} <: AbstractMixingLength
    ℂˢ :: FT  # wall length coefficient
    ℂⁱ :: FT  # interior envelope slope
    ℂˡᵇ :: FT # coefficient of the buoyancy length ℂˡᵇ √e / N
end

function GradientLimitedMixingLength(; ℂˢ = 1.316, ℂⁱ = ℂˢ, ℂˡᵇ = 1)
    ℂˢ, ℂⁱ, ℂˡᵇ = promote(ℂˢ, ℂⁱ, ℂˡᵇ)
    return GradientLimitedMixingLength(ℂˢ, ℂⁱ, ℂˡᵇ)
end

GradientLimitedMixingLength(ℂˢ::Number) = GradientLimitedMixingLength(ℂˢ, ℂˢ)
GradientLimitedMixingLength(ℂˢ::Number, ℂⁱ::Number) = GradientLimitedMixingLength(promote(ℂˢ, ℂⁱ, one(ℂˢ))...)
GradientLimitedMixingLength{FT}(ℂˢ) where FT = GradientLimitedMixingLength{FT}(ℂˢ, ℂˢ)
GradientLimitedMixingLength{FT}(ℂˢ, ℂⁱ) where FT = GradientLimitedMixingLength(convert(FT, ℂˢ), convert(FT, ℂⁱ), one(FT))

Adapt.@adapt_structure LocalMinimumMixingLength
Adapt.@adapt_structure GradientLimitedMixingLength

@inline interior_slope(ml::GradientLimitedMixingLength) = ml.ℂⁱ
@inline interior_slope(ml::AbstractMixingLength) = ml.ℂˢ

@inline buoyancy_length_coefficient(ml::GradientLimitedMixingLength) = ml.ℂˡᵇ
@inline buoyancy_length_coefficient(ml::AbstractMixingLength) = 1

Base.summary(ml::LocalMinimumMixingLength{FT}) where FT = "LocalMinimumMixingLength{$FT}"
Base.summary(ml::IntegralMixingLength{FT}) where FT = "IntegralMixingLength{$FT}"
Base.summary(ml::GradientLimitedMixingLength{FT}) where FT = "GradientLimitedMixingLength{$FT}"

Base.show(io::IO, ml::AbstractMixingLength) = print(io, summary(ml), " (ℂˢ = ", prettysummary(ml.ℂˢ), ")")
Base.show(io::IO, ml::GradientLimitedMixingLength) =
    print(io, summary(ml), " (ℂˢ = ", prettysummary(ml.ℂˢ), ", ℂⁱ = ", prettysummary(ml.ℂⁱ), ", ℂˡᵇ = ", prettysummary(ml.ℂˡᵇ), ")")

@inline convert_eltype(::Type{FT}, ml::LocalMinimumMixingLength) where FT = LocalMinimumMixingLength{FT}(convert(FT, ml.ℂˢ))
@inline convert_eltype(::Type{FT}, ml::IntegralMixingLength) where FT = IntegralMixingLength{FT}(convert(FT, ml.ℂˢ))
@inline convert_eltype(::Type{FT}, ml::GradientLimitedMixingLength) where FT =
    GradientLimitedMixingLength(convert(FT, ml.ℂˢ), convert(FT, ml.ℂⁱ), convert(FT, ml.ℂˡᵇ))

#####
##### The closure
#####
"""
$(TYPEDEF)

A vertical eddy-diffusivity closure with a prognostic turbulent kinetic energy ``e``, in the spirit of CATKE
([Wagner et al. 2025](@cite Wagner25catke)): ``Kᵘ = 𝕊ᵘ ℓ \\sqrt{e}``, ``Kᶜ = 𝕊ᶜ ℓ \\sqrt{e}``, ``Kᵉ = 𝕊ᵉ ℓ \\sqrt{e}``
and ``ε = 𝕊ᴰ e^{3/2} / ℓ``.
"""
struct TKEBasedTurbulenceClosure{TD, ML, SF, SS, FT, SH, SD, HB} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    mixing_length :: ML                   # the primary mixing length ℓ
    stability_functions :: SF             # 𝕊ᵘ, 𝕊ᶜ, 𝕊ᵉ, 𝕊ᴰ
    static_stability :: SS                # how N² is diagnosed
    minimum_tke :: FT                     # floor on e inside √e, m² s⁻²
    shear :: SH                           # the squared shear S² and its floor S²ᵐⁱⁿ in Ri = N² / (S² + S²ᵐⁱⁿ)
    negative_tke_damping_time_scale :: FT # time scale on which negative e is damped, s
    subgrid_distribution :: SD            # `nothing`, or a `DoubleGaussian`
    heating_budget_flux :: HB             # `nothing`, or a `HeatingBudgetFlux`
end

const TKEClosureArray{TD} = AbstractArray{<:TKEBasedTurbulenceClosure{TD}} where TD

function Adapt.adapt_structure(to, closure::TKEBasedTurbulenceClosure{TD}) where TD
    return TKEBasedTurbulenceClosure{TD}(adapt(to, closure.mixing_length),
                                         adapt(to, closure.stability_functions),
                                         adapt(to, closure.static_stability),
                                         adapt(to, closure.minimum_tke),
                                         adapt(to, closure.shear),
                                         adapt(to, closure.negative_tke_damping_time_scale),
                                         adapt(to, closure.subgrid_distribution),
                                         adapt(to, closure.heating_budget_flux))
end
"""Either a single `TKEBasedTurbulenceClosure` or an ensemble array of them."""
const FlavorOfTKEClosure{TD} = Union{TKEBasedTurbulenceClosure{TD}, TKEClosureArray{TD}} where TD

function TKEBasedTurbulenceClosure{TD}(mixing_length::ML, stability_functions::SF, static_stability::SS,
                                       minimum_tke::FT,
                                       shear::SH,
                                       negative_tke_damping_time_scale::FT,
                                       subgrid_distribution::SD = nothing,
                                       heating_budget_flux::HB = nothing) where {TD, ML, SF, SS, FT, SH, SD, HB}
    return TKEBasedTurbulenceClosure{TD, ML, SF, SS, FT, SH, SD, HB}(mixing_length, stability_functions, static_stability,
                                                                         minimum_tke,
                                                                         shear,
                                                                         negative_tke_damping_time_scale,
                                                                         subgrid_distribution, heating_budget_flux)
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`TKEBasedTurbulenceClosure`](@ref). The defaults are the calibrated configuration: a
[`GradientLimitedMixingLength`](@ref), [`TurbulentRichardsonStabilityFunction`](@ref), the [`DoubleGaussian`](@ref)
subgrid distribution with its latent static stability, and a [`HeatingBudgetFlux`](@ref).
"""
function TKEBasedTurbulenceClosure(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                   FT = Oceananigans.defaults.FloatType;
                                   mixing_length = GradientLimitedMixingLength(ℂˢ = 1.299067385616381, ℂⁱ = 1.5609936948461662),
                                   stability_functions = TurbulentRichardsonStabilityFunction(),
                                   subgrid_distribution = DoubleGaussian(FT),
                                   static_stability = default_static_stability(subgrid_distribution),
                                   minimum_tke = 1e-6,
                                   shear = GridMeanShear(),
                                   minimum_shear_squared = nothing,
                                   negative_tke_damping_time_scale = 1minute,
                                   heating_budget_flux = HeatingBudgetFlux(FT)) where TD

    shear = with_minimum_shear_squared(shear, minimum_shear_squared)
    validate_shear(shear)

    mixing_length = convert_eltype(FT, mixing_length)
    stability_functions = convert_eltype(FT, stability_functions)
    static_stability = convert_eltype(FT, static_stability)
    subgrid_distribution = convert_eltype(FT, subgrid_distribution)
    validate_subgrid_coupling(static_stability, subgrid_distribution)
    isnothing(subgrid_distribution) ||
        validate_variance_stability_functions(stability_functions, subgrid_distribution.variance_stability_functions)
    isnothing(subgrid_distribution) || validate_moment_time_scale(stability_functions, subgrid_distribution.time_scale)
    heating_budget_flux = convert_eltype(FT, heating_budget_flux)
    validate_heating_budget_flux(heating_budget_flux, static_stability, subgrid_distribution)

    return TKEBasedTurbulenceClosure{TD}(mixing_length,
                                         stability_functions,
                                         static_stability,
                                         convert(FT, minimum_tke),
                                         convert_eltype(FT, shear),
                                         convert(FT, negative_tke_damping_time_scale),
                                         subgrid_distribution,
                                         heating_budget_flux)
end

TKEBasedTurbulenceClosure(FT::DataType; kw...) =
    TKEBasedTurbulenceClosure(VerticallyImplicitTimeDiscretization(), FT; kw...)

default_static_stability(::Nothing) = MoistStaticStability()
default_static_stability(subgrid_distribution) = SaturationWeightedStaticStability(CondensateFluxFraction())

@inline convert_eltype(::Type{FT}, sf::ConstantStabilityFunctions) where FT =
    ConstantStabilityFunctions{FT}(convert(FT, sf.ℂᵘ), convert(FT, sf.ℂᶜ), convert(FT, sf.ℂᵉ), convert(FT, sf.ℂᴰ))
@inline convert_eltype(::Type{FT}, ss::DryStaticStability) where FT = ss
@inline convert_eltype(::Type{FT}, ss::MoistStaticStability) where FT = ss
@inline convert_eltype(::Type{FT}, ::Nothing) where FT = nothing
@inline convert_eltype(::Type{FT}, ss::SaturationWeightedStaticStability) where FT = ss

function absorb_stratification_coefficient(Cᴺ, sf::ConstantStabilityFunctions)
    mixing_length = LocalMinimumMixingLength(ℂˢ = 1 / Cᴺ)
    stability_functions = ConstantStabilityFunctions(ℂᵘ = Cᴺ * sf.ℂᵘ, ℂᶜ = Cᴺ * sf.ℂᶜ, ℂᵉ = Cᴺ * sf.ℂᵉ, ℂᴰ = sf.ℂᴰ / Cᴺ)
    return mixing_length, stability_functions
end

#####
##### Tracer wiring
#####
"""The name of the prognostic TKE tracer, which holds the density ``ρ e``."""
const TKE_NAME = :ρe

Oceananigans.TurbulenceClosures.closure_required_tracers(::FlavorOfTKEClosure) = tuple(TKE_NAME)

const TKEClosureOf{FT} = TKEBasedTurbulenceClosure{<:Any, <:Any, <:Any, <:Any, FT} where FT

AtmosphereBoundaryConditions.materialize_gustiness(::DefaultGustiness, ::TKEClosureOf{FT}) where FT =
    TKEGustiness(FT)

AtmosphereBoundaryConditions.materialize_gustiness(::DefaultGustiness, ::AbstractArray{<:TKEClosureOf{FT}}) where FT =
    TKEGustiness(FT)

function Utils.with_tracers(tracer_names, closure::FlavorOfTKEClosure)
    TKE_NAME ∈ tracer_names ||
        throw(ArgumentError("Tracers must contain :$(TKE_NAME) to represent turbulent kinetic " *
                            "energy for `TKEBasedTurbulenceClosure`."))
    return closure
end

#####
##### Closure fields
#####

struct TKEClosureFields{K, N, L, KC, LC, SG, CS, C}
    Kᵘ :: K # eddy diffusivity for momentum, at (Center, Center, Face)
    Kᶜ :: K # eddy diffusivity for scalars, at (Center, Center, Face)
    Kᵉ :: K # eddy diffusivity for turbulent kinetic energy, at (Center, Center, Face)
    N² :: N # static stability, at (Center, Center, Face)
    ℓᵐ :: N # the primary mixing length, at (Center, Center, Face)
    ℓˢ :: N # stratification-only envelope, before capping at the surface length
    ℓˢᶜ :: C
    Lᵉ :: L
    tupled_tracer_diffusivities :: KC         # per-tracer diffusivity lookup, by closure-scalar position
    tupled_implicit_linear_coefficients :: LC # `Lᵉ` for the TKE tracer, zero for every other
    subgrid_cloud :: SG # the fields of the subgrid cloud formulation, or `nothing`
    column_stability :: CS
end

Adapt.adapt_structure(to, fields::TKEClosureFields) =
    TKEClosureFields(adapt(to, fields.Kᵘ),
                     adapt(to, fields.Kᶜ),
                     adapt(to, fields.Kᵉ),
                     adapt(to, fields.N²),
                     adapt(to, fields.ℓᵐ),
                     adapt(to, fields.ℓˢ),
                     adapt(to, fields.ℓˢᶜ),
                     adapt(to, fields.Lᵉ),
                     adapt(to, fields.tupled_tracer_diffusivities),
                     adapt(to, fields.tupled_implicit_linear_coefficients),
                     adapt(to, fields.subgrid_cloud),
                     adapt(to, fields.column_stability))

BoundaryConditions.fill_halo_regions!(fields::TKEClosureFields, args...; kw...) =
    fill_halo_regions!((fields.Kᵘ, fields.Kᶜ, fields.Kᵉ), args...; kw...)

function Oceananigans.TurbulenceClosures.build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfTKEClosure)
    face_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()))
    default_bcs = (Kᵘ = face_bcs, Kᶜ = face_bcs, Kᵉ = face_bcs)
    bcs = merge(default_bcs, bcs)

    Kᵘ = ZFaceField(grid, boundary_conditions=bcs.Kᵘ)
    Kᶜ = ZFaceField(grid, boundary_conditions=bcs.Kᶜ)
    Kᵉ = ZFaceField(grid, boundary_conditions=bcs.Kᵉ)
    N² = ZFaceField(grid)
    ℓ = ZFaceField(grid)
    ℓˢ = ZFaceField(grid)
    ℓˢᶜ = CenterField(grid)
    Lᵉ = CenterField(grid)

    tracer_diffusivities = NamedTuple(name => name === TKE_NAME ? Kᵉ : Kᶜ for name in tracer_names)
    implicit_linear_coefficients = NamedTuple(name => name === TKE_NAME ? Lᵉ : ZeroField() for name in tracer_names)
    subgrid_cloud = build_subgrid_cloud_fields(grid, subgrid_distribution(closure), closure_static_stability(closure), tracer_names,
                                               heating_budget_flux(closure))

    column_stability = column_stability_field(grid, closure_stability_functions(closure))

    return TKEClosureFields(Kᵘ, Kᶜ, Kᵉ, N², ℓ, ℓˢ, ℓˢᶜ, Lᵉ, tracer_diffusivities, implicit_linear_coefficients, subgrid_cloud,
                            column_stability)
end

subgrid_distribution(closure::TKEBasedTurbulenceClosure) = closure.subgrid_distribution
subgrid_distribution(closure::TKEClosureArray) = first(closure).subgrid_distribution
closure_static_stability(closure::TKEBasedTurbulenceClosure) = closure.static_stability
closure_static_stability(closure::TKEClosureArray) = first(closure).static_stability
closure_stability_functions(closure::TKEBasedTurbulenceClosure) = closure.stability_functions
closure_stability_functions(closure::TKEClosureArray) = first(closure).stability_functions

column_stability_field(grid, stability_functions) = nothing

compute_column_stability!(::Nothing, grid, closure, N², model; parameters = :xyz) = nothing
compute_column_stability!(λ, grid, closure, N², model; parameters = :xyz) =
    compute_column_stability!(closure_stability_functions(closure), λ, grid, closure, N², model; parameters)

function return_to_isotropy_time_scaleᶜᶜᶠ end
@inline Oceananigans.TurbulenceClosures.viscosity_location(::FlavorOfTKEClosure) = (Center(), Center(), Face())
@inline Oceananigans.TurbulenceClosures.diffusivity_location(::FlavorOfTKEClosure) = (Center(), Center(), Face())

@inline Oceananigans.TurbulenceClosures.viscosity(::FlavorOfTKEClosure, fields) = fields.Kᵘ

@inline Oceananigans.TurbulenceClosures.diffusivity(::FlavorOfTKEClosure, fields, ::Val{id}) where id =
    fields.tupled_tracer_diffusivities[id]

@inline Oceananigans.TurbulenceClosures.implicit_linear_coefficient(i, j, k, grid,
                                                                    closure::FlavorOfTKEClosure{<:VerticallyImplicitTimeDiscretization},
                                                                    fields, ::Val{id}, args...) where id =
    @inbounds fields.tupled_implicit_linear_coefficients[id][i, j, k]

const ConstantStabilityClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:ConstantStabilityFunctions}

@inline shear_production_fraction(stability_functions) = 1

@inline diffusivity_stability_functionsᶜᶜᶠ(i, j, k, grid, closure, velocities, e, closure_fields) =
    (momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, closure_fields.N², e, closure_fields.ℓᵐ),
     tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, closure_fields.N², e, closure_fields.ℓᵐ),
     tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, closure_fields.N², e, closure_fields.ℓᵐ))

@inline momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.ℂᵘ
@inline tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.ℂᶜ
@inline tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.ℂᵉ
@inline dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.ℂᴰ
@inline dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure::ConstantStabilityClosure, args...) = closure.stability_functions.ℂᴰ

@inline face_valueᶜᶜᶠ(i, j, k, grid, field) = @inbounds field[i, j, k]

@inline inverse_buoyancy_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²) =
    inverse_buoyancy_length(closure.mixing_length, ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e), @inbounds(N²[i, j, k]))

@inline buoyancy_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²) = inv(inverse_buoyancy_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²))

@inline function local_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²)
    d = closure.mixing_length.ℂˢ * height_above_bottomᶜᶜᶠ(i, j, k, grid)
    λᵇ = inverse_buoyancy_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²)
    ℓ = inv(max(inv(d), λᵇ))
    return ifelse(isnan(ℓ), d, ℓ)
end

@inline function local_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, N²)
    d = closure.mixing_length.ℂˢ * height_above_bottomᶜᶜᶜ(i, j, k, grid)
    N²ᶜ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
    λᵇ = inverse_buoyancy_length(closure.mixing_length, turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e), N²ᶜ)
    ℓ = inv(max(inv(d), λᵇ))
    return ifelse(isnan(ℓ), d, ℓ)
end

@inline function fill_mixing_length!(ℓ, ℓˢ, i, j, grid, ::LocalMinimumMixingLength, closure, e, N²)
    for k in 1:grid.Nz+1
        @inbounds ℓ[i, j, k] = local_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²)
        @inbounds ℓˢ[i, j, k] = ℓ[i, j, k]
    end
    return nothing
end

@inline fill_mixing_length!(ℓ, ℓˢ, ℓˢᶜ, i, j, grid, ml, closure, e, N²) =
    fill_mixing_length!(ℓ, ℓˢ, i, j, grid, ml, closure, e, N²)

@inline function fill_mixing_length!(ℓ, ℓˢ, ℓˢᶜ, i, j, grid, ml::GradientLimitedMixingLength, closure, e, N²)
    FT = grid_float_type(grid)
    Nz = grid.Nz
    @inbounds begin
        ℓˢ[i, j, 1] = FT(Inf)
        for k in 2:Nz+1
            ℓᵇ = buoyancy_lengthᶜᶜᶠ(i, j, k, grid, closure, e, N²)
            ℓˢ[i, j, k] = ifelse(isnan(ℓᵇ), FT(Inf), ℓᵇ)
        end
        ℓᵈᵒʷⁿ = ℓˢ[i, j, Nz+1]
        for k in Nz:-1:1
            ℓˢᶜ[i, j, k] = ℓᵈᵒʷⁿ
            ℓᵈᵒʷⁿ = min(ℓˢ[i, j, k], ℓᵈᵒʷⁿ + ml.ℂⁱ * Δzᶜᶜᶜ(i, j, k, grid))
        end
        for k in 2:Nz+1
            ℓˢ[i, j, k] = min(ℓˢ[i, j, k], ℓˢ[i, j, k-1] + ml.ℂⁱ * Δzᶜᶜᶜ(i, j, k-1, grid))
        end
        for k in 1:Nz
            Δℓ = ml.ℂⁱ * Δzᶜᶜᶜ(i, j, k, grid) / 2
            ℓˢᶜ[i, j, k] = min(ℓˢ[i, j, k] + Δℓ, ℓˢᶜ[i, j, k] + Δℓ)
        end
        for k in Nz:-1:1
            ℓˢ[i, j, k] = min(ℓˢ[i, j, k], ℓˢ[i, j, k+1] + ml.ℂⁱ * Δzᶜᶜᶜ(i, j, k, grid))
        end
        ℓ[i, j, 1] = 0
        for k in 2:Nz+1
            ℓ[i, j, k] = min(ℓˢ[i, j, k], ml.ℂˢ * height_above_bottomᶜᶜᶠ(i, j, k, grid))
        end
    end
    return nothing
end

@inline function crossing_distance(Δb, N², Δz, r)
    s = 2r / (Δb + sqrt(max(Δb^2 + 2 * N² * r, 0)))
    return clamp(s, 0, Δz)
end

@inline function parcel_ascent(i, j, k, grid, N², e★)
    FT = grid_float_type(grid)
    Nz = grid.Nz
    s = zero(FT); D = zero(FT); Δb = zero(FT)
    m = k
    @inbounds while m ≤ Nz && D < e★
        Δz = Δzᶜᶜᶜ(i, j, m, grid)
        N²ₘ = (N²[i, j, m] + N²[i, j, m+1]) / 2
        cost = Δb * Δz + N²ₘ * Δz^2 / 2
        stops = D + cost ≥ e★
        s += ifelse(stops, crossing_distance(Δb, N²ₘ, Δz, e★ - D), Δz)
        D = ifelse(stops, e★, D + cost)
        Δb += N²ₘ * Δz
        m += 1
    end
    return ifelse(D < e★, FT(Inf), s) # out the top with energy to spare: no obstacle above
end

@inline function parcel_descent(i, j, k, grid, N², e★)
    FT = grid_float_type(grid)
    s = zero(FT); D = zero(FT); Δb = zero(FT)
    m = k - 1
    @inbounds while m ≥ 1 && D < e★
        Δz = Δzᶜᶜᶜ(i, j, m, grid)
        N²ₘ = (N²[i, j, m] + N²[i, j, m+1]) / 2
        cost = Δb * Δz + N²ₘ * Δz^2 / 2
        stops = D + cost ≥ e★
        s += ifelse(stops, crossing_distance(Δb, N²ₘ, Δz, e★ - D), Δz)
        D = ifelse(stops, e★, D + cost)
        Δb += N²ₘ * Δz
        m -= 1
    end
    return s # the ground stops what the stratification does not
end

@inline function fill_mixing_length!(ℓ, ℓˢ, i, j, grid, ml::IntegralMixingLength, closure, e, N²)
    ℂˢ = ml.ℂˢ
    for k in 1:grid.Nz+1
        w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
        e★ = w★^2 / (2 * ℂˢ^2)
        ascent = parcel_ascent(i, j, k, grid, N², e★)
        descent = parcel_descent(i, j, k, grid, N², e★)
        @inbounds ℓ[i, j, k] = ℂˢ * min(ascent, descent)
        @inbounds ℓˢ[i, j, k] = ℓ[i, j, k]
    end
    return nothing
end

@kernel function _compute_mixing_length!(ℓ, ℓˢ, ℓˢᶜ, grid, closure, e, N²)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    fill_mixing_length!(ℓ, ℓˢ, ℓˢᶜ, i, j, grid, closure_ij.mixing_length, closure_ij, e, N²)
end

compute_mixing_length!(ℓ, ℓˢ, ℓˢᶜ, grid, closure, e, N²; parameters = :xyz) =
    launch!(grid.architecture, grid, column_kernel_parameters(parameters),
            _compute_mixing_length!, ℓ, ℓˢ, ℓˢᶜ, grid, closure, e, N²)

column_kernel_parameters(::Symbol) = :xy
column_kernel_parameters(::KernelParameters{S, O}) where {S, O} = KernelParameters(S[1:2], O[1:2])

@inline mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields) =
    mixing_lengthᶜᶜᶜ(i, j, k, grid, closure.mixing_length, closure, e, closure_fields)

@inline mixing_lengthᶜᶜᶜ(i, j, k, grid, ::LocalMinimumMixingLength, closure, e, closure_fields) =
    local_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields.N²)

@inline function mixing_lengthᶜᶜᶜ(i, j, k, grid, ml::GradientLimitedMixingLength, closure, e, closure_fields)
    ℓᶠ = @inbounds closure_fields.ℓˢᶜ[i, j, k]
    return min(ℓᶠ, local_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields.N²))
end

@inline function mixing_lengthᶜᶜᶜ(i, j, k, grid, ml::AbstractMixingLength, closure, e, closure_fields)
    ℓˢ = closure_fields.ℓˢ
    ℓᶠ = @inbounds min(ℓˢ[i, j, k], ℓˢ[i, j, k+1]) + interior_slope(ml) * Δzᶜᶜᶜ(i, j, k, grid) / 2
    return min(ℓᶠ, local_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields.N²))
end

#####
##### The closure with the microphysics of a model
#####

#####
##### Diffusivities
#####

@kernel function _compute_tke_static_stability!(N², grid, closure, tracers, buoyancy, closure_fields)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    @inbounds N²[i, j, k] = static_stabilityᶜᶜᶠ(i, j, k, grid, closure_ij.static_stability, buoyancy, tracers, closure_fields)
end

@kernel function _compute_tke_closure_fields!(closure_fields, grid, closure, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    e = tracers[TKE_NAME]

    ℓᵐ = closure_fields.ℓᵐ
    ℓ = @inbounds ℓᵐ[i, j, k]
    K = ℓ * ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij, e)

    𝕊ᵘ, 𝕊ᶜ, 𝕊ᵉ = diffusivity_stability_functionsᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, e, closure_fields)

    Kᵘ = 𝕊ᵘ * K
    Kᶜ = 𝕊ᶜ * K
    Kᵉ = 𝕊ᵉ * K

    FT = grid_float_type(grid)
    @inbounds begin
        closure_fields.Kᵘ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(Kᵘ))
        closure_fields.Kᶜ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(Kᶜ))
        closure_fields.Kᵉ[i, j, k] = mask_diffusivity(i, j, k, grid, FT(Kᵉ))
    end
end

@inline shear_productionᶜᶜᶠ(i, j, k, grid, Kᵘ, u, v) = @inbounds Kᵘ[i, j, k] * shearᶜᶜᶠ(i, j, k, grid, u, v)

@inline buoyancy_productionᶜᶜᶠ(i, j, k, grid, Kᶜ, N²) = @inbounds -Kᶜ[i, j, k] * N²[i, j, k]

@inline function dissipationᶜᶜᶜ(i, j, k, grid, closure, e, velocities, closure_fields)
    eᵢ = @inbounds e[i, j, k]
    ℓ = mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields)
    𝕊ᴰ = dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure, velocities, closure_fields.N², e, closure_fields.ℓᵐ)
    turbulent = eᵢ > 0
    e₊ = ifelse(turbulent, eᵢ, one(eᵢ))
    return ifelse(turbulent, 𝕊ᴰ * e₊ * sqrt(e₊) / ℓ, zero(eᵢ))
end

@inline function tke_sink_rate(i, j, k, grid, closure, e, B, velocities, closure_fields)
    eᵢ = @inbounds e[i, j, k]
    ℓ = mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, closure_fields)
    𝕊ᴰ = dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure, velocities, closure_fields.N², e, closure_fields.ℓᵐ)

    τ = closure.negative_tke_damping_time_scale
    ω = ifelse(eᵢ < 0, 1 / τ, 𝕊ᴰ * sqrt(abs(eᵢ)) / ℓ)

    B⁻ = min(0, B)
    ωᴮ = -B⁻ / buoyancy_sink_tke(i, j, k, grid, closure_fields.subgrid_cloud, closure, e)

    return ω + ωᴮ
end

@inline tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, closure_fields) =
    tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, closure_fields.subgrid_cloud, closure_fields)

@inline tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, ::Nothing, closure_fields) =
    ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, closure_fields.Kᶜ, closure_fields.N²)

@kernel function _compute_tke_implicit_linear_coefficient!(Lᵉ, grid, closure, closure_fields, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    e = tracers[TKE_NAME]

    B = tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, closure_fields)
    ω = tke_sink_rate(i, j, k, grid, closure_ij, e, B, velocities, closure_fields)
    active = !inactive_cell(i, j, k, grid)

    @inbounds Lᵉ[i, j, k] = - ω * active
end

function Oceananigans.TurbulenceClosures.compute_closure_fields!(closure_fields,
                                                         closure::FlavorOfTKEClosure,
                                                         model; parameters = :xyz)
    grid = model.grid
    arch = grid.architecture
    tracers = Oceananigans.TurbulenceClosures.buoyancy_tracers(model)

    compute_stability_and_diffusivities!(closure_fields, closure, model, tracers, parameters)

    diagnose_third_moment!(closure_fields.subgrid_cloud, subgrid_distribution(closure), closure_fields, closure, model, tracers, parameters)

    compute_subgrid_cloud!(closure_fields.subgrid_cloud, closure_fields, closure, model, tracers, parameters)
    compute_applied_flux_conversions!(closure_fields.subgrid_cloud, closure_fields, closure, model, tracers)
    compute_tke_budget!(closure_fields.subgrid_cloud, subgrid_distribution(closure), closure_fields, closure, model, tracers)

    launch!(arch, grid, parameters, _compute_tke_implicit_linear_coefficient!,
            closure_fields.Lᵉ, grid, closure, closure_fields, model.velocities, tracers)

    return nothing
end

function compute_stability_and_diffusivities!(closure_fields, closure, model, tracers, parameters)
    grid = model.grid
    arch = grid.architecture
    buoyancy = Oceananigans.TurbulenceClosures.buoyancy_force(model)

    launch!(arch, grid, parameters, _compute_tke_static_stability!,
            closure_fields.N², grid, closure, tracers, buoyancy, closure_fields)

    compute_mixing_length_and_diffusivities!(closure_fields, closure, model, tracers, parameters)

    return nothing
end

function compute_mixing_length_and_diffusivities!(closure_fields, closure, model, tracers, parameters)
    grid = model.grid
    compute_mixing_length!(closure_fields.ℓᵐ, closure_fields.ℓˢ, closure_fields.ℓˢᶜ, grid, closure, tracers[TKE_NAME],
                           closure_fields.N²; parameters)
    compute_column_stability!(closure_fields.column_stability, grid, closure, closure_fields.N², model; parameters)
    launch!(grid.architecture, grid, parameters, _compute_tke_closure_fields!, closure_fields, grid, closure, model.velocities, tracers)
    return nothing
end

compute_subgrid_cloud!(::Nothing, closure_fields, closure, model, tracers, parameters) = nothing

#####
##### The TKE equation: sources in the stage tendency
#####

@inline explicit_tke_sinks(i, j, k, grid, ::TKEBasedTurbulenceClosure{<:VerticallyImplicitTimeDiscretization},
                           closure_fields, e) = zero(grid)

@inline explicit_tke_sinks(i, j, k, grid, ::TKEBasedTurbulenceClosure{<:ExplicitTimeDiscretization},
                           closure_fields, e) = @inbounds closure_fields.Lᵉ[i, j, k] * e

@kernel function _add_tke_tendencies!(Gρe, grid, closure, closure_fields, velocities, ρe, ρ, injected)
    i, j, k = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)

    u, v = velocities.u, velocities.v
    P = shear_production_fraction(closure_ij.stability_functions) *
        explicit_tke_shear_productionᶜᶜᶜ(i, j, k, grid, closure_fields.subgrid_cloud, closure_fields, u, v, injected)
    B = tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, closure_fields)
    B⁺ = explicit_tke_buoyancy_production(closure_fields.subgrid_cloud, i, j, k, B, injected)

    ρᵢ = @inbounds ρ[i, j, k]
    e = @inbounds ρe[i, j, k] / ρᵢ
    sinks = explicit_tke_sinks(i, j, k, grid, closure_ij, closure_fields, e)

    T = tke_budget_transportᶜᶜᶜ(i, j, k, grid, closure_ij.subgrid_distribution, closure_fields.subgrid_cloud)

    @inbounds Gρe[i, j, k] += ρᵢ * (P + B⁺ + T + sinks)
end

function AtmosphereModels.compute_closure_tendencies!(Gⁿ, closure_fields, closure::FlavorOfTKEClosure, model)
    grid = model.grid
    arch = grid.architecture
    ρ = AtmosphereModels.total_density(model.dynamics)

    launch!(arch, grid, :xyz, _add_tke_tendencies!,
            Gⁿ[TKE_NAME], grid, closure, closure_fields,
            model.velocities, model.tracers[TKE_NAME], ρ, solved_conversions_before_tke_solve(closure_fields.subgrid_cloud, closure, model))

    add_dissipative_heating!(Gⁿ, closure_fields.subgrid_cloud, closure_fields, closure, model)
    add_shear_production_heating!(Gⁿ, closure_stability_functions(closure), closure_fields.subgrid_cloud, closure_fields, closure, model)
    draw_latent_conversion!(Gⁿ, closure_fields.subgrid_cloud, closure, model)
    add_heating_budget_flux_tendency!(Gⁿ, closure_fields.subgrid_cloud, heating_budget_flux(closure), model)

    return nothing
end

#####
##### Show
#####

function Base.summary(closure::TKEBasedTurbulenceClosure)
    TD = nameof(typeof(time_discretization(closure)))
    return string("TKEBasedTurbulenceClosure{$TD}")
end

function Base.show(io::IO, closure::TKEBasedTurbulenceClosure)
    print(io, summary(closure), '\n', "├── mixing_length: ")
    show(io, closure.mixing_length)
    print(io, '\n', "├── stability_functions: ", summary(closure.stability_functions), '\n')
    show_stability_function_lines(io, closure.stability_functions, "│   ")
    print(io, '\n',
              "├── static_stability: ", summary(closure.static_stability), '\n',
              "├── minimum_tke: ", prettysummary(closure.minimum_tke), '\n',
              "├── shear: ")
    show(io, closure.shear)
    print(io, '\n',
              "├── negative_tke_damping_time_scale: ", prettysummary(closure.negative_tke_damping_time_scale), '\n',
              "├── subgrid_distribution: ", summary(closure.subgrid_distribution), "\n")
    isnothing(closure.heating_budget_flux) || print(io, "├── heating_budget_flux: ", summary(closure.heating_budget_flux), "\n")
end

Oceananigans.prognostic_state(::TKEClosureFields) = nothing
Oceananigans.restore_prognostic_state!(closure_fields::TKEClosureFields, ::Nothing) = closure_fields
